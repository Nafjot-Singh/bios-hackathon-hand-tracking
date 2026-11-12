# HandTrace

**BIOS Hackathon | Forge Industry Track | MyTron Labs Challenge**
*Turning head-cam factory video into tracked hands, hand-object contact, and candidate task events, with evidence frames.*

---

## 1. Pipeline

```
video --+--> hand detection (MediaPipe) -> hand tracking -> smoothing -> hands.jsonl
        |
        +--> object detection (YOLO-World) + tracking (ByteTrack)
                              |
   hands + objects ----> contact cues -> state machine -> contacts.csv
                                                      \-> events_proposals.json + evidence_frames/
   hands.jsonl ----> metrics -> graphs -> report.html
```

One command runs all of it: `python main.py`.

| Stage | Problem statement | Code |
| :--- | :--- | :--- |
| Hand tracking and quality | PS1 Hand Tracking | `detector/`, `tracker/`, `utils/`, `quality/` |
| Objects and contact | PS2 Contact Estimation | `objects/`, `contact/` |
| Evaluation and events | PS3 Task Timeline | `member3/` (not yet in this repository, see section 9) |

Primary declared track: **PS5 Open Innovation**, built on PS1 and PS2 and supplying candidate intervals to PS3.

## 2. Hard rules

1. **No hard-coding.** Distances, frame rates, thresholds and model settings live in `config.py`.
2. **No fake results.** A hand that is not detected is not written and its landmarks are never invented (so there is no interpolation). Missing hands, or a Tracking Quality Score below 0.40, give `unknown`.
3. **Hardware context.** The device configuration is recorded with every run.
4. **Frozen schemas.** Contact records are Pydantic models (`contact/schemas.py`) and are validated before they are written.
5. **Data privacy.** No video, extracted frames, API keys or personal paths are committed (see `.gitignore`).
6. **Multi-cue contact.** Bounding-box overlap alone is never contact (see section 6).

## 3. Setup

```powershell
git clone https://github.com/tandon-divyansh/HandTrace-Member2.git
cd HandTrace-Member2

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt          # everything
pip install -r requirements-core.txt     # hands only (then set HT_OBJECTS=0)
```

Linux/macOS: `source .venv/bin/activate`. For GPU, install a CUDA build of PyTorch first (see `requirements.txt`).

On the first run with objects enabled, Ultralytics downloads the YOLO-World weights (`yolov8s-worldv2.pt`) and any extra files it needs for open-vocabulary prompts, so the machine needs internet access once.

| What | Where | Notes |
| :--- | :--- | :--- |
| Hand model | `models/hand_landmarker.task` | Included. Override with `HT_MODEL`. |
| Input video | `input/demo.mp4` | Not committed. Override with `HT_VIDEO`. |
| API keys | `.env` | Gitignored. Used only by event naming. |

## 4. Run

```powershell
python main.py                        # hands + objects + contact + metrics + report
python main.py --max-frames 100       # quick test
$env:HT_OBJECTS = "0"; python main.py # hands only (no YOLO, no contact stage)
$env:HT_DEBUG = "0"; python main.py   # progress and summary only
python -m pytest                      # 38 tests, no video or model needed
```

`main.py` handles **one clip per run** and overwrites `outputs/`. Run it once per clip (change `HT_VIDEO`) and keep or merge the outputs yourself.

### Outputs (`outputs/`)

| File | Content |
| :--- | :--- |
| `hands.jsonl` | One line per visible hand per frame |
| `debug_video.mp4` | Skeleton, landmarks, box, centroid, ID, handedness, confidence, and detected objects |
| `contacts.csv` | One row per contact interval |
| `events_proposals.json` | Candidate events for review and naming |
| `evidence_frames/<clip>_<frame>.png` | Frames with hands and objects drawn on them |
| `metrics.json`, `graphs/*.png`, `report.html` | Tracking quality metrics, graphs and a one-file report |

Rebuild only the report: `python -m quality.report`.

## 5. Configuration (`config.py`)

| Setting | Default | Environment variable |
| :--- | :--- | :--- |
| Input video / hand model | `input/demo.mp4`, `models/hand_landmarker.task` | `HT_VIDEO`, `HT_MODEL` |
| Max hands | 2 | `HT_MAX_HANDS` |
| Max track distance (px) / max missing frames | 80 / 5 | `HT_MAX_DISTANCE` |
| Smoothing alpha (1 = off) | 0.5 | `HT_SMOOTHING_ALPHA` |
| Objects and contact stage | on | `HT_OBJECTS` |
| YOLO-World weights | `yolov8s-worldv2.pt` | `HT_YOLO_WEIGHTS` |
| Object confidence / device | 0.25 / auto | `HT_OBJECT_CONF`, `HT_DEVICE` |
| Debug logging / debug video / fallback FPS | on / on / 30 | `HT_DEBUG`, `HT_SAVE_DEBUG_VIDEO`, `HT_FPS` |

Object prompts (`Objects.portable_classes`, `Objects.fixed_classes`) and every contact threshold (`Contact`) are in `config.py`.

## 6. How objects and contact work

**Objects.** YOLO-World detects the prompted classes and Ultralytics' ByteTrack gives each object a persistent ID. Every class is either *portable* (tool, bottle, ...) or *fixed* (workbench, ...).

**Per-frame decision** (`contact/cues.py`). For each visible hand and each object:

| Cue | Meaning |
| :--- | :--- |
| Fingertip distance | Nearest fingertip to the object box, divided by hand scale (wrist to middle-finger base). **Required.** |
| Palm distance | Palm center to the box, same scaling |
| Overlap | Share of the hand box inside the object box |

A fingertip must be close **and** palm distance or overlap must agree; the three cues are then fused into a score that must pass a threshold. A portable object beats a fixed surface. A "fixed surface" covering more than 70% of the frame is rejected (`conflicting_cues`). A hand with Tracking Quality Score below 0.40 is `unknown` (`low_quality`). The score is handedness confidence times the share of landmarks inside the image.

**Smoothing** (`contact/state_machine.py`). A state is accepted only after it lasts long enough (3 frames for nothing and contact states, 2 for unknown), short runs are absorbed, boundaries are back-dated to where the run began, and a change of object ID starts a new interval. Missing hands become `unknown` with reason `gap` (up to 4 frames) or `no_hand` (longer).

**Naming.** Hands are named `<side>_<n>`: `left_0`, `right_0`, and `left_1` if a second left track appears (for example after an ID switch).

**Events.** A portable-object interval of at least 0.4 s becomes an event proposal with up to 3 evidence frames, `review_status: pending` and `action: null` (event naming fills it in). Long unknown intervals (1 s or more) also get evidence frames for the failure audit.

## 7. Output schemas

**`hands.jsonl`:** `frame, timestamp_ms, track_id, handedness, confidence, centroid{x,y}, landmarks[21]{x,y,z}` (landmarks normalized 0-1 and smoothed; centroid in pixels).

**`contacts.csv`:** `clip, t_start, t_end, hand, state, object_name, object_id, persistence_frames, evidence_frames, unknown_reason`. The first nine columns are the shared contract; `unknown_reason` (`gap`, `no_hand`, `low_quality`, `conflicting_cues`) is an added last column. Times are seconds and end-exclusive, so intervals tile. `evidence_frames` lists frame indices separated by `|`.

**`events_proposals.json`:** a list of records with `clip_id, event_id, start_sec, end_sec, hand_track_id, hand_side, object_id, object_label, contact_state, tracking_quality, evidence_frames, review_status, action, trigger_reason`.

Other shared files: `failure_log.md` (`clip, timestamp_range, symptom, suspected_cause, mitigation`), `outputs/metrics_report.md` (precision, recall, F1, unknown share, ID persistence, boundary error) and `evaluation/labels_template.csv` (`clip, t_start, t_end, hand, state, object`).

## 8. Reported results and hardware (from the team's earlier runs)

These numbers come from the earlier objects-and-contact work and were **not** reproduced with this code.

- **Hardware:** NVIDIA GeForce RTX 2050 Laptop GPU (4,096 MiB), CUDA 12.6, cuDNN 91002, driver 610.78, Python 3.10.4, PyTorch 2.14.1+cu126.
- **Working setup:** 960x540 letterboxed from 1920x1080, every 2nd frame of the 30 FPS source (15 FPS); YOLO-World `yolov8s-worldv2.pt` with ByteTrack; 17.7-19.1 FPS end to end.
- **Demo clips:** 1,261 contact intervals, 29 candidate events, 63.1% of intervals `unknown` (796 / 1,261).
- **Not measured (no human labels yet):** contact accuracy, per-class precision / recall / F1, 25-frame challenge accuracy, event boundary error, object ID switches. Assistant-scaffolded reference files were invalidated (`agent_scaffold_INVALID_*`).
- **Failure audit:** see [`failure_log.md`](failure_log.md) (6 cases, flicker 124-272 intervals per minute, every root cause still **unverified**).

Tracking quality for hands (detection rate, tracks, gaps, estimated ID switches) is produced on every run in `outputs/report.html`.

## 9. Known limits and open items

| # | Item |
| :-- | :--- |
| 1 | **The objects-and-contact stage was written from the earlier README and failure log, not from the original code.** Thresholds are untuned defaults, the Tracking Quality Score is the simple formula above (the evaluation workstream's `3_tqs_engine.py` may define it differently), and the object prompts in `config.py` are a starting list. Tune them on real clips. |
| 2 | **Not run with the real MediaPipe and YOLO-World models in the build environment.** It was tested with stand-ins for both (38 tests plus an end-to-end run). The first real run is the real test. |
| 3 | **One clip per run.** `contacts.csv` has a `clip` column for multi-clip use, but `main.py` overwrites the files each run. |
| 4 | **Frame rate.** The earlier results used 15 FPS (every 2nd frame). This pipeline processes every frame of the input. `persistence` settings are in frames, so they cover half the time at 30 FPS; adjust them or pre-sample the video. `Tracking.max_track_distance` (pixels) also depends on resolution and frame rate. |
| 5 | **Evaluation and event naming code** (`member3/`, VLM naming, metrics against human labels) is not in this repository yet. |
| 6 | **Tracker name.** An earlier README said BoT-SORT; `failure_log.md` says ByteTrack, not BoT-SORT. ByteTrack is used. Confirm. |
| 7 | **Dependency versions.** The earlier work used MediaPipe 0.10.21 with `numpy<2`; the hand-tracking environment used MediaPipe 1.2.0 with NumPy 2.5.4. `requirements.txt` uses lower bounds only. |
| 8 | **Folders unified** on `outputs/` and `models/`. |

## 10. Library inventory

| Library | Used for |
| :--- | :--- |
| MediaPipe | 21-joint hand landmarks and handedness (Hand Landmarker, VIDEO mode) |
| Ultralytics (YOLO-World, ByteTrack), PyTorch | Open-vocabulary object detection and object tracking |
| OpenCV | Video I/O, annotation, evidence frames |
| NumPy, Matplotlib | Arrays, graphs |
| Pydantic | Frozen output schemas |
| pandas, tqdm, google-genai | Evaluation and event naming |
| pytest | Automated tests |
