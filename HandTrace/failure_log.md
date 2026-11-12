# HandTrace — Failure Log & Hard Cases Audit

**Workstream**: Member 2 (Objects & Contact) & Member 3 (Evaluation & Events)  
**Schema Version**: `0.1.0-provisional`  
**Tracker Used**: ByteTrack (`bytetrack.yaml` via Ultralytics; not BoT-SORT)  
**Working Resolution**: 960x540 letterboxed from original 1920x1080  
**Purpose**: Rigorous, data-driven documentation of edge cases, detection failures, and ambiguous contact moments across real factory video. Every failure mode is cited with exact clip ID, timestamp, CSV rows in `outputs/contacts.csv`, and a saved frame in `outputs/evidence_frames/`.

---

## 1. Failure & Ambiguity Log

| Clip ID | Timestamp / Window | CSV Rows (`outputs/contacts.csv`) | Saved Evidence Frame | Symptom (Observed Behavior) | Root Cause Status | System Action / Mitigation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `MTL-SGF00106-021` | `73.533s - 75.400s` | Row 21 (`right_0`), Row 151 (`left_0`) | `outputs/evidence_frames/MTL-SGF00106-021_2206.png` | Missing hand evidence across 56 original frames (1.867s gap). | **unverified** (pending user review of frame) | `max_gap_frames: 4` splits the interval, terminates previous state at 73.533s, and logs an explicit `unknown` interval with `unknown_reason: gap`. |
| `MTL-SGF00106-021` | `86.867s - 89.467s` | Row 50 (`right_0`), Row 181 (`left_0`) | `outputs/evidence_frames/MTL-SGF00106-021_2640.png` | Hand track marked `unknown` for 2.600s. | **unverified** (pending user review of frame) | System strictly refused to hallucinate position; logged as `unknown` with `unknown_reason: no_hand`. |
| `MTL-SGF00106-021` | `105.000s - 105.867s` | Rows 95–101 (`right_0`), Rows 220–227 (`left_0`) | `outputs/evidence_frames/MTL-SGF00106-021_3154.png` | Rapid alternating flip-flops between `nothing` and `unknown` in 0.867s window. | **unverified** (pending user review of frame) | Controlled by `unknown_enter_persistence_frames: 2`, `unknown_exit_persistence_frames: 2`, and `min_state_duration_frames: 3`. |
| `MTL-SKL00089-011` | `17.533s - 18.067s` | Row 306 (`right_0`), Row 457 (`left_0`) | `outputs/evidence_frames/MTL-SKL00089-011_526.png` | Both `right_0` and `left_0` registered contact with `handheld tool` (id=55) simultaneously. | **unverified** (pending user review of frame) | Both hand contacts registered; produces dual-hand candidate events in `outputs/events_proposals.json`. |
| `mytronlabs-000003-006` | `8.800s - 9.133s` | Row 578 (`left_0`), Row 843 (`right_0`) | `outputs/evidence_frames/mytronlabs-000003-006_270.png` | YOLO detected full-frame `workbench` $[12, 0, 923, 540]$ covering 96% of frame. | **unverified** (pending user review of frame) | Mitigated by `max_fixed_surface_area_ratio: 0.70` which rejected physical contact and assigned `unknown` with `unknown_reason: conflicting_cues`. |
| `mytronlabs-000060-010` | `50.267s - 56.733s` | Rows 1106–1109 (`left_0`), Rows 1248–1251 (`right_0`) | `outputs/evidence_frames/mytronlabs-000060-010_1635.png` | Both hands completely absent from camera field of view for >6 seconds. | **unverified** (pending user review of frame) | System recorded absence honestly without synthesising landmarks; logged as `unknown` with `unknown_reason: no_hand` and `gap`. |

---

## 2. Accuracy & Evaluation Status

- **Contact Accuracy**: **not measured: no human labels yet**
- **Per-Class Precision / Recall / F1**: **not measured: no human labels yet**
- **25-Frame Challenge Accuracy**: **not measured: no human labels yet**
- **Event Boundary Error**: **not measured: no human labels yet**
- **Ground Truth Note**: All assistant-scaffolded reference and grading files have been marked invalid with `agent_scaffold_INVALID_*` prefixes. Zero metrics derived from assistant labels are reported.
- **ID Switches**: **not measured: no human labels yet** (pending human ground-truth review CSV).

---

## 3. Real Flicker Rates (Transitions per Minute)

Flicker is measured as total state transitions divided by active duration in minutes:

| Clip ID | Hand | Active Duration | State Intervals | Measured Flicker (Transitions / Min) |
| :--- | :--- | :--- | :--- | :--- |
| `MTL-SGF00106-021` | `left_0` | 56.33s (0.939 min) | 122 | **129.9** |
| `MTL-SGF00106-021` | `right_0` | 56.33s (0.939 min) | 129 | **137.4** |
| `MTL-SKL00089-011` | `left_0` | 59.87s (0.998 min) | 124 | **124.3** |
| `MTL-SKL00089-011` | `right_0` | 59.87s (0.998 min) | 173 | **173.4** |
| `mytronlabs-000003-006` | `left_0` | 59.87s (0.998 min) | 271 | **271.6** |
| `mytronlabs-000003-006` | `right_0` | 59.87s (0.998 min) | 140 | **140.3** |
| `mytronlabs-000060-010` | `left_0` | 59.00s (0.983 min) | 163 | **165.8** |
| `mytronlabs-000060-010` | `right_0` | 59.00s (0.983 min) | 139 | **141.4** |
