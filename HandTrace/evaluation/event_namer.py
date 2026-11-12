"""
event_namer.py

Validates candidate events and names human-verifiable actions using Groq AI.
Default model: openai/gpt-oss-120b
Generates:
  - outputs/events_validated.json
  - outputs/actions.csv

Usage:
    python event_namer.py
    python event_namer.py --api-key gsk_...
    python event_namer.py --model openai/gpt-oss-120b
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import cv2
from groq import Groq

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from config import CONFIG

log = logging.getLogger("event_namer")

DEFAULT_MODEL = "openai/gpt-oss-120b"

FIXED_VERBS = {
    "pick",
    "place",
    "grasp",
    "release",
    "fasten",
    "inspect",
    "hold",
    "position",
    "press",
    "adjust",
}


def load_dotenv_fallback() -> None:
    """Fallback .env reader if python-dotenv is not installed or placed in venv."""
    candidates = [
        Path(".env"),
        CONFIG.paths.root / ".env",
        CONFIG.paths.root.parent / ".env",
        CONFIG.paths.root / "venv" / ".env",
    ]
    for c in candidates:
        if c.is_file():
            try:
                for line in c.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip('"').strip("'")
                        if k and k not in os.environ:
                            os.environ[k] = v
            except Exception:
                pass


load_dotenv_fallback()


def sample_evidence_frames(
    video_path: str | Path,
    start_s: float,
    end_s: float,
    evidence_dir: str | Path,
) -> tuple[list[int], list[str]]:
    """Sample up to 3 frames across an event interval from the video."""
    evidence_path = Path(evidence_dir)
    evidence_path.mkdir(parents=True, exist_ok=True)

    video_str = str(video_path)
    if not os.path.exists(video_str):
        return [int(start_s * 30), int(end_s * 30)], []

    cap = cv2.VideoCapture(video_str)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    clip_id = Path(video_path).stem

    dur = max(end_s - start_s, 0.5)
    timestamps = [start_s + dur * 0.2, start_s + dur * 0.5, start_s + dur * 0.8]
    frame_ids, img_paths = [], []

    for ts in timestamps:
        f_idx = int(round(ts * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ok, frame = cap.read()
        if ok:
            frame_ids.append(f_idx)
            p = str(evidence_path / f"{clip_id}_frame_{f_idx}.jpg")
            cv2.imwrite(p, frame)
            img_paths.append(p)
    cap.release()
    return frame_ids, img_paths


def _clean_json_text(text: str) -> dict[str, Any]:
    """Safely extracts JSON from responses that may include markdown code fences."""
    raw = text.strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    elif raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    return json.loads(raw.strip())


def _encode_image(image_path: str) -> str:
    """Encode an image as base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def query_llm_with_cache(
    clip_id: str,
    prop: dict[str, Any],
    img_paths: list[str],
    cache_dir: str | Path,
    api_key: str | None = None,
    model_name: str | None = None,
) -> tuple[dict[str, Any], bool]:
    """Query Groq AI with caching."""
    cache_path_dir = Path(cache_dir)
    cache_path_dir.mkdir(parents=True, exist_ok=True)

    key_str = f"{clip_id}_{prop['start_sec']}_{prop['end_sec']}_{prop['hand_side']}"
    cache_file = cache_path_dir / f"{hashlib.md5(key_str.encode()).hexdigest()}.json"

    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8")), False
        except Exception:
            pass

    fallback = {
        "action": "interaction",
        "object_label": prop.get("object_label", "unknown"),
        "review_status": "needs_review",
    }

    effective_key = (
        api_key
        or os.environ.get("GROQ_API_KEY")
        or os.environ.get("GROQ_KEY")
        or os.environ.get("GEMINI_API_KEY")
    )
    if float(prop.get("tracking_quality", 1.0)) < 0.55 or not effective_key:
        return fallback, False

    selected_model = model_name or os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
    client = Groq(api_key=effective_key)

    dur = max(float(prop.get("end_sec", 0.0)) - float(prop.get("start_sec", 0.0)), 0.1)
    prompt_text = (
        f"First-person factory worker video event:\n"
        f"- Hand: {prop['hand_side']}\n"
        f"- Contact object: {prop.get('object_label', 'unknown')}\n"
        f"- Contact state: {prop.get('contact_state', 'unknown')}\n"
        f"- Time window: {prop['start_sec']}s - {prop['end_sec']}s ({dur:.2f}s duration)\n\n"
        f"Task: Classify this task action.\n"
        f"Pick the single best action verb from this list ONLY: {sorted(list(FIXED_VERBS))}.\n"
        f"Identify the object being manipulated.\n"
        'Return strict JSON in this format: {"action": "<verb>", "object_label": "<object_name>"}'
    )

    messages: list[dict[str, Any]] = []

    # If model supports vision (e.g., vision in name) and images exist, attach base64 image
    valid_images = [p for p in img_paths if os.path.exists(p)]
    if "vision" in selected_model.lower() and valid_images:
        content_parts: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
        for img_p in valid_images[:3]:
            b64 = _encode_image(img_p)
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        messages.append({"role": "user", "content": content_parts})
    else:
        messages.append({"role": "user", "content": prompt_text})

    try:
        response = client.chat.completions.create(
            model=selected_model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        content = response.choices[0].message.content or "{}"
        data = _clean_json_text(content)
        verb = str(data.get("action", "")).lower().strip()
        obj = str(data.get("object_label", prop.get("object_label", "unknown"))).lower().strip()

        if verb not in FIXED_VERBS:
            res = fallback
        else:
            res = {
                "action": verb,
                "object_label": obj or "unknown",
                "review_status": "accepted" if prop.get("contact_state") != "unknown" else "needs_review",
            }
        cache_file.write_text(json.dumps(res, indent=2), encoding="utf-8")
        time.sleep(0.5)
        return res, True
    except Exception as exc:
        log.warning("Groq AI fallback used: %s", exc)
        return fallback, False


def load_candidate_events(path: str | Path) -> list[dict[str, Any]]:
    """Load candidate events from either contacts.csv or events_proposals.json."""
    p = Path(path)
    if not p.exists():
        return []

    if p.suffix.lower() == ".csv":
        events: list[dict[str, Any]] = []
        with p.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, r in enumerate(reader):
                state = r.get("state", r.get("contact_state", "portable"))
                if state in ("nothing", "none"):
                    continue
                hand = r.get("hand", r.get("hand_side", "right"))
                start_sec = float(r.get("t_start", r.get("start_sec", 0.0)))
                end_sec = float(r.get("t_end", r.get("end_sec", 0.0)))
                clip = r.get("clip", r.get("clip_id", "demo"))
                obj_label = r.get("object_name", r.get("object_label", "object"))
                obj_id = r.get("object_id", f"obj_{idx+1}")
                hand_id = r.get("hand_track_id", f"hand_{hand}_1")
                tq = float(r.get("tracking_quality", 0.84))
                ev_frames_str = r.get("evidence_frames", "")
                events.append({
                    "clip_id": clip,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "hand_track_id": hand_id,
                    "hand_side": hand,
                    "object_id": obj_id,
                    "object_label": obj_label,
                    "contact_state": state,
                    "tracking_quality": tq,
                    "evidence_frames": [ef.strip() for ef in ev_frames_str.split("|") if ef.strip()] if ev_frames_str else [],
                })
        return events
    else:
        return json.loads(p.read_text(encoding="utf-8"))


def run_event_namer(
    video_path: str | Path | None = None,
    candidate_events_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> tuple[Path, Path, int]:
    """
    Main function to name events and write events_validated.json & actions.csv using Groq AI.
    Returns (json_path, csv_path, live_api_calls).
    """
    video = Path(video_path or CONFIG.paths.video)
    out_path = Path(out_dir or CONFIG.paths.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    candidates = Path(
        candidate_events_path
        or (CONFIG.paths.events_json if CONFIG.paths.events_json.exists() else CONFIG.paths.contacts_csv)
    )

    proposals = load_candidate_events(candidates)
    if not proposals and candidates != CONFIG.paths.contacts_csv and CONFIG.paths.contacts_csv.exists():
        proposals = load_candidate_events(CONFIG.paths.contacts_csv)

    if not proposals:
        log.warning("No candidate events found in %s", candidates)

    cache_dir = out_path / "api_cache"
    evidence_dir = out_path / "evidence_frames"
    cache_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    clip_id = video.stem if video.exists() else "demo"
    events_out: list[dict[str, Any]] = []
    api_calls = 0

    for i, p in enumerate(proposals):
        img_paths: list[str] = []
        ev_frames = p.get("evidence_frames", [])

        if ev_frames:
            for ef in ev_frames:
                for base in [out_path, evidence_dir, CONFIG.paths.output_dir, Path(".")]:
                    cand = base / str(ef) if not os.path.isabs(str(ef)) else Path(str(ef))
                    if cand.exists():
                        img_paths.append(str(cand))
                        break

        if not img_paths and video.exists():
            ev_frames, img_paths = sample_evidence_frames(
                video, p["start_sec"], p["end_sec"], evidence_dir=evidence_dir
            )

        if not ev_frames:
            ev_frames = [int(p["start_sec"] * 30), int(p["end_sec"] * 30)]

        named, called = query_llm_with_cache(
            clip_id,
            p,
            img_paths,
            cache_dir=cache_dir,
            api_key=api_key,
            model_name=model,
        )
        if called:
            api_calls += 1

        events_out.append({
            "clip_id": p.get("clip_id", clip_id),
            "event_id": f"{clip_id}_evt_{i+1:03d}",
            "start_sec": round(float(p["start_sec"]), 2),
            "end_sec": round(float(p["end_sec"]), 2),
            "action": named["action"],
            "hand_track_id": p.get("hand_track_id", f"hand_{p['hand_side']}_1"),
            "hand_side": p["hand_side"],
            "object_id": p.get("object_id", "obj_unknown"),
            "object_label": named["object_label"],
            "contact_state": p.get("contact_state", "unknown"),
            "tracking_quality": round(float(p.get("tracking_quality", 0.8)), 3),
            "evidence_frames": ev_frames,
            "review_status": named["review_status"],
        })

    json_path = out_path / "events_validated.json"
    csv_path = out_path / "actions.csv"

    json_path.write_text(json.dumps(events_out, indent=CONFIG.json_indent) + "\n", encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["clip", "start_s", "end_s", "verb", "object", "hand", "review_status"]
        )
        w.writeheader()
        for r in events_out:
            w.writerow({
                "clip": r["clip_id"],
                "start_s": r["start_sec"],
                "end_s": r["end_sec"],
                "verb": r["action"],
                "object": r["object_label"],
                "hand": r["hand_side"],
                "review_status": r["review_status"],
            })

    print(f"Validated {len(events_out)} events ({api_calls} live Groq API calls). Saved '{json_path}' and '{csv_path}'.")
    return json_path, csv_path, api_calls


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Name and validate contact events using Groq AI.")
    parser.add_argument("--video", default=str(CONFIG.paths.video), help="Input video file")
    parser.add_argument(
        "--candidates",
        "--candidate-events",
        "--proposals",
        dest="candidates",
        default=str(CONFIG.paths.events_json if CONFIG.paths.events_json.exists() else CONFIG.paths.contacts_csv),
        help="Path to contacts.csv or events_proposals.json",
    )
    parser.add_argument("--out-dir", default=str(CONFIG.paths.output_dir), help="Output directory")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Groq model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--api-key", default=None, help="Groq API Key (optional, defaults to GROQ_API_KEY env var)")
    args = parser.parse_args()

    run_event_namer(
        video_path=args.video,
        candidate_events_path=args.candidates,
        out_dir=args.out_dir,
        model=args.model,
        api_key=args.api_key,
    )
