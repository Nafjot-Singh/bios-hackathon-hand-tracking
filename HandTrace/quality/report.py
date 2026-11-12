"""
report.py

Builds a single-file HTML report from the tracking metrics and graphs.

Input:
    - Metrics dictionary from MetricsCalculator.calculate() (or metrics.json)
    - Graph PNG paths from GraphGenerator.generate_all() (optional)

Output:
    output/report.html (graphs are embedded, so the file can be shared as is)

This module DOES NOT:
- Compute metrics
- Draw graphs
- Modify tracking results

Those responsibilities belong to metrics.py and graph.py.

Usage:
    ReportGenerator(metrics, graphs).generate()

    python -m quality.report            # rebuild everything from output/metrics.json
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from config import CONFIG

log = logging.getLogger(__name__)

GRAPH_TITLES: dict[str, str] = {
    "hands_over_time": "Hands detected per frame",
    "hands_per_frame": "Frames by number of hands",
    "confidence": "Confidence distribution",
    "handedness": "Handedness",
    "track_timeline": "Track timeline",
    "track_lifetimes": "Track lifetimes",
}

_CSS = """
:root {
  --bg: #f8fafc; --card: #ffffff; --text: #0f172a; --muted: #64748b;
  --border: #e2e8f0; --ok: #16a34a; --info: #2563eb; --warning: #d97706;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f172a; --card: #1e293b; --text: #f1f5f9; --muted: #94a3b8;
    --border: #334155; --ok: #4ade80; --info: #60a5fa; --warning: #fbbf24;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
       font: 15px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 32px 20px 64px; }
h1 { margin: 0 0 4px; font-size: 28px; }
h2 { margin: 36px 0 12px; font-size: 20px; }
.meta { color: var(--muted); margin: 0; }
.cards { display: grid; gap: 12px; margin-top: 24px;
         grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); }
.card { background: var(--card); border: 1px solid var(--border);
        border-radius: 10px; padding: 14px 16px; }
.card .value { font-size: 26px; font-weight: 650; }
.card .label { color: var(--muted); font-size: 13px; }
ul.findings { list-style: none; margin: 0; padding: 0; }
ul.findings li { background: var(--card); border: 1px solid var(--border);
                 border-left-width: 4px; border-radius: 8px;
                 padding: 10px 14px; margin-bottom: 8px; }
li.ok { border-left-color: var(--ok); }
li.info { border-left-color: var(--info); }
li.warning { border-left-color: var(--warning); }
table { width: 100%; border-collapse: collapse; background: var(--card);
        border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border); }
th { color: var(--muted); font-weight: 600; font-size: 13px; }
tr:last-child td { border-bottom: none; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.table-wrap { overflow-x: auto; }
figure { margin: 0 0 20px; background: var(--card); border: 1px solid var(--border);
         border-radius: 10px; padding: 12px; }
figure img { max-width: 100%; height: auto; display: block; margin: 0 auto;
             background: #fff; border-radius: 6px; }
figcaption { color: var(--muted); font-size: 13px; margin-bottom: 8px; }
.note { color: var(--muted); font-size: 13px; margin-top: 8px; }
"""


class ReportGenerator:
    """Creates a self-contained HTML report."""

    # Rules used for the "Findings" section.
    LOW_DETECTION_RATE = 0.5          # warn below this share of frames with hands
    LOW_CONFIDENCE_SHARE = 0.10       # warn when more records than this are low-confidence
    MAX_TRACK_ROWS = 100
    MAX_SWITCH_ROWS = 50

    def __init__(
        self,
        metrics: Mapping[str, Any],
        graphs: Mapping[str, Path | str] | None = None,
        output_path: Path | str | None = None,
        title: str = "Hand Tracking Report",
        source_name: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        metrics:
            Output of MetricsCalculator.calculate() or a loaded metrics.json.
        graphs:
            Graph name -> PNG path (from GraphGenerator.generate_all()).
        output_path:
            Report file. Defaults to <output dir>/report.html.
        title:
            Heading of the report.
        source_name:
            Name of the analyzed video, shown under the title.
        """
        self.metrics: dict[str, Any] = dict(metrics)
        self.graphs: dict[str, Path] = {k: Path(v) for k, v in (graphs or {}).items()}
        self.output_path = Path(output_path or CONFIG.paths.output_dir / "report.html")
        self.title = title
        self.source_name = source_name

    # ----------------------------------------------------
    # Public API
    # ----------------------------------------------------

    def generate(self) -> Path:
        """Write the report to disk and return its path."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Temp file first so a failure never leaves a half-written report.
        temp = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        temp.write_text(self.to_html(), encoding="utf-8")
        temp.replace(self.output_path)
        return self.output_path

    def to_html(self) -> str:
        """Return the full report as an HTML string."""
        generated = datetime.now().strftime("%Y-%m-%d %H:%M")
        source = f"Video: {self.source_name} · " if self.source_name else ""

        body = "\n".join(
            [
                f"<h1>{self._esc(self.title)}</h1>",
                f'<p class="meta">{self._esc(source)}Generated {generated}</p>',
                self._cards(),
                "<h2>Findings</h2>",
                self._findings_html(),
                "<h2>Details</h2>",
                self._details(),
                self._graphs_html(),
                self._tracks_table(),
                self._switches_table(),
            ]
        )

        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"<title>{self._esc(self.title)}</title>\n<style>{_CSS}</style>\n"
            f"</head>\n<body>\n<main>\n{body}\n</main>\n</body>\n</html>\n"
        )

    # ----------------------------------------------------
    # Sections
    # ----------------------------------------------------

    def _cards(self) -> str:
        m = self.metrics
        longest = int(m.get("longest_disappearance", 0))
        cards = [
            (f"{m.get('detection_rate', 0.0):.1%}", "Frames with hands"),
            (m.get("unique_tracks", 0), "Unique tracks"),
            (m.get("id_switches", 0), "Estimated ID switches"),
            (f"{longest} frames", "Longest disappearance"),
            (f"{m.get('average_confidence', 0.0):.3f}", "Average confidence"),
            (m.get("records", 0), "Hand records"),
        ]
        items = "".join(
            f'<div class="card"><div class="value">{self._esc(v)}</div>'
            f'<div class="label">{self._esc(label)}</div></div>'
            for v, label in cards
        )
        return f'<section class="cards">{items}</section>'

    def _findings(self) -> list[tuple[str, str]]:
        """Plain-language observations as (level, text); level is ok/info/warning."""
        m = self.metrics
        findings: list[tuple[str, str]] = []

        if m.get("total_frames_is_estimate"):
            findings.append((
                "info",
                "The total frame count was estimated from the last frame with a "
                "hand, so the detection rate may be too high.",
            ))

        if m.get("skipped_lines"):
            findings.append((
                "warning",
                f"{m['skipped_lines']} unreadable line(s) in hands.jsonl were skipped.",
            ))

        rate = float(m.get("detection_rate", 0.0))
        if m.get("total_frames") and rate < self.LOW_DETECTION_RATE:
            findings.append((
                "warning",
                f"Hands were visible in only {rate:.1%} of the frames.",
            ))

        switches = int(m.get("id_switches", 0))
        if switches:
            findings.append((
                "warning",
                f"{switches} likely ID switch(es): a hand got a new ID shortly after "
                f"an old track ended. Consider raising max_missing_frames "
                f"(now {CONFIG.tracking.max_missing_frames}) or max_track_distance "
                f"(now {CONFIG.tracking.max_track_distance:g} px).",
            ))

        records = int(m.get("records", 0))
        low = int(m.get("low_confidence_records", 0))
        if records and low / records > self.LOW_CONFIDENCE_SHARE:
            findings.append((
                "warning",
                f"{low / records:.0%} of records have confidence below "
                f"{m.get('low_confidence_threshold', 0):g}.",
            ))

        longest = int(m.get("longest_disappearance", 0))
        if longest:
            findings.append((
                "info",
                f"The longest disappearance under the same ID was {longest} frame(s) "
                f"(track {m.get('longest_disappearance_track')}).",
            ))

        if not findings:
            findings.append(("ok", "No issues found."))
        return findings

    def _findings_html(self) -> str:
        items = "".join(
            f'<li class="{level}">{self._esc(text)}</li>'
            for level, text in self._findings()
        )
        return f'<ul class="findings">{items}</ul>'

    def _details(self) -> str:
        m = self.metrics
        estimate = " (estimated)" if m.get("total_frames_is_estimate") else ""

        histogram = ", ".join(
            f"{n} hand(s): {c}" for n, c in m.get("hands_per_frame_histogram", {}).items()
        ) or "-"
        handedness = ", ".join(
            f"{k}: {v}" for k, v in m.get("handedness_counts", {}).items()
        ) or "-"

        rows = [
            ("Total frames", f"{m.get('total_frames', 0)}{estimate}"),
            ("Frames with hands", m.get("frames_with_hands", 0)),
            ("Frames without hands", m.get("frames_without_hands", 0)),
            ("Average hands per frame", f"{m.get('average_hands_per_frame', 0.0):.2f}"),
            ("Maximum hands in a frame", m.get("maximum_hands_in_frame", 0)),
            ("Hands per frame", histogram),
            ("Handedness", handedness),
            ("Confidence (avg / median)",
             f"{m.get('average_confidence', 0.0):.3f} / {m.get('median_confidence', 0.0):.3f}"),
            ("Confidence (min / max)",
             f"{m.get('minimum_confidence', 0.0):.3f} / {m.get('maximum_confidence', 0.0):.3f}"),
            (f"Records below {m.get('low_confidence_threshold', 0):g}",
             m.get("low_confidence_records", 0)),
            ("Average track length (visible)", f"{m.get('average_track_length', 0.0):.1f} frames"),
            ("Average track span", f"{m.get('average_track_span', 0.0):.1f} frames"),
            ("Longest / shortest track",
             f"{m.get('longest_track', 0)} / {m.get('shortest_track', 0)} frames"),
            ("Tracks with gaps", m.get("tracks_with_gaps", 0)),
            ("Disappearance events", m.get("disappearance_events", 0)),
            ("Gap frames (total)", m.get("total_gap_frames", 0)),
        ]
        return self._table(["Metric", "Value"], [[k, v] for k, v in rows], numeric_cols=())

    def _graphs_html(self) -> str:
        figures: list[str] = []
        for name, path in self.graphs.items():
            if not path.is_file():
                log.warning("Graph file missing, not embedded: %s", path)
                continue
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            title = GRAPH_TITLES.get(name, name.replace("_", " ").capitalize())
            figures.append(
                f"<figure><figcaption>{self._esc(title)}</figcaption>"
                f'<img alt="{self._esc(title)}" src="data:image/png;base64,{data}"></figure>'
            )
        if not figures:
            return ""
        return "<h2>Graphs</h2>\n" + "\n".join(figures)

    def _tracks_table(self) -> str:
        tracks = sorted(self.metrics.get("tracks", []), key=lambda t: t["track_id"])
        if not tracks:
            return ""

        rows = [
            [t["track_id"], t["handedness"], t["first_frame"], t["last_frame"],
             t["visible_frames"], t["span"], t["gap_frames"], t["longest_gap"]]
            for t in tracks[: self.MAX_TRACK_ROWS]
        ]
        table = self._table(
            ["ID", "Hand", "First", "Last", "Visible", "Span", "Gap frames", "Longest gap"],
            rows,
            numeric_cols=(0, 2, 3, 4, 5, 6, 7),
        )
        note = ""
        if len(tracks) > self.MAX_TRACK_ROWS:
            note = (f'<p class="note">Showing the first {self.MAX_TRACK_ROWS} of '
                    f"{len(tracks)} tracks.</p>")
        return f"<h2>Tracks</h2>\n{table}{note}"

    def _switches_table(self) -> str:
        events = self.metrics.get("id_switch_events", [])
        if not events:
            return ""

        rows = [
            [e["handedness"], f"ID {e['from_track']} → ID {e['to_track']}",
             e["last_frame_old"], e["first_frame_new"], e["gap_frames"]]
            for e in events[: self.MAX_SWITCH_ROWS]
        ]
        table = self._table(
            ["Hand", "Switch", "Old track ended", "New track started", "Gap frames"],
            rows,
            numeric_cols=(2, 3, 4),
        )
        note = ""
        if len(events) > self.MAX_SWITCH_ROWS:
            note = (f'<p class="note">Showing the first {self.MAX_SWITCH_ROWS} of '
                    f"{len(events)} switches.</p>")
        window = self.metrics.get("switch_window_frames", 0)
        explain = (
            f'<p class="note">Estimated: a new track with the same handedness started '
            f"within {self._esc(window)} frames of an old one ending.</p>"
        )
        return f"<h2>ID switches</h2>\n{table}{note}{explain}"

    # ----------------------------------------------------
    # HTML helpers
    # ----------------------------------------------------

    def _table(
        self,
        headers: list[str],
        rows: list[list[Any]],
        numeric_cols: tuple[int, ...],
    ) -> str:
        def cell(tag: str, index: int, value: Any) -> str:
            css = ' class="num"' if index in numeric_cols else ""
            return f"<{tag}{css}>{self._esc(value)}</{tag}>"

        head = "".join(cell("th", i, h) for i, h in enumerate(headers))
        body = "".join(
            "<tr>" + "".join(cell("td", i, v) for i, v in enumerate(row)) + "</tr>"
            for row in rows
        )
        return (f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead>'
                f"<tbody>{body}</tbody></table></div>")

    @staticmethod
    def _esc(value: Any) -> str:
        """Escape any value for safe use in HTML."""
        return html.escape(str(value))


# ----------------------------------------------------
# Standalone use
# ----------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the HTML tracking report.")
    parser.add_argument("--metrics", default=None, help="Path to metrics.json.")
    parser.add_argument("--output", default=None, help="Report file to write.")
    parser.add_argument("--no-graphs", action="store_true", help="Skip graph generation.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    metrics_path = Path(args.metrics or CONFIG.paths.output_metrics)
    if not metrics_path.is_file():
        raise SystemExit(f"Metrics file not found: {metrics_path} (run main.py first)")
    loaded = json.loads(metrics_path.read_text(encoding="utf-8"))

    graph_files: dict[str, Path] = {}
    if not args.no_graphs:
        from quality.graph import GraphGenerator  # needs matplotlib

        graph_files = GraphGenerator(
            loaded, output_dir=CONFIG.paths.output_dir / "graphs"
        ).generate_all()

    report = ReportGenerator(
        loaded, graph_files, args.output, source_name=CONFIG.paths.video.name
    ).generate()
    print(f"Report saved to {report}")
