"""
RULE 4 — LINE QUALITY [MANDATORY]  (optimized)
===============================================
All lines that ARE present (outer border + any dividers) must be:
  - Straight    : no curves, waves, tilts
  - Continuous  : no breaks, gaps, dots, dashes within a single line
  - Uniformly dark : consistent fill, no fading
  - Consistent thickness : no tapering or overly-heavy weight

Only fails if a defect is clearly visible.

Strategy
--------
The original code relied on the *existing* horizontal-line detection
(run_base_cv_analysis) which uses a morphological OPEN with a minimum-width
kernel.  That kernel MISSES dashed/dotted bars because the gaps between
segments cause the kernel to fall through — the bars are never even found,
so their defects can never be reported.

This optimised version adds a second CV stage:

  1. Row-projection scan  — finds every horizontal band of dark pixels,
     regardless of whether it is solid or broken.
  2. Band classifier      — distinguishes divider/border bands from text rows
     using three signals: average row darkness, row-to-row uniformity (std),
     and typical dark-segment width in the column direction.
  3. Quality classifier   — for every divider band decides:
       SOLID_OK           — continuous, correctly weighted
       DASHED             — wide rectangular gaps between solid segments  ≥ 25 px segs
       DOTTED             — small circular/square gaps between dot segments < 25 px segs
       TOO_THICK          — solid but excessively heavy (> 1.8× typical bar)
       UNCERTAIN          — thin transition row; skip
  4. LLM confirmation    — the CV findings drive a focused, evidence-rich
     prompt so the LLM confirms (or overrides with a clear visual reason).

Defect type mapping → rule_4 defect_types list:
  DASHED   → "broken" + "dashed"
  DOTTED   → "broken" + "dotted"
  TOO_THICK → "inconsistent_thickness"
"""

import cv2
import json
import numpy as np
from openai import OpenAI

# ---------------------------------------------------------------------------
# Re-use shared helpers from sfp_common
# ---------------------------------------------------------------------------
try:
    from sfp_common import run_base_cv_analysis, ask_llm, MODEL
except ImportError:  # stand-alone testing fallback
    import os, base64
    from pathlib import Path
    MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

    def ask_llm(client, prompt, image=None, max_tokens=500, json_response=False):
        import base64
        content = []
        if image is not None:
            if isinstance(image, np.ndarray):
                _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                b64 = base64.b64encode(buf.tobytes()).decode()
                mime = "image/jpeg"
            else:
                ext  = Path(image).suffix.lower()
                mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
                with open(image, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
            content.append({"type": "image_url",
                             "image_url": {"url": f"data:{mime};base64,{b64}"}})
        content.append({"type": "text", "text": prompt})
        kwargs = {
            "model":    MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_completion_tokens": max_tokens,
        }
        if json_response:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

    def run_base_cv_analysis(image_path):
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "cannot_read_image"}, None
        return {}, img


# ---------------------------------------------------------------------------
# ── NEW: Row-projection-based divider band detector ─────────────────────────
# ---------------------------------------------------------------------------

def _col_segment_stats(col_dark_frac: np.ndarray, threshold: float = 0.30):
    """
    Splits a per-column darkness array into alternating dark/bright runs.

    Returns (dark_runs_px, bright_runs_px) as Python lists.
    """
    dark_runs: list  = []
    bright_runs: list = []
    in_dark = col_dark_frac[0] >= threshold
    run = 0
    for v in col_dark_frac:
        if (v >= threshold) == in_dark:
            run += 1
        else:
            (dark_runs if in_dark else bright_runs).append(run)
            in_dark = not in_dark
            run = 1
    (dark_runs if in_dark else bright_runs).append(run)
    return dark_runs, bright_runs


def _is_divider_band(avg_row_dk: float, row_dk_std: float,
                     bh: int, n_segs: int, avg_dark_seg_px: float) -> bool:
    """
    Returns True when this horizontal band looks like a line/bar element
    (outer border or section divider) rather than a text row.

    Discriminating signals
    ----------------------
    Text rows  : avg_row_dk < 0.40, many tiny segments (avg_dark_seg < 20px),
                 non-uniform row-to-row (row_dk_std > 0.03).
    Divider bars: high avg_row_dk (≥ 0.65), very uniform rows (std ≈ 0),
                  or moderate darkness with wide segments (≥ 25px).
    Dashed bars : moderate avg_row_dk (0.45–0.75), few wide segments.
    Dotted bars : moderate avg_row_dk (0.40–0.65), uniform row-to-row,
                  segments 10–25px.
    """
    # Thin outer border
    if bh <= 3 and avg_row_dk > 0.85:
        return True
    # Solid thick bar
    if avg_row_dk >= 0.65 and row_dk_std < 0.05:
        return True
    # Dashed bar (wide rectangular dashes, moderate fill)
    if avg_row_dk >= 0.45 and avg_dark_seg_px >= 20 and n_segs <= 15:
        return True
    # Dotted bar (smaller circular dots, uniform rows)
    if avg_row_dk >= 0.40 and row_dk_std < 0.15 and n_segs >= 8 and avg_dark_seg_px >= 10:
        return True
    return False


def detect_divider_band_defects(image_path: str) -> dict:
    """
    Performs a row-projection scan of the image, classifies every horizontal
    divider/border band, and reports quality defects.

    Returns
    -------
    dict with keys:
        defects        — list of defect dicts, one per bad band
        bands_checked  — list of all divider bands analysed (for debugging)
        ref_bar_px     — median thickness of solid bars (px), used for
                         TOO_THICK threshold
        summary        — aggregate counts
    """
    print(f"[RULE-4-CV] Row-projection divider scan: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        return {"error": "cannot_read_image", "defects": [], "bands_checked": []}

    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W    = gray.shape
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Per-row dark fraction (fraction of columns that are dark)
    dark_frac_row = np.sum(binary > 127, axis=1).astype(float) / W

    # Group consecutive dark rows into bands
    DARK_THRESH = 0.10
    in_band = False
    start   = 0
    raw_bands: list = []
    for y in range(H):
        if not in_band and dark_frac_row[y] > DARK_THRESH:
            in_band = True
            start   = y
        elif in_band and dark_frac_row[y] <= DARK_THRESH:
            in_band = False
            raw_bands.append((start, y))
    if in_band:
        raw_bands.append((start, H))

    # ── Classify each band ──────────────────────────────────────────────
    solid_bar_heights: list = []
    bands_checked: list     = []
    defects: list           = []

    for (ys, ye) in raw_bands:
        bh         = ye - ys
        avg_row_dk = float(np.mean(dark_frac_row[ys:ye]))
        row_dk_std = float(np.std(dark_frac_row[ys:ye]))

        # Per-column dark fraction within this band
        band_region = binary[ys:ye, :]
        col_dark    = np.sum(band_region > 127, axis=0).astype(float) / max(bh, 1)
        gap_pct     = round(100.0 * float(np.sum(col_dark < 0.3)) / W, 1)

        dark_segs, bright_segs = _col_segment_stats(col_dark)
        n_segs        = len(dark_segs)
        avg_dark_seg  = float(np.mean(dark_segs))  if dark_segs  else 0.0
        avg_bright_seg= float(np.mean(bright_segs)) if bright_segs else 0.0

        if not _is_divider_band(avg_row_dk, row_dk_std, bh, n_segs, avg_dark_seg):
            continue  # skip text rows

        # ── Quality classification ──────────────────────────────────────
        is_outer_border = bh <= 3 and avg_row_dk > 0.85
        is_solid        = (gap_pct < 15.0
                           and n_segs <= 4
                           and avg_row_dk > 0.85
                           and row_dk_std < 0.02)

        if is_outer_border or is_solid:
            flag = "SOLID_OK"
            if bh > 3:
                solid_bar_heights.append(bh)   # accumulate for TOO_THICK reference
        elif gap_pct >= 15.0 and n_segs >= 6:
            # Distinguish dashes (wide segments) from dots (narrow segments)
            flag = "DASHED" if avg_dark_seg >= 25.0 else "DOTTED"
        else:
            flag = "UNCERTAIN"   # transitional / partial row — skip

        band_info = {
            "y_top":          ys,
            "y_bot":          ye,
            "height_px":      bh,
            "avg_row_dark":   round(avg_row_dk, 3),
            "row_dark_std":   round(row_dk_std, 4),
            "gap_pct":        gap_pct,
            "n_col_segments": n_segs,
            "avg_dark_seg_px":round(avg_dark_seg, 1),
            "avg_gap_seg_px": round(avg_bright_seg, 1),
            "classification": flag,
        }
        bands_checked.append(band_info)

        if flag not in ("SOLID_OK", "UNCERTAIN"):
            defects.append(band_info)

    # ── Post-pass: TOO_THICK check once we know the reference bar height ─
    if solid_bar_heights:
        ref_bar_px = float(np.median(solid_bar_heights))
        too_thick_threshold = ref_bar_px * 1.8   # > 1.8× typical = anomalous
        print(f"[RULE-4-CV] Solid bar reference = {ref_bar_px:.0f}px  "
              f"too-thick threshold = {too_thick_threshold:.0f}px")

        for band in bands_checked:
            if (band["classification"] == "SOLID_OK"
                    and band["height_px"] > too_thick_threshold
                    and band["height_px"] > 3):   # never flag the outer hairline
                band["classification"] = "TOO_THICK"
                defects.append(band)
    else:
        ref_bar_px = 30.0   # fallback default

    # Deduplicate defects (TOO_THICK check may add already-listed bands)
    seen = set()
    unique_defects = []
    for d in defects:
        key = d["y_top"]
        if key not in seen:
            seen.add(key)
            unique_defects.append(d)
    defects = unique_defects

    n_dashed   = sum(1 for d in defects if d["classification"] == "DASHED")
    n_dotted   = sum(1 for d in defects if d["classification"] == "DOTTED")
    n_too_thick= sum(1 for d in defects if d["classification"] == "TOO_THICK")
    n_checked  = len(bands_checked)

    print(f"[RULE-4-CV] Bands checked: {n_checked}  "
          f"Defects → dashed={n_dashed}  dotted={n_dotted}  too_thick={n_too_thick}")

    return {
        "defects":        defects,
        "bands_checked":  bands_checked,
        "ref_bar_px":     round(ref_bar_px, 1),
        "summary": {
            "bands_checked":    n_checked,
            "dashed_count":     n_dashed,
            "dotted_count":     n_dotted,
            "too_thick_count":  n_too_thick,
            "total_defects":    len(defects),
            "cv_rule4_pass":    len(defects) == 0,
        },
    }


# ---------------------------------------------------------------------------
# ── LLM prompt ──────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

RULE_4_PROMPT = """\
You are evaluating an FDA Supplement Facts Panel for compliance with
RULE 4 (LINE QUALITY): every line in the panel (outer border + dividers)
must be straight, continuous (no dashes / dots / gaps within a single line),
uniformly dark, and of consistent thickness.

Only fail this rule if you can CLEARLY SEE a defect.

Computer-Vision pre-analysis (row-projection scan):
  cv_rule4_pass              : {cv_rule4_pass}   <- primary CV signal
  total_defective_bands      : {total_defects}
  dashed_bands_count         : {dashed_count}
  dotted_bands_count         : {dotted_count}
  too_thick_bands_count      : {too_thick_count}
  reference_solid_bar_px     : {ref_bar_px}
  defect_band_details (y_top, height_px, type):
{defect_details}

Use the CV signal as the primary guide. Look at the image and verify.
Only override the CV verdict if you have a clear visual reason.

Respond ONLY with valid JSON (no markdown fences):
{{
  "pass":  <true|false>,
  "notes": "<one sentence — what defects (if any) you observed>",
  "defect_types": ["<broken|dashed|dotted|inconsistent_thickness|faded|tilted>", ...]
}}"""


# ---------------------------------------------------------------------------
# ── Main rule check ──────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def check_rule_4(
    image_path: str,
    client:     "OpenAI",
    cv_data:    "dict | None" = None,       # kept for API compatibility
) -> dict:
    """
    Check Rule 4 — Line Quality.

    Parameters
    ----------
    image_path : str
        Local path to the image file.
    client : OpenAI
        Initialised OpenAI client.
    cv_data : dict, optional
        Pre-computed data from run_base_cv_analysis().  Accepted for API
        compatibility but Rule 4 now runs its own independent row-projection
        scan that reliably catches dashed/dotted bars which the shared CV
        analysis misses.

    Returns
    -------
    dict with keys:
        pass             — bool
        notes            — str
        defect_types     — list of str  (broken, dashed, dotted,
                                         inconsistent_thickness, faded, tilted)
        cv_evidence      — dict  (all raw CV findings)
    """
    print(f"\n[RULE-4] Checking line quality (optimised)...")

    # ── Stage 1: row-projection CV scan ────────────────────────────────
    scan = detect_divider_band_defects(image_path)

    if "error" in scan:
        return {
            "pass":         False,
            "notes":        f"CV error: {scan['error']}",
            "defect_types": [],
            "cv_evidence":  scan,
        }

    s = scan["summary"]

    # ── Build defect detail string for the prompt ───────────────────────
    if scan["defects"]:
        detail_lines = []
        for d in scan["defects"]:
            detail_lines.append(
                f"    y={d['y_top']}  h={d['height_px']}px  "
                f"gap={d['gap_pct']}%  segs={d['n_col_segments']}  "
                f"type={d['classification']}"
            )
        defect_details = "\n".join(detail_lines)
    else:
        defect_details = "    (none)"

    # ── Stage 2: LLM visual confirmation ────────────────────────────────
    prompt = RULE_4_PROMPT.format(
        cv_rule4_pass   = s["cv_rule4_pass"],
        total_defects   = s["total_defects"],
        dashed_count    = s["dashed_count"],
        dotted_count    = s["dotted_count"],
        too_thick_count = s["too_thick_count"],
        ref_bar_px      = scan["ref_bar_px"],
        defect_details  = defect_details,
    )

    print(f"[RULE-4] CV verdict: {'PASS' if s['cv_rule4_pass'] else 'FAIL'}  "
          f"(dashed={s['dashed_count']} dotted={s['dotted_count']} "
          f"too_thick={s['too_thick_count']})  — asking LLM to confirm...")

    response_text = ask_llm(client, prompt, image=image_path,
                            max_tokens=250, json_response=True)
    result = json.loads(response_text)

    # ── Map CV classification names to standard defect_types vocabulary ─
    defect_types: list = list(result.get("defect_types", []))
    if not defect_types:
        # Synthesise from CV if LLM returned empty list but we have defects
        if s["dashed_count"] > 0:
            defect_types.extend(["broken", "dashed"])
        if s["dotted_count"] > 0:
            defect_types.extend(["broken", "dotted"])
        if s["too_thick_count"] > 0:
            defect_types.append("inconsistent_thickness")
        defect_types = sorted(set(defect_types))

    print(f"[RULE-4] Result: {'PASS' if result.get('pass') else 'FAIL'} — "
          f"{result.get('notes', '')}")

    return {
        "pass":         bool(result.get("pass", False)),
        "notes":        result.get("notes", ""),
        "defect_types": defect_types,
        "cv_evidence": {
            # legacy keys (kept for backward compatibility with callers)
            "any_broken_lines": s["dashed_count"] + s["dotted_count"] > 0,
            "any_dotted_lines": s["dotted_count"] > 0,
            "broken_count":     s["dashed_count"],
            "dotted_count":     s["dotted_count"],
            # new detailed keys
            "dashed_band_count":    s["dashed_count"],
            "dotted_band_count":    s["dotted_count"],
            "too_thick_band_count": s["too_thick_count"],
            "total_defective_bands":s["total_defects"],
            "ref_solid_bar_px":     scan["ref_bar_px"],
            "cv_rule4_pass":        s["cv_rule4_pass"],
            "bands_detail":         scan["bands_checked"],
        },
    }


# ---------------------------------------------------------------------------
# ── Quick self-test (run this file directly) ─────────────────────────────────
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    from pathlib import Path
    from openai import OpenAI

    _test_dir = Path(__file__).resolve().parent / "test for rule 4"
    test_images = {
        "15.4.R.1 (GOOD — expect PASS)":
            _test_dir / "15.4.R.1.jpg",
        "15.4.W.1 (DASHED — expect FAIL)":
            _test_dir / "15.4.W.1.jpg",
        "15.4.W.2 (DOTTED — expect FAIL)":
            _test_dir / "15.4.W.2.jpg",
        "15.4.W.3 (TOO THICK — expect FAIL)":
            _test_dir / "15.4.W.3.jpg",
    }

    api_key = os.environ.get("OPENAI_API_KEY", "").strip() or "api_key"
    if not api_key:
        # CV-only mode (no LLM call)
        print("No OPENAI_API_KEY found — running CV-only self-test.\n")
        for label, path in test_images.items():
            path = str(path)
            scan = detect_divider_band_defects(path)
            s    = scan.get("summary", {})
            verdict = "PASS" if s.get("cv_rule4_pass") else "FAIL"
            print(f"  {label}")
            print(f"    CV verdict : {verdict}")
            print(f"    defects    : dashed={s.get('dashed_count',0)}  "
                  f"dotted={s.get('dotted_count',0)}  "
                  f"too_thick={s.get('too_thick_count',0)}")
            for b in scan.get("bands_checked", []):
                if b["classification"] not in ("SOLID_OK", "UNCERTAIN"):
                    print(f"    >> y={b['y_top']} h={b['height_px']}px "
                          f"gap={b['gap_pct']}% segs={b['n_col_segments']} "
                          f"-> {b['classification']}")
            print()
    else:
        client = OpenAI(api_key=api_key)
        print("Running full CV + LLM self-test.\n")
        for label, path in test_images.items():
            path = str(path)
            result = check_rule_4(path, client)
            verdict_str = "PASS" if result["pass"] else "FAIL"
            print(f"\n  {label}")
            print(f"    verdict      : {verdict_str}")
            print(f"    notes        : {result['notes']}")
            print(f"    defect_types : {result['defect_types']}")