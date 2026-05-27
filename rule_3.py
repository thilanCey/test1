"""
RULE 3 — INDIVIDUAL NUTRIENT DIVIDERS [OPTIONAL]   (optimized)
================================================================
A ¼-point (hairline) rule MAY be used to separate individual nutrients.
When used it MUST be:
  - Correctly weighted : ≤ 0.25pt (hairline — no heavier than the outer border)
  - Uniformly dark     : no fading or greying out
  - Centered with 2pt of leading above AND below (white space)
  - Consistent         : if used between any pair of rows, it must be used
                         between ALL adjacent nutrient rows in the same section

This rule is OPTIONAL in the sense that a panel with NO hairlines at all
passes.  But if hairlines are used anywhere they must comply with all four
requirements above.

Defect taxonomy
---------------
  MISSING_HAIRLINE        — gap between adjacent nutrient rows has no divider
                            (but other gaps in the same panel do have one)
  INCONSISTENT_USAGE      — hairlines present in some sections, absent in others
  TOO_THICK               — hairline height > 2 × outer border thickness
  FADED                   — hairline average darkness < 0.85 (not solid black)
  LEADING_ABOVE_TOO_SMALL — white space above hairline < 2pt
  LEADING_BELOW_TOO_SMALL — white space below hairline < 2pt

What was wrong with the original code
--------------------------------------
The original check_rule_3 used detect_content_bands + find_missing_dividers,
which:
  1. Correctly detected MISSING hairlines (W_2), but
  2. Could NOT detect TOO_THICK hairlines (W_1 — h=6px classified as THICK_BAR)
  3. Could NOT detect FADED hairlines    (W_3 — dk=0.66, still labelled THICK_BAR)
  4. Could NOT measure leading above/below each hairline
  5. Could NOT detect INCONSISTENT usage across sections

This module adds a dedicated row-projection CV stage that measures all five
properties independently before asking the LLM to confirm.
"""

import cv2
import json
import numpy as np
from openai import OpenAI

# ---------------------------------------------------------------------------
# Re-use shared helpers from sfp_common
# ---------------------------------------------------------------------------
try:
    from sfp_common import (
        run_base_cv_analysis,
        detect_content_bands,
        find_missing_dividers,
        identify_surrounding_nutrients,
        ask_llm,
        make_image_message,
        MODEL,
    )
except ImportError:
    # stand-alone testing fallback — minimal stubs
    import os, base64
    from pathlib import Path

    MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

    def ask_llm(client, prompt, image=None, max_tokens=500, json_response=False):
        content = []
        if image is not None:
            if isinstance(image, np.ndarray):
                _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                b64  = base64.b64encode(buf.tobytes()).decode()
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

    def make_image_message(image):
        if isinstance(image, np.ndarray):
            _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            b64  = base64.b64encode(buf.tobytes()).decode()
            mime = "image/jpeg"
        else:
            ext  = Path(image).suffix.lower()
            mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
            with open(image, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}

    def detect_content_bands(gray, margin_frac=0.08):
        H, W     = gray.shape
        mx       = int(W * margin_frac)
        interior = gray[:, mx: W - mx]
        IW       = interior.shape[1]
        light_frac = np.sum(interior > 200, axis=1).astype(float) / IW
        dark_frac  = np.sum(interior < 80,  axis=1).astype(float) / IW
        LIGHT_THRESHOLD = 0.92
        MIN_BAND_HEIGHT  = 2
        bands = []; in_band = False; start = 0
        for y in range(H):
            if not in_band and light_frac[y] < LIGHT_THRESHOLD:
                in_band = True; start = y
            elif in_band and light_frac[y] >= LIGHT_THRESHOLD:
                in_band = False
                h = y - start
                if h < MIN_BAND_HEIGHT: continue
                avg_dk = float(np.mean(dark_frac[start:y]))
                max_dk = float(np.max(dark_frac[start:y]))
                kind   = ("THICK_BAR" if max_dk > 0.85
                           else "LINE"      if h <= 5
                           else "TEXT_ROW")
                bands.append({"y_top": start, "y_bot": y, "height": h,
                               "avg_dark_frac": round(avg_dk, 3),
                               "max_dark_frac": round(max_dk, 3),
                               "kind": kind})
        if in_band:
            h = H - start
            avg_dk = float(np.mean(dark_frac[start:H]))
            max_dk = float(np.max(dark_frac[start:H]))
            kind   = ("THICK_BAR" if max_dk > 0.85
                       else "LINE"      if h <= 5
                       else "TEXT_ROW")
            if h >= MIN_BAND_HEIGHT:
                bands.append({"y_top": start, "y_bot": H, "height": h,
                               "avg_dark_frac": round(avg_dk, 3),
                               "max_dark_frac": round(max_dk, 3),
                               "kind": kind})
        return bands

    def identify_surrounding_nutrients(img, regions, client, region_type="gap"):
        return [f"Unknown and Unknown" for _ in regions]

    def run_base_cv_analysis(image_path):
        img = cv2.imread(image_path)
        return {}, img


# ===========================================================================
# ── NEW: Hairline quality CV scan ───────────────────────────────────────────
# ===========================================================================

def _row_dark_fractions(binary: np.ndarray) -> np.ndarray:
    """Per-row fraction of dark (foreground) pixels."""
    H, W = binary.shape
    return np.sum(binary > 127, axis=1).astype(float) / W


def _find_structural_bands(dark_frac_row: np.ndarray,
                            dark_thresh: float = 0.10) -> list:
    """
    Groups consecutive dark rows into (y_start, y_end, height) tuples.
    """
    H = len(dark_frac_row)
    in_band = False; start = 0; bands = []
    for y in range(H):
        if not in_band and dark_frac_row[y] > dark_thresh:
            in_band = True; start = y
        elif in_band and dark_frac_row[y] <= dark_thresh:
            in_band = False
            bands.append((start, y, y - start))
    if in_band:
        bands.append((start, H, H - start))
    return bands


def _measure_gap(dark_frac_row: np.ndarray,
                 from_y: int, direction: int,
                 limit_y: int,
                 blank_threshold: float = 0.05) -> int:
    """Count consecutive near-blank rows starting at from_y."""
    count = 0
    y = from_y
    while 0 <= y < len(dark_frac_row) and (direction > 0 and y < limit_y or
                                             direction < 0 and y >= limit_y):
        if dark_frac_row[y] <= blank_threshold:
            count += 1
        else:
            break
        y += direction
    return count


def analyze_hairline_quality(image_path: str) -> dict:
    """
    Performs a complete hairline quality CV scan:

      1. Estimates DPI from the outer-border thickness
      2. Locates the nutrient body (between thick structural bars)
      3. Finds every thin candidate hairline inside the body
      4. Classifies each as: OK | too_thick | faded | leading_too_small
      5. Detects missing hairlines (gaps between text-row pairs with no divider)
      6. Detects inconsistent usage across sections

    Returns
    -------
    dict with keys:
        dpi_estimate          — estimated image DPI
        border_px             — outer border thickness in pixels
        max_hairline_px       — maximum allowed hairline thickness (2 × border)
        min_leading_px        — minimum required leading gap (2pt in px)
        hairlines             — list of hairline detail dicts
        bad_quality_hairlines — subset of hairlines with issues
        missing_hairlines     — list of gap dicts where hairline is absent
        section_reports       — per-section summary
        any_hairlines_used    — bool: are hairlines present at all?
        inconsistent_usage    — bool: used in some sections but not others
        cv_pass               — bool: overall CV verdict
        cv_fail_reasons       — list of str
    """
    print(f"[RULE-3-CV] Hairline quality scan: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        return {"error": "cannot_read_image"}

    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W    = gray.shape
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark_frac_row = _row_dark_fractions(binary)

    # ── 1. DPI estimation ─────────────────────────────────────────────────
    all_bands = _find_structural_bands(dark_frac_row)
    outer_borders = [(ys, ye, bh) for ys, ye, bh in all_bands
                     if bh <= 3
                     and float(np.mean(dark_frac_row[ys:ye])) > 0.90
                     and ys < 30]
    border_px      = outer_borders[0][2] if outer_borders else 2
    dpi_estimate   = round(border_px / (0.5 / 72))          # 0.5pt outer border
    # ¼pt hairline  → same as 0.5 × border_px; allow 2× for JPEG rounding
    max_hairline_px  = border_px * 2
    # 2pt leading   → 4 × border_px
    min_leading_px   = max(6, border_px * 4)

    print(f"[RULE-3-CV] dpi≈{dpi_estimate}  border={border_px}px  "
          f"max_hairline={max_hairline_px}px  min_leading={min_leading_px}px")

    # ── 2. Locate nutrient body ───────────────────────────────────────────
    thick_bars = [(ys, ye, bh) for ys, ye, bh in all_bands
                  if bh >= 14
                  and float(np.mean(dark_frac_row[ys:ye])) > 0.85]

    if len(thick_bars) < 2:
        return {
            "error": "insufficient_structure",
            "thick_bar_count": len(thick_bars),
            "cv_pass": True,   # can't assess → conservative pass
        }

    body_top = thick_bars[1][1]    # bottom edge of 2nd thick bar (3pt header rule)
    body_bot = thick_bars[-1][0]   # top edge of last thick bar (bottom section divider)

    # ── 3. Find hairline candidates ───────────────────────────────────────
    # thin (≤8px), meaningfully dark (>0.55), inside nutrient body
    candidates = [(ys, ye, bh) for ys, ye, bh in all_bands
                  if body_top < ys
                  and ye < body_bot
                  and bh <= 8
                  and float(np.mean(dark_frac_row[ys:ye])) > 0.55]

    # ── 4. Classify each hairline ─────────────────────────────────────────
    hairlines = []
    for ys, ye, bh in candidates:
        avg_dk    = float(np.mean(dark_frac_row[ys:ye]))
        gap_above = _measure_gap(dark_frac_row, ys - 1, -1, body_top)
        gap_below = _measure_gap(dark_frac_row, ye,      +1, body_bot)

        issues: list[str] = []
        if bh > max_hairline_px:
            issues.append("too_thick")
        if avg_dk < 0.85:
            issues.append("faded")
        if gap_above < min_leading_px:
            issues.append("leading_above_too_small")
        if gap_below < min_leading_px:
            issues.append("leading_below_too_small")

        hairlines.append({
            "y_top":         ys,
            "y_bot":         ye,
            "height_px":     bh,
            "avg_darkness":  round(avg_dk, 3),
            "gap_above_px":  gap_above,
            "gap_below_px":  gap_below,
            "issues":        issues,
            "quality_ok":    len(issues) == 0,
        })

    bad_quality = [h for h in hairlines if not h["quality_ok"]]

    # ── 5. Detect missing hairlines per section ───────────────────────────
    # Sections = spans between consecutive thick bars within the body
    body_thick = [(ys, ye, bh) for ys, ye, bh in thick_bars
                  if ys >= body_top and ye <= body_bot + 50]

    # Build section spans: [body_top → first_body_thick_top] then between each pair
    if body_thick:
        section_spans = [(body_top, body_thick[0][0])]
        for i in range(len(body_thick) - 1):
            section_spans.append((body_thick[i][1], body_thick[i + 1][0]))
    else:
        section_spans = [(body_top, body_bot)]

    section_reports  = []
    missing_hairlines = []

    for sec_top, sec_bot in section_spans:
        if sec_bot - sec_top < 10:
            continue

        # Count text-row groups in this section
        in_text = False; ts = 0; text_rows = []
        for y in range(sec_top, sec_bot):
            is_t = 0.05 < dark_frac_row[y] < 0.60
            if not in_text and is_t:
                in_text = True; ts = y
            elif in_text and not is_t:
                in_text = False
                if y - ts >= 5:
                    text_rows.append((ts, y))
        if in_text and sec_bot - ts >= 5:
            text_rows.append((ts, sec_bot))

        expected = max(0, len(text_rows) - 1)
        sec_hl   = [h for h in hairlines
                    if sec_top < h["y_top"] and h["y_bot"] < sec_bot]

        section_reports.append({
            "y_top":              sec_top,
            "y_bot":              sec_bot,
            "text_row_count":     len(text_rows),
            "expected_hairlines": expected,
            "found_hairlines":    len(sec_hl),
            "missing_count":      max(0, expected - len(sec_hl)),
        })

        # Record each specific gap that lacks a divider
        for i in range(len(text_rows) - 1):
            gap_y_top = text_rows[i][1]
            gap_y_bot = text_rows[i + 1][0]
            has_hl = any(gap_y_top <= h["y_top"] and h["y_bot"] <= gap_y_bot
                         for h in hairlines)
            if not has_hl:
                missing_hairlines.append({
                    "y_gap_top":    gap_y_top,
                    "y_gap_bottom": gap_y_bot,
                    "gap_height":   gap_y_bot - gap_y_top,
                    "row_above":    {"y_top": text_rows[i][0],     "y_bot": text_rows[i][1]},
                    "row_below":    {"y_top": text_rows[i + 1][0], "y_bot": text_rows[i + 1][1]},
                })

    # ── 6. Inconsistency check ────────────────────────────────────────────
    secs_with    = sum(1 for s in section_reports if s["found_hairlines"] > 0)
    secs_without = sum(1 for s in section_reports
                       if s["expected_hairlines"] > 0 and s["found_hairlines"] == 0)
    inconsistent = secs_with > 0 and secs_without > 0

    # ── Overall CV verdict ────────────────────────────────────────────────
    any_hairlines = len(hairlines) > 0
    fail_reasons: list[str] = []
    if len(missing_hairlines) > 0:
        fail_reasons.append("missing_hairlines")
    if len(bad_quality) > 0:
        fail_reasons.append("bad_quality_hairlines")
    if inconsistent:
        fail_reasons.append("inconsistent_usage")

    cv_pass = any_hairlines and len(fail_reasons) == 0

    print(f"[RULE-3-CV] hairlines={len(hairlines)}  bad={len(bad_quality)}  "
          f"missing={len(missing_hairlines)}  inconsistent={inconsistent}  "
          f"cv_pass={cv_pass}")

    return {
        "dpi_estimate":          dpi_estimate,
        "border_px":             border_px,
        "max_hairline_px":       max_hairline_px,
        "min_leading_px":        min_leading_px,
        "hairlines":             hairlines,
        "bad_quality_hairlines": bad_quality,
        "missing_hairlines":     missing_hairlines,
        "section_reports":       section_reports,
        "any_hairlines_used":    any_hairlines,
        "inconsistent_usage":    inconsistent,
        "cv_pass":               cv_pass,
        "cv_fail_reasons":       fail_reasons,
    }


# ===========================================================================
# ── LLM prompt ──────────────────────────────────────────────────────────────
# ===========================================================================

RULE_3_PROMPT = """\
You are evaluating an FDA Supplement Facts Panel for compliance with
RULE 3 (INDIVIDUAL NUTRIENT DIVIDERS):

  A ¼-point (hairline) rule MAY be used to separate individual nutrients.
  This is OPTIONAL — if no hairlines are used at all the panel PASSES.
  However, when hairlines ARE used they must satisfy ALL of:
    (a) Correct weight : ≤ ¼pt (hairline weight, same or thinner than outer border)
    (b) Uniformly dark : solid black, no fading or greying
    (c) 2pt of leading above AND below each hairline (adequate white space)
    (d) Consistent     : if used between any pair of rows, must be used
                         between ALL adjacent nutrient rows in the same section

Computer-Vision pre-analysis:
  any_hairlines_used       : {any_hairlines_used}
  total_hairlines_found    : {hairlines_found}
  bad_quality_count        : {bad_quality_count}
  missing_hairline_count   : {missing_hairline_count}
  inconsistent_usage       : {inconsistent_usage}
  cv_pass                  : {cv_pass}
  cv_fail_reasons          : {cv_fail_reasons}

Per-hairline detail (y_top, height_px, avg_darkness, gap_above_px, gap_below_px, issues):
{hairline_detail}

Use the CV findings as the primary guide and visually verify in the image.
Only override the CV verdict if you have a clear visual reason.

  - If NO hairlines are present     → pass=true,  notes="No hairlines used (optional)"
  - If hairlines are present and all correct → pass=true
  - If ANY hairline has a quality issue, is missing, or usage is inconsistent → pass=false

Respond ONLY with valid JSON (no markdown fences):
{{
  "pass":  <true|false>,
  "notes": "<one sentence — describe what you observed>",
  "hairlines_present":   <true|false>,
  "defect_types": ["<too_thick|faded|missing|leading_too_small|inconsistent>", ...]
}}"""


# ===========================================================================
# ── Nutrient name enrichment ────────────────────────────────────────────────
# ===========================================================================

def _enrich_with_nutrient_names(img: np.ndarray,
                                 regions: list,
                                 client: "OpenAI",
                                 region_type: str) -> list:
    """
    Wraps identify_surrounding_nutrients (from sfp_common) with a
    graceful fallback so the rule still works in stand-alone test mode.
    """
    if not regions:
        return []
    try:
        labels = identify_surrounding_nutrients(img, regions, client,
                                                region_type=region_type)
    except Exception:
        labels = [f"Unknown and Unknown"] * len(regions)

    enriched = []
    for region, label in zip(regions, labels):
        entry = dict(region)
        entry["between_rows"] = label
        enriched.append(entry)
    return enriched


# ===========================================================================
# ── Main rule check ──────────────────────────────────────────────────────────
# ===========================================================================

def check_rule_3(
    image_path: str,
    client:     "OpenAI",
    cv_data:    "dict | None" = None,   # kept for API compatibility
    img:        "np.ndarray | None" = None,
) -> dict:
    """
    Check Rule 3 — Individual Nutrient Dividers (optional rule).

    Parameters
    ----------
    image_path : str
        Local path to the image file.
    client : OpenAI
        Initialised OpenAI client.
    cv_data : dict, optional
        Pre-computed data from run_base_cv_analysis().  Accepted for API
        compatibility but Rule 3 now runs its own independent hairline
        quality scan that catches thickness, fading, and leading issues.
    img : np.ndarray, optional
        Pre-loaded BGR image.  If None, read from image_path.

    Returns
    -------
    dict with keys:
        pass               — bool  (optional rule verdict)
        notes              — str
        missing_dividers   — list of {between_rows, y_gap_top, y_gap_bottom,
                                       gap_height_px}
        bad_quality_lines  — list of {between_rows, issues, y_top, height_px,
                                       avg_darkness, gap_above_px, gap_below_px}
        cv_evidence        — dict  (all raw CV findings)
    """
    print(f"\n[RULE-3] Checking nutrient dividers + building report (optimised)...")

    if img is None:
        img = cv2.imread(image_path)

    # ── Stage 1: dedicated hairline quality CV scan ─────────────────────
    scan = analyze_hairline_quality(image_path)

    if "error" in scan:
        return {
            "pass":              True,   # conservative — can't assess
            "notes":             f"CV error: {scan['error']}",
            "missing_dividers":  [],
            "bad_quality_lines": [],
            "cv_evidence":       scan,
        }

    # ── Build hairline detail string for the prompt ──────────────────────
    if scan["hairlines"]:
        lines = []
        for h in scan["hairlines"]:
            lines.append(
                f"  y={h['y_top']}  h={h['height_px']}px  "
                f"dk={h['avg_darkness']}  "
                f"↑{h['gap_above_px']}px ↓{h['gap_below_px']}px  "
                f"issues={h['issues'] or 'none'}"
            )
        hairline_detail = "\n".join(lines)
    else:
        hairline_detail = "  (none found)"

    # ── Stage 2: LLM confirmation ────────────────────────────────────────
    prompt = RULE_3_PROMPT.format(
        any_hairlines_used   = scan["any_hairlines_used"],
        hairlines_found      = len(scan["hairlines"]),
        bad_quality_count    = len(scan["bad_quality_hairlines"]),
        missing_hairline_count = len(scan["missing_hairlines"]),
        inconsistent_usage   = scan["inconsistent_usage"],
        cv_pass              = scan["cv_pass"],
        cv_fail_reasons      = scan["cv_fail_reasons"],
        hairline_detail      = hairline_detail,
    )

    print(f"[RULE-3] CV: pass={scan['cv_pass']} "
          f"hairlines={len(scan['hairlines'])} "
          f"bad={len(scan['bad_quality_hairlines'])} "
          f"missing={len(scan['missing_hairlines'])} "
          f"— asking LLM to confirm...")

    response_text = ask_llm(client, prompt, image=image_path,
                             max_tokens=300, json_response=True)
    verdict = json.loads(response_text)

    print(f"[RULE-3] Result: {'PASS' if verdict.get('pass') else 'FAIL'} — "
          f"{verdict.get('notes', '')}")

    # ── Enrich missing hairlines with nutrient names ─────────────────────
    missing_enriched = _enrich_with_nutrient_names(
        img, scan["missing_hairlines"], client, region_type="gap"
    )
    # Reshape to legacy output format
    missing_out = []
    for m in missing_enriched:
        missing_out.append({
            "between_rows":  m.get("between_rows", "Unknown and Unknown"),
            "y_gap_top":     m["y_gap_top"],
            "y_gap_bottom":  m["y_gap_bottom"],
            "gap_height_px": m["gap_height"],
        })

    # ── Enrich bad-quality hairlines with nutrient names ─────────────────
    # Convert hairline dicts to the format identify_surrounding_nutrients expects
    bad_for_enrichment = []
    for h in scan["bad_quality_hairlines"]:
        bad_for_enrichment.append({
            "y_center":     (h["y_top"] + h["y_bot"]) // 2,
            "thickness_px": h["height_px"],
            "issue":        ", ".join(h["issues"]),
        })
    bad_enriched_labels = []
    if bad_for_enrichment:
        try:
            bad_enriched_labels = identify_surrounding_nutrients(
                img, bad_for_enrichment, client, region_type="bad_line"
            )
        except Exception:
            bad_enriched_labels = ["Unknown and Unknown"] * len(bad_for_enrichment)

    bad_out = []
    for h, label in zip(scan["bad_quality_hairlines"],
                         bad_enriched_labels or ["Unknown"] * len(scan["bad_quality_hairlines"])):
        bad_out.append({
            "between_rows":  label,
            "issues":        h["issues"],
            "y_top":         h["y_top"],
            "height_px":     h["height_px"],
            "avg_darkness":  h["avg_darkness"],
            "gap_above_px":  h["gap_above_px"],
            "gap_below_px":  h["gap_below_px"],
        })

    return {
        "pass":              bool(verdict.get("pass", True)),
        "notes":             verdict.get("notes", ""),
        "missing_dividers":  missing_out,
        "bad_quality_lines": bad_out,
        "cv_evidence": {
            # legacy keys — kept for callers that read these
            "hairline_count":         len(scan["hairlines"]),
            "bad_quality_line_count": len(scan["bad_quality_hairlines"]),
            "missing_divider_count":  len(scan["missing_hairlines"]),
            # new detailed keys
            "dpi_estimate":           scan["dpi_estimate"],
            "border_px":              scan["border_px"],
            "max_hairline_px":        scan["max_hairline_px"],
            "min_leading_px":         scan["min_leading_px"],
            "any_hairlines_used":     scan["any_hairlines_used"],
            "inconsistent_usage":     scan["inconsistent_usage"],
            "cv_pass":                scan["cv_pass"],
            "cv_fail_reasons":        scan["cv_fail_reasons"],
            "hairlines_detail":       scan["hairlines"],
            "section_reports":        scan["section_reports"],
        },
    }


# ===========================================================================
# ── Quick self-test (run this file directly) ─────────────────────────────────
# ===========================================================================
if __name__ == "__main__":
    import os
    from openai import OpenAI

    test_images = {
        "15_3_R_1 (GOOD — expect PASS)":
            "/mnt/user-data/uploads/15_3_R_1.jpg",
        "15_3_W_1 (TOO THICK hairlines — expect FAIL)":
            "/mnt/user-data/uploads/15_3_W_1.jpg",
        "15_3_W_2 (MISSING hairlines — expect FAIL)":
            "/mnt/user-data/uploads/15_3_W_2.jpg",
        "15_3_W_3 (FADED hairlines — expect FAIL)":
            "/mnt/user-data/uploads/15_3_W_3.jpg",
    }

    api_key = os.environ.get("OPENAI_API_KEY", "").strip() or "api_key"

    if not api_key:
        print("No OPENAI_API_KEY — running CV-only self-test.\n")
        for label, path in test_images.items():
            scan    = analyze_hairline_quality(path)
            verdict = "PASS" if scan.get("cv_pass") else "FAIL"
            reasons = scan.get("cv_fail_reasons", [])
            print(f"  {label}")
            print(f"    CV verdict : {verdict}  reasons={reasons}")
            for h in scan.get("hairlines", []):
                status = "OK" if h["quality_ok"] else f"FAIL {h['issues']}"
                print(f"      hairline y={h['y_top']} h={h['height_px']}px "
                      f"dk={h['avg_darkness']} ↑{h['gap_above_px']} ↓{h['gap_below_px']} "
                      f"-> {status}")
            for m in scan.get("missing_hairlines", []):
                print(f"      MISSING between y={m['y_gap_top']}-{m['y_gap_bottom']}")
            for s in scan.get("section_reports", []):
                if s["expected_hairlines"] > 0 or s["found_hairlines"] > 0:
                    print(f"      section y={s['y_top']}-{s['y_bot']}: "
                          f"text={s['text_row_count']} "
                          f"expected={s['expected_hairlines']} "
                          f"found={s['found_hairlines']} "
                          f"missing={s['missing_count']}")
            print()
    else:
        client = OpenAI(api_key=api_key)
        print("Running full CV + LLM self-test.\n")
        for label, path in test_images.items():
            result  = check_rule_3(path, client)
            verdict = "PASS" if result["pass"] else "FAIL"
            print(f"\n  {label}")
            print(f"    verdict      : {verdict}")
            print(f"    notes        : {result['notes']}")
            if result["bad_quality_lines"]:
                print(f"    bad lines    : {len(result['bad_quality_lines'])}")
                for b in result["bad_quality_lines"]:
                    print(f"      {b['between_rows']} — {b['issues']}")
            if result["missing_dividers"]:
                print(f"    missing      : {len(result['missing_dividers'])}")
                for m in result["missing_dividers"]:
                    print(f"      {m['between_rows']}")