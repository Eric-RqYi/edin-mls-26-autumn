#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError as e:
    raise SystemExit(
        "PyMuPDF is required. Install with:\n"
        "  python -m pip install -r requirements.txt"
    ) from e


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def load_rules(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _page_structure(page):
    data = page.get_text("dict")
    lines = []
    image_blocks = []

    for block in data.get("blocks", []):
        if block.get("type") == 1:
            bbox = block.get("bbox")
            if bbox:
                image_blocks.append(tuple(bbox))
            continue

        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue

            text = "".join(s.get("text", "") for s in spans).strip()
            if not text:
                continue

            x0 = min(s["bbox"][0] for s in spans)
            y0 = min(s["bbox"][1] for s in spans)
            x1 = max(s["bbox"][2] for s in spans)
            y1 = max(s["bbox"][3] for s in spans)

            lines.append({
                "text": text,
                "bbox": (x0, y0, x1, y1),
                "width": x1 - x0,
                "chars": len(text),
                "spans": spans,
            })

    return lines, image_blocks


def _percentile(values, p):
    if not values:
        return None
    vals = sorted(values)
    pos = (len(vals) - 1) * p
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vals[lo])
    frac = pos - lo
    return float(vals[lo] * (1 - frac) + vals[hi] * frac)


def _weighted_mode_font_size(spans):
    weights = {}
    for span in spans:
        text = span.get("text", "").strip()
        if not text:
            continue
        size = round(float(span["size"]), 1)
        weights[size] = weights.get(size, 0) + max(1, len(text))

    if not weights:
        return None, {}
    return max(weights, key=weights.get), weights


def _is_ruler_line(line):
    return (
        re.fullmatch(r"\d{3}", line["text"].strip()) is not None
        and 50 <= line["bbox"][0] <= 100
    )


def _looks_like_body_line(line, expected_header_norm=""):
    text_norm = _norm(line["text"])
    if expected_header_norm and expected_header_norm in text_norm:
        return False
    if _is_ruler_line(line):
        return False

    x0, y0, x1, y1 = line["bbox"]

    # Ignore page header/footer and left-side ruler.
    if y0 < 55 or y1 > 742 or x0 < 98:
        return False

    # Body prose, headings, captions and equations with visible text can all
    # establish that main-body material appears before References.
    return line["chars"] >= 4 and line["width"] >= 20


def detect_references_boundary(doc, expected_header_norm=""):
    """
    Locate an exact References heading and estimate the number of main-body
    pages.

    Rule:
    - If substantive text or an image/figure appears above References on the
      same page, that page counts as a main-body page.
    - If References is the first substantive content on the page, main body
      ends on the previous page.

    Returns a dict with reliable=False if the boundary is missing/ambiguous.
    """
    matches = []

    for page_index, page in enumerate(doc):
        lines, image_blocks = _page_structure(page)

        for line in lines:
            if _norm(line["text"]) in {"references", "reference"}:
                matches.append({
                    "page_index": page_index,
                    "page_number": page_index + 1,
                    "line": line,
                    "all_lines": lines,
                    "image_blocks": image_blocks,
                })

    if len(matches) != 1:
        return {
            "reliable": False,
            "reason": (
                "References heading not found"
                if len(matches) == 0
                else f"Multiple References headings found ({len(matches)})"
            ),
            "matches": len(matches),
            "main_body_pages": None,
        }

    m = matches[0]
    ref_y0 = m["line"]["bbox"][1]

    substantive_text_before = []
    for line in m["all_lines"]:
        if line is m["line"]:
            continue
        if line["bbox"][3] >= ref_y0 - 1:
            continue
        if _looks_like_body_line(line, expected_header_norm):
            substantive_text_before.append(line)

    image_before = [
        bbox for bbox in m["image_blocks"]
        if bbox[3] < ref_y0 - 1 and bbox[1] > 50
    ]

    same_page_main_body = bool(substantive_text_before or image_before)

    if same_page_main_body:
        main_body_pages = m["page_number"]
    else:
        main_body_pages = max(0, m["page_number"] - 1)

    return {
        "reliable": True,
        "references_page": m["page_number"],
        "references_y_pt": float(ref_y0),
        "references_starts_new_page": not same_page_main_body,
        "main_body_pages": main_body_pages,
        "substantive_lines_before_references": len(substantive_text_before),
        "images_before_references": len(image_before),
    }


def check_pdf(pdf_path: Path, rules: dict) -> dict:
    doc = fitz.open(pdf_path)

    issues = []
    warnings = []
    metrics = {
        "pages_total": doc.page_count,
        "per_page": [],
    }

    expected_header = _norm(rules.get("expected_header_contains", ""))

    pw = float(rules["paper_width_pt"])
    ph = float(rules["paper_height_pt"])
    pst = float(rules["paper_size_tolerance_pt"])

    exp_left = float(rules["expected_text_left_pt"])
    exp_right = float(rules["expected_text_right_pt"])
    htol = float(rules["horizontal_tolerance_pt"])

    exp_font = float(rules["body_font_size_pt"])
    ftol = float(rules["body_font_size_tolerance_pt"])

    header_hits = 0
    ruler_tokens_total = 0
    body_spans = []

    for pageno, page in enumerate(doc, start=1):
        rect = page.rect

        if (
            abs(rect.width - pw) > pst
            or abs(rect.height - ph) > pst
        ):
            issues.append(
                f"Page {pageno}: page size is "
                f"{rect.width:.1f} x {rect.height:.1f} pt; expected "
                f"{pw:.1f} x {ph:.1f} pt (US Letter)."
            )

        page_text = _norm(page.get_text("text"))
        if expected_header and expected_header in page_text:
            header_hits += 1

        lines, _ = _page_structure(page)

        # ICLR review ruler: count individual three-digit tokens in the left area.
        left_text = "\n".join(
            line["text"] for line in lines
            if line["bbox"][0] < 100
        )
        ruler_tokens = re.findall(r"(?<!\d)\d{3}(?!\d)", left_text)
        ruler_tokens_total += len(ruler_tokens)

        # Likely prose lines used to infer the body rectangle.
        candidates = [
            line for line in lines
            if 65 <= line["bbox"][1] <= 730
            and line["bbox"][0] >= 98
            and line["chars"] >= 30
            and line["width"] >= 180
            and not (
                expected_header
                and expected_header in _norm(line["text"])
            )
        ]

        lefts = [x["bbox"][0] for x in candidates]
        rights = [x["bbox"][2] for x in candidates]

        left_q10 = _percentile(lefts, 0.10)
        right_q95 = _percentile(rights, 0.95)

        if len(candidates) >= 8:
            if left_q10 is not None and abs(left_q10 - exp_left) > htol:
                issues.append(
                    f"Page {pageno}: body left edge ≈ {left_q10:.1f} pt; "
                    f"expected ≈ {exp_left:.1f} pt. Possible margin/template change."
                )

            if right_q95 is not None:
                delta = right_q95 - exp_right
                if delta > htol or delta < -28:
                    issues.append(
                        f"Page {pageno}: body right edge ≈ {right_q95:.1f} pt; "
                        f"expected ≈ {exp_right:.1f} pt. Possible text-width/"
                        "margin change."
                    )
                elif abs(delta) > htol:
                    warnings.append(
                        f"Page {pageno}: inferred right edge ≈ {right_q95:.1f} pt "
                        f"(expected ≈ {exp_right:.1f}). This may be caused by "
                        "short/ragged lines; inspect manually if needed."
                    )

        for line in candidates:
            body_spans.extend(line["spans"])

        metrics["per_page"].append({
            "page": pageno,
            "width_pt": float(rect.width),
            "height_pt": float(rect.height),
            "candidate_body_lines": len(candidates),
            "left_q10_pt": left_q10,
            "right_q95_pt": right_q95,
            "ruler_tokens": len(ruler_tokens),
        })

    # Course header / template fingerprint.
    if expected_header:
        if header_hits == 0:
            issues.append(
                "Expected INFR11269 coursework header was not found. "
                "This usually indicates the wrong or modified template."
            )
        elif header_hits < max(1, math.ceil(doc.page_count * 0.6)):
            warnings.append(
                f"Coursework header detected on only {header_hits}/"
                f"{doc.page_count} pages."
            )

    # Review line-number ruler.
    if rules.get("require_review_line_numbers", True):
        if ruler_tokens_total < max(10, doc.page_count * 5):
            issues.append(
                "ICLR review line-number ruler was not reliably detected. "
                "Do not enable \\iclrfinalcopy and do not replace the review style."
            )

    # Dominant body font.
    mode_size, font_hist = _weighted_mode_font_size(body_spans)
    metrics["dominant_body_font_size_pt"] = mode_size
    metrics["body_font_histogram_char_weighted"] = font_hist
    metrics["header_pages_detected"] = header_hits
    metrics["review_ruler_tokens_detected"] = ruler_tokens_total

    if mode_size is None:
        warnings.append("Could not determine dominant body font size.")
    elif abs(mode_size - exp_font) > ftol:
        issues.append(
            f"Dominant body font size is ≈ {mode_size:.1f} pt; "
            f"expected ≈ {exp_font:.1f} pt."
        )

    # Main-body / references boundary.
    boundary = detect_references_boundary(doc, expected_header)
    metrics["references_boundary"] = boundary
    metrics["estimated_main_body_pages"] = boundary.get("main_body_pages")

    limit = rules.get("main_body_page_limit")
    metrics["configured_main_body_page_limit"] = limit

    if not boundary["reliable"]:
        warnings.append(
            "Could not reliably detect the References boundary: "
            f"{boundary['reason']}. Main-body page count was not guessed."
        )
    else:
        if isinstance(limit, int) and limit > 0:
            if boundary["main_body_pages"] > limit:
                issues.append(
                    f"Estimated main body is {boundary['main_body_pages']} pages, "
                    f"exceeding the configured limit of {limit} pages."
                )

    status = "FAIL" if issues else ("WARN" if warnings else "PASS")

    return {
        "file": str(pdf_path),
        "status": status,
        "issues": issues,
        "warnings": warnings,
        "metrics": metrics,
    }
