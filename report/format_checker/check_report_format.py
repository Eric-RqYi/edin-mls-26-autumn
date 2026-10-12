#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from report_format_core import check_pdf, load_rules


def main():
    here = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser(
        description=(
            "Check an INFR11269 coursework report PDF for common "
            "format/template violations."
        )
    )
    ap.add_argument("pdf", type=Path)
    ap.add_argument(
        "--rules",
        type=Path,
        default=here / "course_report_rules.json",
    )
    ap.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional detailed JSON output.",
    )
    args = ap.parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"PDF not found: {args.pdf}")

    result = check_pdf(args.pdf, load_rules(args.rules))

    print("=" * 72)
    print("INFR11269 REPORT FORMAT CHECK")
    print("=" * 72)
    print(f"File:   {args.pdf}")
    print(f"Status: {result['status']}")
    print()

    if result["issues"]:
        print("FAILURES")
        for issue in result["issues"]:
            print(f"  - {issue}")
        print()

    if result["warnings"]:
        print("WARNINGS")
        for warning in result["warnings"]:
            print(f"  - {warning}")
        print()

    m = result["metrics"]
    print("MEASURED")
    print(f"  total PDF pages: {m['pages_total']}")
    print(f"  dominant body font: {m.get('dominant_body_font_size_pt')} pt")
    print(
        f"  coursework header pages: "
        f"{m.get('header_pages_detected')}/{m['pages_total']}"
    )
    print(f"  review-ruler tokens: {m.get('review_ruler_tokens_detected')}")

    boundary = m.get("references_boundary", {})
    if boundary.get("reliable"):
        print(f"  References starts on PDF page: {boundary['references_page']}")
        print(
            f"  References starts on a new page: "
            f"{boundary['references_starts_new_page']}"
        )
        print(
            f"  estimated main-body pages: "
            f"{boundary['main_body_pages']}"
        )
    else:
        print("  estimated main-body pages: unavailable")

    limit = m.get("configured_main_body_page_limit")
    if limit is None:
        print("  main-body page limit: not configured")
    else:
        print(f"  configured main-body page limit: {limit}")

    if args.json:
        args.json.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
        print(f"\nDetailed JSON: {args.json}")

    print(
        "\nNote: this checker validates visible PDF properties. "
        "A PDF alone cannot prove that the LaTeX style source was never edited."
    )

    return 1 if result["status"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
