# Student PDF format checker

Use this before submitting your report.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python check_report_format.py path/to/report.pdf
```

Optional detailed output:

```bash
python check_report_format.py path/to/report.pdf --json format_check.json
```

## What is checked?

The checker looks for common accidental or deliberate formatting changes:

- US Letter page size;
- ICLR-style text rectangle / margins;
- base body font size;
- INFR11269 course-template header;
- ICLR review line-number ruler;
- the beginning of the References section;
- the six-page main-body limit.

### Main-body page counting

The course allows References to use additional pages. The checker therefore
does **not** simply use the total PDF page count.

It finds the `References` heading and checks whether substantive main-body
content appears earlier on that same page:

- if References starts on a fresh page, that page is not counted;
- if the main body continues earlier on the same page, that page is counted.

If the checker cannot identify this boundary reliably, it produces a warning
instead of guessing.

## Current page-limit setting

`course_report_rules.json` enforces a maximum of **6 main-body pages**.

```json
"main_body_page_limit": 6
```

References and optional appendices do not count toward this limit. Reviewers
are not obliged to read appendices, so essential arguments and results belong
in the main body. The supplied template places references before the appendix;
preserve this order so the checker can identify the main-body boundary.

## Review

The checker measures visible PDF properties. Review any warnings in the PDF
and retain the supplied LaTeX style files.
