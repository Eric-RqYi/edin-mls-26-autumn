# Coursework report

These resources support the whole coursework, including the ASR work and RAG
systems investigation.

## Step 1: open the template

Upload `report_template_overleaf.zip` to Overleaf as a new project. Set
`report.tex` as the main document. The archive contains the style files and
an example bibliography. For a local LaTeX installation, use `latex_template/`.

Replace the title, example content, references and group information with your
own material. Include the AI-use declaration supplied by the template.
See [the Overleaf guide](latex_template/README_OVERLEAF.md) for more details.

## Step 2: explain the investigation

The main body is limited to **6 pages**. References and optional appendices
are excluded. Reviewers are not obliged to read appendices, so essential
arguments, methods and results belong in the main body.

The suggested section headings are flexible. Your report should make it easy
to find the technical question, baseline, changes, experimental controls,
quality and cost results, interpretation and limitations. Explain what the
evidence supports and why the system behaves that way.

## Step 3: check the PDF

Preserve the supplied style, page geometry and review mode. From the repository
root, install the checker dependencies and check your compiled PDF:

```bash
python -m pip install -r report/format_checker/requirements.txt
python report/format_checker/check_report_format.py /path/to/your-report.pdf
```

The checker prints PASS, WARN or FAIL and the properties it measured. It
estimates main-body pages using the References heading. A page containing
both body text and the start of references counts as a body page. Inspect
warnings in the PDF, especially if it cannot locate the boundary reliably.
See [the checker guide](format_checker/README.md) for detailed output.
