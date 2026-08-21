# architecture.pdf generator

Two scripts produce docs/architecture.pdf:

1. `make_diagrams.py` — renders Figures 1–7 as PNGs (matplotlib) into
   `docs/generator/diagrams/`.
2. `build_pdf.py` — assembles `docs/architecture.pdf` (reportlab).

Paths are repo-relative — no editing needed. Run from anywhere:

    .venv/bin/python docs/generator/make_diagrams.py
    .venv/bin/python docs/generator/build_pdf.py

Requirements (once): `.venv/bin/pip install matplotlib reportlab pillow`.

## Regeneration rule

The PDF is a rendered snapshot of PLAN.md + DECISIONS.md + phase specs — it is
never edited independently. To update it:

1. Read the DECISIONS.md entries and benchmarks/results.md rows added since the
   date in the PDF footer.
2. Edit the affected prose/tables in `build_pdf.py` (and diagram data in
   `make_diagrams.py` — e.g. replace planning estimates in Figure 3/5 with
   measured values once Phase 0/2 rows exist, and update the roadmap's phase
   status).
3. Bump the version/date string in `build_pdf.py` (footer + title block).
4. Run both scripts; commit the scripts, the PNGs, and the PDF together.

Trigger: end of each phase, or whenever DECISIONS.md gains an entry that changes
an invariant, an exit criterion, or a measured number shown in the document.
