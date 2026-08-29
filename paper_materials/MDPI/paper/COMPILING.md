# Compiling main.tex

There is no LaTeX toolchain on this machine (checked 2026-08-29: no `pdflatex` on
PATH, no MiKTeX/TeX Live install found under the usual paths). Use one of the two
options below.

## Option A — Overleaf (no install, fastest)

1. Go to <https://www.overleaf.com>, sign in, **New Project → Upload Project**.
2. Zip and upload the whole `paper_materials/MDPI/paper/` folder (must include
   `main.tex`, `mybibliography.bib`, `wand-uzabsa.png`, and the `Definitions/`
   subfolder with `mdpi.cls`, the three `.bst` files, `journalnames.tex`, and the
   logo files).
3. Overleaf auto-detects `main.tex` as the root file. Set the compiler to
   **pdfLaTeX** (Menu → Settings → Compiler) if it doesn't default to it.
4. Click **Recompile**. Overleaf runs the full pdflatex → bibtex → pdflatex →
   pdflatex cycle for you on every build.
5. Download the resulting PDF, or keep working in Overleaf directly.

This is the recommended path if you just need a PDF to check the template
upgrade — no local setup, and it matches what MDPI's own editorial system uses.

## Option B — Install MiKTeX locally (for repeated local builds)

1. Download MiKTeX from <https://miktex.org/download> (Windows installer).
2. During install, set **"Install missing packages on the fly" → Yes** — the
   MDPI class pulls in several packages (`ifthen`, `adjustbox`, `pgfplots`,
   `multirow`, etc.) that MiKTeX will fetch automatically on first compile.
3. Open a **new** PowerShell window after install (PATH needs to refresh) and
   verify:
   ```powershell
   pdflatex --version
   ```
4. Compile from `paper_materials/MDPI/paper/`:
   ```powershell
   cd paper_materials\MDPI\paper
   pdflatex -interaction=nonstopmode main.tex
   bibtex main
   pdflatex -interaction=nonstopmode main.tex
   pdflatex -interaction=nonstopmode main.tex
   ```
   Four passes are required, not optional: pass 1 generates `main.aux` (the
   citation keys used); `bibtex` reads that and writes `main.bbl` (the formatted
   reference list) using `Definitions/mdpi.bst`; pass 2 pulls `main.bbl` into the
   document and generates correct `\cite` numbers; pass 3 resolves any
   cross-references/page numbers that shifted once the bibliography was inserted.
5. Check the tail of `main.log` for errors:
   ```powershell
   Select-String -Path main.log -Pattern "^!|Undefined|Overfull \\hbox" | Select-Object -Last 40
   ```
   `! ` at the start of a line is a fatal LaTeX error. "Undefined" citations/refs
   and "Overfull \hbox" are warnings, not failures — worth fixing before
   submission but they don't stop the PDF from being produced.
6. Open `main.pdf`.

## What to send back if it errors

If pdflatex stops with a `!` error, paste the ~15 lines of `main.log` starting
at the first `!` line (not the whole log — the actual cause is almost always
right there) plus the last 5 lines before the error.

## Template version note (2026-08-29)

`Definitions/` was upgraded from the 13 March 2026 MDPI class to 27 July 2026
this session (see `paper_materials/revision_v2/REVISION_LOG.md`). The old copy
is kept at `Definitions_backup_20260313/` for reference/rollback — it is not
used by `main.tex` and can be deleted once the new template is confirmed to
compile cleanly. `\documentclass` was updated to drop the now-removed `pdftex`
option to match.
