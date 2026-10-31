# ACIIDS 2027 paper: how to compile and edit

This folder holds the Springer LNCS version of the UzABSA-LLM paper for ACIIDS 2027.

| File | What it is |
|---|---|
| `samplepaper.tex` | The paper. Everything is in this one file, including all figures (drawn with TikZ/pgfplots, no image files). |
| `mybibliography.bib` | The references. |
| `llncs.cls`, `splncs04.bst` | Springer's LNCS class and bibliography style. Do not edit. |
| `samplepaper.bbl` | The generated bibliography. Keep it: Springer asks for it with the source files. |
| `samplepaper.pdf` | The compiled paper. |

## Requirements

A TeX distribution with `pdflatex`, `bibtex` and `latexmk`. MacTeX is already installed on this Mac (`/Library/TeX/texbin`). The paper uses only standard packages (newtx, pgfplots, tikz, tcolorbox, booktabs, cleveref, hyperref). If one is missing, install it with:

```bash
sudo tlmgr install <package-name>
```

## Compile with one command (recommended)

```bash
cd paper_materials/aciids
latexmk -pdf samplepaper.tex
```

`latexmk` runs `pdflatex` and `bibtex` as many times as needed and stops when the PDF is up to date. Run it again after every change.

## Compile by hand

```bash
cd paper_materials/aciids
pdflatex samplepaper
bibtex samplepaper
pdflatex samplepaper
pdflatex samplepaper
```

The first run collects citations and labels, `bibtex` builds the reference list, and the last two runs fill in reference numbers and cross-references. If you see `[?]` or `??` in the PDF, you skipped a run.

## Compile in VS Code

1. Install the **LaTeX Workshop** extension.
2. Open `samplepaper.tex`. Saving the file rebuilds the PDF (it uses `latexmk`).
3. `Cmd+Option+B` builds by hand; `Cmd+Option+V` opens the PDF next to the source.

## Compile on Overleaf

Upload `samplepaper.tex`, `mybibliography.bib`, `llncs.cls` and `splncs04.bst` to a new project and keep the compiler at pdfLaTeX.

## Check before submitting

```bash
grep "Output written" samplepaper.log   # must say 15 pages or fewer
grep -i "undefined" samplepaper.log     # must print nothing
grep "Overfull" samplepaper.log         # should print nothing
```

- **Page limit:** 12–15 pages including references. Page 15 is already full, so if you add a line anywhere you must cut one somewhere else. Both easy cuts are used up: the zero-shot rows of Table 3 and the "Formatting versus extraction" paragraph (now one sentence in Sect. 5.2) went on 31 October 2026 to make room for the vote, the Cyrillic result and the re-checked gold. Look instead for paragraphs whose last line holds one or two words.
- **Harmless output:** the warning `Unable to redefine math accent \vec` and the file `DescriptionTexts.txt` (written by `llncs.cls`) can be ignored.

## Clean up

```bash
latexmk -c    # deletes .aux, .log and other helper files; keeps the PDF and .bbl
latexmk -C    # also deletes the PDF; still keeps the .bbl
```

## Where to change things

Search `samplepaper.tex` for the text in the right-hand column.

| To change | Search for |
|---|---|
| Title, author, affiliation | `\title`, `\author`, `\institute` |
| Abstract, keywords | `\begin{abstract}`, `\keywords` |
| Main results table (Table 2) | `\label{tab:main}`; bold marks the best fine-tuned score per column and is set by hand |
| Matching-robustness table (Table 3) | `\label{tab:robust}` |
| Human-validation table (Table 4) | `\label{tab:human}` |
| Training curves (Fig. 3) | `\label{fig:loss}`; the data are `(step, value)` pairs after `coordinates` |
| Domain chart (Fig. 4) | `\label{fig:domains}` |
| Judge calibration (Fig. 5) | `\label{fig:judge}` |
| Chart colours | `\definecolor{cQwen}` and the lines below it, near the top of the file |
| Acknowledgments | `\ackname` |

Where the numbers come from (run the commands from the repository root):

- **Table 3:** `paper_materials/revision_v2/robustness/robustness.json`, regenerated with `python3 scripts/robustness_analysis.py`.
- **Judge numbers:** the judge-panel numbers in Sects. 4.4, 5.5 and 5.6, the judge columns of Table 4, and Figs. 4 and 5 come from `experiments/v3/results/judge_paper_numbers.json`. Regenerate it with `python3 experiments/v3/run_v3.py collect`, which needs no API calls; the panel's raw scores are in `experiments/v3/judge/`.
- **Re-checked gold (Sect. 5.3):** `experiments/v3/test_recheck/summary.json` (changes, `outcomes_by_reason`, scores) and `agreement.json` (κ), from `PYTHONHASHSEED=0 python3 scripts/test_recheck.py apply`. The p-values are in `test_recheck/significance/`.
- **Majority vote (Table 2, Sect. 5.2):** `experiments/v3/results/ensembles/majority_of_5_v2_systems_metrics.json`.
- **Script numbers:** the script shares in Sect. 3 and the Uzbek-Cyrillic transliteration result in Sect. 5.5 come from `experiments/v3/translit/report.json`. To regenerate:
  1. `python3 scripts/cyrillic_experiment.py annotate` downloads the Qwen checkpoint and annotates the 152 Uzbek-Cyrillic reviews twice, in about 40 min on an Apple M2 Max.
  2. `scripts/judge_panel.py score --rubric v1` judges each arm (the commands are in the script's docstring), for about US$0.60.
  3. `python3 scripts/cyrillic_experiment.py report` writes the report.

## Adding a reference

1. Add the entry to `mybibliography.bib`.
2. Wrap the title in a second pair of braces, e.g. `title = {{Uzbek Sentiment Analysis with LLMs}}`. Without them, `splncs04` lowercases words such as "Uzbek" or "LLMs".
3. Cite it with `\cite{key}` and recompile.

## Troubleshooting

| Problem | Fix |
|---|---|
| `File 'llncs.cls' not found` | Run the command inside `paper_materials/aciids`. |
| Citations show as `[?]` | Run `bibtex` and then `pdflatex` twice, or just use `latexmk`. |
| `File 'something.sty' not found` | `sudo tlmgr install something` |
| `latexmk` stops after an error | Fix the error it reports (the line number is in the message), then run `latexmk -C` and compile again. |
| The PDF does not change | Save the `.tex` file first; `latexmk` only rebuilds when a source file changed. |

## Submitting

- **Review:** upload `samplepaper.pdf` to EasyChair (https://easychair.org/conferences/?conf=aciids2027).
- **Camera-ready (if accepted):** the source files (`samplepaper.tex`, `mybibliography.bib`, `samplepaper.bbl`, `llncs.cls`, `splncs04.bst`), the final PDF and the signed copyright form.
