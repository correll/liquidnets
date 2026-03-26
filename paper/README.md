# CoRL Paper Template (Draft)

This folder contains a CoRL-style draft for your Liquid Push-T experiments.

## Files
- `main.tex` — paper draft with equations and pseudocode
- `references.bib` — starter bibliography

## Compile
1. Ensure the official CoRL style file is available in this folder (e.g., `corl_2024.sty`, depending on the current CoRL release).
2. Compile with:
   - `pdflatex main.tex`
   - `bibtex main`
   - `pdflatex main.tex`
   - `pdflatex main.tex`

## Notes
- The draft summarizes the exact training ideas you implemented:
  - CfC encoder
  - autoregressive decoder
  - scheduled sampling
  - hybrid teacher-forced/free-running loss
  - warmup + cosine LR schedule
- Update title/authors and CoRL style package name to the official year version if needed.
