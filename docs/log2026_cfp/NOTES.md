# LoG 2026 — Call for Papers, consolidated (pulled 2026-07-20)

Source: https://logconference.org/cfp, https://logconference.org/conduct,
https://logconference.org/call-for-reviewers, and the official style-file
package (`style_files/`, downloaded from
https://logconference.org/cfp/Submission_Guidelines_and_Formatting_Instructions_for_LoG_Conference_2026.zip).

## Key dates (all "Anywhere on Earth")
- **Jul 29, 2026** — Abstract submission deadline (both tracks) — title + short abstract via OpenReview
- **Aug 1, 2026** — Full paper/extended-abstract submission deadline (both tracks)
- Aug 8–22 — review period
- Aug 25 — rebuttal starts
- Sep 1 — rebuttal ends, author-reviewer discussion starts
- Sep 8 — discussion ends
- Sep 13 — final decisions
- Oct 15 — camera-ready deadline
- Nov 20–22, 2026 (tentative) — conference, Northeastern University, Boston (in-person; at least one author must attend, or paper excluded from proceedings/program)

## Tracks
- **Proceedings track**: up to 9 pages + unlimited refs/appendix, published in PMLR, archival (cannot be already published/under review elsewhere), spotlights available.
- **Extended Abstract track** (this is our track): up to 4 pages + unlimited refs/appendix, **non-archival** (authors keep copyright; acceptance doesn't block publishing elsewhere), dual submission allowed. Explicitly welcomes "insightful negative results," "preliminary results warranting rapid dissemination," novel datasets, reproducibility studies — i.e. exactly the shape of what we have (a real negative result — live-student-as-target instability — plus a real, if early, positive distillation result). Eligible for "abstract spotlights." Same OpenReview submission link as proceedings track; track chosen at submission time.

## Scope
"Broadly related to learning on graphs and geometry." Topic list (non-exhaustive) includes Graph Generative Models, Geometric/graph generative models (Diffusion, Flow Matching, ...), Graph/Geometric ML for Physical sciences, Graph/Geometric ML for Molecules — our fit (CSPNet crystal-graph backbone + diffusion) sits squarely inside this, not a stretch. If in doubt, contact pcs@logconference.org.

## Review process
- Double-blind: OpenReview-hosted, public discussion allowed, reviewer identities anonymous (program chairs can see them).
- Anonymous GitHub (https://anonymous.4open.science/) recommended for any linked code.
- Preprint policy: existing non-anonymous preprints do NOT cause rejection; may cite own preprint anonymously (don't cite it in a way that reveals identity).
- No separate supplementary-material submission — appendices go in the same PDF, and **reviewers are not obliged to read appendices**. Anything load-bearing for the review must be in the 4-page body.
- Post-decision: accepted papers deanonymized; rejected ones can opt out of deanonymization.
- No published numeric scoring rubric (checked call-for-reviewers page too) — reviewers rated by authors/ACs for quality, top reviewers get monetary rewards, but no public rubric for novelty/soundness/etc.

## Formatting (from log_2026.tex / log_2026.sty)
- LaTeX only, official style file `log_2026.sty` — no alternative accepted.
- Package invocation for our case: `\usepackage[review,eabstract]{log_2026}` (anonymized extended-abstract submission). Later, accepted camera-ready uses `\usepackage[eabstract]{log_2026}` (no `review`).
- One-column, US letter, text block 5.5in × 9in, 1.5in left/right margins, 1in top margin.
- Times New Roman body text, Computer Modern for math, Type-1 fonts only in the final PDF.
- Title: 14pt bold, centered between two 1pt rules, Initial Caps.
- Abstract: single paragraph, **4–6 sentences** (per the template's own example abstract's stated ideal).
- Third-person self-citation in the anonymized version (no "in our earlier work we...").
- No acknowledgements/author-contributions section in the review version (added only in camera-ready; doesn't count toward the page limit).
- `[review]` option adds line numbers for reviewers and strips author info.
- Natbib (`numbers,compress,sort`) or bibLaTeX both fine; any citation style OK if consistent.
- Style file itself does NOT enforce the page limit programmatically (no LaTeX-level truncation/error) — it's policy-enforced by chairs/reviewers, so we must self-police the 4-page cap.

## Local copies
- `style_files/Submission Guidelines and Formatting Instructions for LoG Conference 2026/`
  - `log_2026.tex` — annotated example source (read this over the PDF; more precise)
  - `log_2026.sty` — the required style file
  - `log_2026.pdf` — rendered example output
  - `reference.bib` — example bib file

## Open items / not found on the public site
- No numeric review rubric published.
- Google Form link for reviewer signup not fetched (not needed for submission).
- Templates page / FAQ page not present as separate nav items — CFP page is the single source of truth.
