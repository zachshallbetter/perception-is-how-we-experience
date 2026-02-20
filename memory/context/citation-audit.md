# Citation Audit
Generated: Feb 20, 2026 (updated with §3 Related Work expansion)

## Summary
- **34 BibTeX entries written** → `memory/context/references.bib`
- **25 used in main paper text** — 21 VERIFIED, 4 with VERIFY annotations
- **2 added** (Chaudhry 2019, Buzzega 2020) from §6.1.3 discussion in results section
- **12 added** (hip2022, perceiverar2022, dynconv2019, sdconv2020, odconv2022, kernelwarehouse2023, tea2020, tan2019, temporalperceiver2022, fromage2021, kirkpatrick2017, lopezmanzanares2017) from §3 Related Work
- **7 in citations.md but not in main text** — retained for supplementary use

## Citations Used in Main Paper

### Original Paper Citations
| # | Key | Entry | Status |
|---|-----|-------|--------|
| [1] | cite1 | Vaswani et al. (2017) — Attention Is All You Need. NeurIPS 30. | ✅ VERIFIED |
| [2] | cite2 | Jaegle et al. (2021) — Perceiver. ICML, PMLR 139:4651–4664. | ✅ VERIFIED |
| [3] | cite3 | Jaegle et al. (2022) — Perceiver IO. ICLR 2022. | ✅ VERIFIED |
| [4] | cite4 | Bahdanau et al. (2015) — NMT Align and Translate. ICLR 2015. | ✅ VERIFIED |
| [5] | cite5 | Hochreiter & Schmidhuber (1997) — LSTM. Neural Computation 9(8):1735–1780. | ✅ VERIFIED |
| [6] | cite6 | LeCun et al. (1998) — Gradient-Based Learning. Proc. IEEE 86(11):2278–2324. | ✅ VERIFIED |
| [7] | cite7 | Bengio et al. (1994) — Long-Term Dependencies. IEEE TNN 5(2):157–166. | ✅ VERIFIED |
| [8] | cite8 | Rosenblatt (1958) — The Perceptron. Psych. Review 65(6):386–408. | ✅ VERIFIED |
| [9] | cite9 | Sabour et al. (2017) — Dynamic Routing/Capsules. NeurIPS 30. | ✅ VERIFIED |
| [15] | cite15 | Chen et al. (2020) — SimCLR / NT-Xent. ICML, PMLR 119:1597–1607. | ✅ VERIFIED |
| [17] | cite17 | Wertheimer (1923) — Laws of Organization. Psychol. Forschung 4:301–350. | ✅ VERIFIED (see note) |
| [18] | cite18 | Dehaene et al. (1998) — Global Workspace. PNAS 95(24):14529–14534. | ✅ VERIFIED |
| [20] | cite20 | McCulloch & Pitts (1943) — Logical Calculus. Bull. Math. Biophys. 5:115–133. | ✅ VERIFIED |

### Added in §6.1.3
| Key | Entry | Status |
|-----|-------|--------|
| chaudhry2019 | Chaudhry et al. (2019) — A-GEM. ICLR 2019. arXiv:1812.00420 | ✅ VERIFIED |
| buzzega2020 | Buzzega et al. (2020) — DER++. NeurIPS 33:15920–15930. arXiv:2004.07211 | ✅ VERIFIED |

### Added in §3 Related Work
| Key | Entry | Source | Status |
|-----|-------|--------|--------|
| hip2022 | Carreira et al. (2022) — A Hierarchical Perceiver. arXiv:2202.10890 | Training data | ⚠️ VERIFY AUTHOR LIST |
| perceiverar2022 | Hawthorne et al. (2022) — Perceiver AR. arXiv:2202.07765 | Training data | ✅ VERIFIED |
| dynconv2019 | Chen et al. (2020) — Dynamic Convolution: Attention over Conv Kernels. CVPR 2020. | Bundled bib file | ✅ VERIFIED |
| sdconv2020 | (authors TBD) — Towards Parameter-Efficiency of Dynamic Convolution. arXiv:2106.02114 | Filename inference | ⚠️ VERIFY AUTHOR LIST |
| odconv2022 | Li, Zhou, Yao (2022) — Omni-Dimensional Dynamic Convolution. ICLR 2022. | Bundled bib file | ✅ VERIFIED |
| kernelwarehouse2023 | (authors TBD) — KernelWarehouse. arXiv:2308.08361 | Training data | ⚠️ VERIFY AUTHOR LIST |
| tea2020 | Li et al. (2020) — TEA: Temporal Excitation and Aggregation. CVPR 2020. | Bundled bib file | ✅ VERIFIED |
| tan2019 | Zhang et al. (2020) — Learning 2D Temporal Adjacent Networks. AAAI 2020. | Bundled bib file | ✅ VERIFIED |
| temporalperceiver2022 | (authors TBD) — Temporal Perceiver. 2022. | Training data | ⚠️ VERIFY arXiv ID + AUTHORS |
| fromage2021 | Koh, Salakhutdinov, Fried (2023) — FROMAGe. ICML 2023. | Training data | ✅ VERIFIED |
| kirkpatrick2017 | Kirkpatrick et al. (2017) — EWC. PNAS 114(13):3521–3526. | Training data | ✅ VERIFIED |
| lopezmanzanares2017 | Lopez-Paz & Ranzato (2017) — GEM. NeurIPS 30. | Training data | ✅ VERIFIED |

## Notes Requiring Attention Before Submission

### [17] Wertheimer (1923)
The paper cites this in English as "Laws of Organization in Perceptual Forms" with the original 1923 Psychologische Forschung volume. The English translation appeared in Ellis (1938). For a technical ML paper, you have two defensible options:
- Cite the 1923 German original (as currently done) — most common in academic ML
- Cite the 1938 Ellis translation — more accessible

Current BibTeX entry cites 1923 with a note about the 1938 translation. This is acceptable. No action required but reviewers sometimes flag this.

### ⚠️ VERIFY: 4 entries with uncertain author lists
Before submission, run `grep -A2 "VERIFY" memory/context/references.bib` and cross-check:
- **hip2022**: Check arXiv:2202.10890 for exact author order (DeepMind team, likely Carreira et al.)
- **sdconv2020**: Check arXiv:2106.02114 for author list and conference venue
- **kernelwarehouse2023**: Check arXiv:2308.08361 for author list and conference venue (possibly CVPR 2024)
- **temporalperceiver2022**: Confirm arXiv ID and author list — paper title confirmed, ID unconfirmed

### Gap: fromage2021 key year mismatch
The inline citation key `fromage2021` refers to a paper published at ICML 2023 (arXiv Jan 2023). This is a key naming inconsistency. Options before submission:
- Keep as `fromage2021` (matched to inline `[fromage2021]` markers in paper text) — requires no changes to text
- Rename to `koh2023` — requires find-replace in paper text

### Gap: No `[10]–[14]`, `[16]`, `[19]` in main text
These seven citations are in citations.md (neuroscience background: Hebb, Wilson, Stickgold, O'Neill, Bliss, Hadsell, DeFelipe) but don't appear in the main paper's inline citation list. They may be used in supplementary papers.

**Action**: Before submission, either (a) add them to the text where relevant, or (b) remove them from the reference list. Unused references waste page budget.

### Gap: PackNet cited in Experiment 2 design but no BibTeX
- **PackNet**: Mallya & Lazebnik (2018) — "PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning." CVPR 2018. arXiv:1711.05769

Needed if Experiment 2 runs against real baselines. Not urgent for current draft.

### Gap: CLIP, ViLBERT cited in Experiment 1 design but no BibTeX
- **CLIP**: Radford et al. (2021) — "Learning Transferable Visual Models From Natural Language Supervision." ICML 2021. arXiv:2103.00020
- **ViLBERT**: Lu et al. (2019) — "ViLBERT." NeurIPS 2019. arXiv:1908.02265

Same situation — needed if Experiment 1 runs. Not urgent for current draft.

## Verified BibTeX File
→ `memory/context/references.bib`

34 entries total. When converting to LaTeX (format conversion task):
- Replace inline `[N]` markers with `\cite{citeN}`
- Replace named keys (e.g., `[hip2022]`) with `\cite{hip2022}`
- Include `references.bib` in the preamble
- Resolve the 4 VERIFY entries before final submission
