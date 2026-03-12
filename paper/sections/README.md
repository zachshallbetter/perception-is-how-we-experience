# Main Paper Sections

This directory is the canonical source for paper claims, scope, and implementation alignment.

Current emphasis:

- main validated quantitative evidence: tuned-core 6-seed Split-CIFAR-100 continual-learning result
- main active perception-side evidence path: `sensor_streaming` canonical-vs-merged routing ladder
- generated artifacts live under `../figures/generated/results_package/`

## Canonical Claim Sources

Authoritative files for current paper claims:

1. `00-implementation-status-matrix.md`
2. `01-abstract.md`
3. `02-introduction.md`
4. `03-related-work.md`
5. `04-methods.md`
6. `05-experiments.md`
7. `06-results.md`
8. `07-discussion.md`
9. `08-conclusion.md`
10. `10-concept-implementation-audit.md`

`09-examples.md` is non-empirical and illustrative only.

## Recommended Reading Paths

If you want:

- the shortest current claim surface: read `01-abstract.md`, `06-results.md`, and `07-discussion.md`
- the implementation boundary and honesty constraints: read `00-implementation-status-matrix.md` and `10-concept-implementation-audit.md`
- the broader user/product interpretation: pair the paper with `../Paper Overview.md`, `../../README.md`, and `../../docs/productization-roadmap.md`

## Monolith Generation Policy

`../Perception Is How We Experience.md` is a generated artifact assembled from canonical sections.
Edit section files first, then regenerate the monolith.

## Assembly

```bash
cat 01-abstract.md 02-introduction.md 03-related-work.md \
    04-methods.md 05-experiments.md 06-results.md \
    07-discussion.md 08-conclusion.md \
    00-implementation-status-matrix.md 10-concept-implementation-audit.md \
    > "../Perception Is How We Experience.md"
```
