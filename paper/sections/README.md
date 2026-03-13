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

`09-examples.md` is a non-canonical archive file. It is illustrative only and must not be cited as current evidence.

## Claim Labels

- **Implemented**: present in the current code path
- **Evidenced**: backed by packaged results or completed reports
- **Conceptual**: architecture pattern not yet promoted into the canonical paper
- **Patent candidate**: novelty-oriented idea tracked in [paper/patents](/Users/zachshallbetter/Projects/Perception/paper/patents)
- **Blocked / open problem**: not runnable or not defined enough

Additional future-use design material lives under [reference/future-concepts/README.md](/Users/zachshallbetter/Projects/Perception/reference/future-concepts/README.md) and must not be treated as canonical evidence.

## Recommended Reading Paths

If you want:

- the shortest current claim surface: read [01-abstract.md](/Users/zachshallbetter/Projects/Perception/paper/sections/01-abstract.md), [06-results.md](/Users/zachshallbetter/Projects/Perception/paper/sections/06-results.md), and [07-discussion.md](/Users/zachshallbetter/Projects/Perception/paper/sections/07-discussion.md)
- the implementation boundary and honesty constraints: read [00-implementation-status-matrix.md](/Users/zachshallbetter/Projects/Perception/paper/sections/00-implementation-status-matrix.md) and [10-concept-implementation-audit.md](/Users/zachshallbetter/Projects/Perception/paper/sections/10-concept-implementation-audit.md)
- the broader user/product interpretation: pair the paper with [Paper Overview.md](/Users/zachshallbetter/Projects/Perception/paper/Paper%20Overview.md), [README.md](/Users/zachshallbetter/Projects/Perception/README.md), and [productization-roadmap.md](/Users/zachshallbetter/Projects/Perception/docs/productization-roadmap.md)

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
