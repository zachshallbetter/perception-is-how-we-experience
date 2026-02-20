# Main Paper Sections

This directory is structured to support section-by-section editing and modular paper development.

## Section Files

1. **01-abstract.md** - Abstract (150-250 words)
2. **02-introduction.md** - Introduction & motivation
3. **03-related-work.md** - Related work & positioning
4. **04-methods-updated.md** - Technical methods & architecture (code-aligned)
5. **05-experiments.md** - Experimental setup & design
6. **06-results-updated.md** - Results & findings (phase-based)
7. **07-discussion-updated.md** - Discussion & implications (implementation-aligned)
8. **08-conclusion.md** - Conclusion & future work
9. **00-implementation-status-matrix.md** - Claim/status/evidence matrix

## Status

Currently all sections are in the main paper file: `../Perception Is How We Experience.md`

**Next Step (Week 2-3)**: Split main paper into individual sections for easier editing during final review.

## Assembly

To generate submission PDF:
```bash
cat 01-abstract.md 02-introduction.md 03-related-work.md \
    04-methods-updated.md 05-experiments.md 06-results-updated.md \
    07-discussion-updated.md 08-conclusion.md > final-paper.md
```

---

See `../Perception Is How We Experience.md` for the complete current paper.
