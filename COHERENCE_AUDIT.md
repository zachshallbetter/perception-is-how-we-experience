# Paper Coherence Audit - Feb 20, 2026

## Summary
The paper has achieved significant consolidation (all 9 sections complete) and contains solid empirical results (M1 baseline/tuned on Split-CIFAR-100). However, there are **critical misalignments** between Methods, Results, and Discussion sections that must be resolved before submission.

---

## Critical Issues (Must Fix)

### 1. **Parameter Isolation: Methods vs. Results Mismatch**
**Severity: HIGH** — Architecture-Results misalignment

**Problem:**
- **Methods §4.5** describes "Parameter Isolation via Task Adapters" as a core architectural component
- **Methods §4.6** lists "Baseline" and "Tuned" training configurations, but **neither mentions parameter isolation enabled**
- **Results §6.1** reports results for "Baseline" (replay_weight=0.1) and "Tuned" (replay_weight=0.5), but **neither used parameter isolation**
- **Results §6.1.3** concludes: "NT-Xent replay loss provides insufficient consolidation pressure in either configuration tested" and recommends "architectural extensions" (§7.9)
- **Discussion §7.9** lists future work as: Logit Replay, Hierarchical Grouping, etc. — but **does NOT mention Parameter Isolation** as a future extension

**Reality:** Parameter isolation WAS implemented in code (TaskAdapter, freeze_task()) during Phase 2 work but was NOT tested in the reported results.

**Resolution Options:**
1. **Immediate (Recommended)**: Update Methods §4.5 to move parameter isolation from "core architecture" to "Phase 2 extensions being evaluated." Add a note in Results §6.1.3 stating "Parameter isolation was designed to address this failure mode and results are pending for Phase 2 validation."

2. **Alternative**: Add a "Phase 2 Preliminary Findings" subsection in Discussion §7.9 explaining parameter isolation design and indicating results are pending.

3. **If not testing parameter isolation**: Remove §4.5 entirely and move parameter isolation discussion to §7.9 as future work.

**Action:** Clarify which configuration actually used parameter isolation (if any) or explicitly defer results to Phase 2.

---

### 2. **"Results" Section Referenced But Not Fully Delivered**
**Severity: MEDIUM** — Documentation gap

**Problem:**
- Experiments §6 states: "Detailed ablation results (impact of replay, MIME-Type tokens, and attention), efficiency analysis, and application domain evaluations are presented in the Results section"
- Results §6.2-6.5 show ablation and domain results, BUT these are all **marked as SYNTHETIC** (not real empirical data)
- Conversely, only §6.1 has real empirical data (Split-CIFAR-100), which is relatively brief

**Reality:** The paper describes a full validation pipeline (4 experiments) but only delivers empirical data for Experiment 2 (continual learning). Experiments 1, 3, 4 are marked as "planned future validation."

**Resolution:**
- Retitle Results §6.2-6.5 as "Synthetic Architectural Validation" (currently labeled but could be clearer in context)
- Add a clear summary table in Results §6.6 distinguishing empirical vs. synthetic results
- Explicitly state in paper Abstract or Methods that full real-world validation is future work

---

### 3. **Contradictory Tuning Analysis**
**Severity: MEDIUM** — Confusing interpretation of results

**Problem:**
- Tuned configuration shows:
  - **CIL: 5.02% → 5.16%** (+0.14 pp, minimal improvement)
  - **BWT: −53.6% → −57.6%** (−4.0 pp, WORSENING)
  - **FWT: 0.00% → 0.00%** (no change)
  - **Within-task avg: 58.0% → 57.0%** (−1.0 pp, degradation)

- Results §6.1.3 correctly interprets: "BWT worsens by 4 pp. The increased learning rate... simultaneously increases the amplitude of the forgetting gradient applied to T0–T8. The net effect is near-zero."

**But:** Introduction and Abstract promise "improved scalability and generalizability" and discuss "replay mechanism improves continual learning" — creating expectation of improvement, while actual empirical results show replay alone is insufficient.

**Resolution:**
- **Abstract:** Adjust language from "results supporting hypotheses" to "preliminary validation with identified limitations" or "proof-of-concept results identifying necessary extensions"
- **Introduction:** Add explicit discussion of "catastrophic forgetting" phenomenon and why NT-Xent replay alone is insufficient (currently under-emphasized)
- **Results §6.1.3:** Is well-written; keep as-is

---

## Major Issues (Should Fix)

### 4. **Citation Quality**
**Severity: MEDIUM** — Some citations are incorrect or incomplete

**Current issues in references:**
- **[17] Bubeck et al. (2023)**: "Sparks of artificial general intelligence: Early experiments with GPT-4" — This is about LLMs, not relevant to Sensory Perceiver or Global Workspace Theory. This may be a template citation.
- **[18] Hassabis et al. (2017)**: "Neuroscience-inspired artificial intelligence" — This is correct but is cited in Discussion for Gestalt principles and Global Workspace Theory; the reference might be incomplete or there's a better citation (Baars 1988 on GWT should be cited explicitly if discussed).
- **Missing reference:** Global Workspace Theory (Baars, 1988) mentioned in Background §2.2 but not explicitly cited; reference [18] is Hassabis et al., not Baars.

**Resolution:**
- [ ] Verify all 34 BibTeX entries are correct and referenced in-text
- [ ] Add proper citation for Baars (1988) Global Workspace Theory
- [ ] Correct or replace reference [17] with a relevant citation
- [ ] Cross-check citation labels match the references.bib file

---

### 5. **Methods §4.6 Hyperparameter Table Ambiguity**
**Severity: LOW** — Minor clarity issue

**Problem:**
- Hyperparameter table lists "Baseline" and "Tuned" configurations
- Table does NOT explicitly state "parameter_isolation: False" for either
- If reader assumes parameter isolation is part of the architecture (per §4.5), they may be confused about whether these results include it

**Resolution:**
- Add column to table: `Parameter Isolation | False | False`
- Add footnote: "Phase 2 evaluates parameter isolation enabled; results pending."

---

## Moderate Issues (Nice to Have)

### 6. **Synthetic vs. Empirical Clarity**
**Current state:** Results §6.2-6.5 begin with disclaimers about synthetic data, which is good.

**Improvement:** Add a **Summary Table in §6.6** comparing all results:

```
| Experiment | Type | Status | Key Finding |
|---|---|---|---|
| 6.1.1 Baseline | Empirical | ✅ Complete | CIL 5.02%, BWT −53.6% |
| 6.1.2 Tuned | Empirical | ✅ Complete | CIL 5.16%, BWT −57.6% (worse) |
| 6.2 Ablation | Synthetic | ✅ Complete | Replay +9%, MIME +12%, Attention +16% |
| 6.3 Efficiency | Synthetic | ✅ Complete | 48ms latency, 5.2M params |
| 6.4 Scalability | Synthetic | ✅ Complete | 6% degradation 4→16 modalities |
| 6.5 Domains | Synthetic | ✅ Complete | +7% vs Perceiver IO avg |
```

---

### 7. **Figure/Table Cross-Reference**
**Current state:** Results §6.2 mentions "Figure 2" and §6.3 mentions "Figure 3", but I don't see figures embedded.

**Check:** Verify all figures are actually present in the sections/ files. If missing, either:
- [ ] Add placeholder sections for figures with captions
- [ ] Remove references to missing figures
- [ ] Add figures to the paper

---

### 8. **Introduction: Catastrophic Forgetting Emphasis**
**Current state:** Introduction §2.2 mentions catastrophic forgetting briefly but doesn't emphasize it as the core problem being solved.

**Suggestion:** Add a sentence in Intro explicitly framing the problem: "A key challenge in continual learning is catastrophic forgetting—the tendency for neural networks to lose performance on prior tasks when training on new ones. The Sensory Perceiver addresses this through experience replay and parameter isolation mechanisms."

---

## Minor Issues (Polish)

### 9. **Consolidated Paper vs. Modular Sections**
**Current state:** Two paper versions exist:
- `paper/Perception Is How We Experience.md` (~1,279 lines) — appears to be older/incomplete
- `paper/sections/01-09.md` — modular, up-to-date, contains correct results

**Recommendation:**
- [ ] Update the main consolidated file to reflect sections 01-09 content, OR
- [ ] Remove the consolidated file and use sections/ only in submission

---

## Checklist for Paper Readiness

**BEFORE SUBMISSION, verify:**

- [ ] **Parameter Isolation clarity**: Methods §4.5 either removed OR explicitly states "Phase 2, results pending"
- [ ] **Citation quality**: All 34 references correct and relevant; [17] and [18] verified
- [ ] **Empirical vs. Synthetic**: Results §6 clearly distinguishes with summary table
- [ ] **Tuning paradox addressed**: Abstract/Intro revised to acknowledge that replay alone is insufficient
- [ ] **Figures present**: All referenced figures (Figure 2, 3, 6) exist with captions
- [ ] **Consolidated file updated**: Main paper reflects latest section files
- [ ] **Hyperparameter table**: Shows `parameter_isolation=False` for Baseline/Tuned
- [ ] **Cross-references**: All section links work (especially Discussion →  Results callouts)

---

## Path Forward

**Immediate Actions (1-2 hours):**
1. Fix parameter isolation documentation (Methods §4.5 → clarify Phase 2 status)
2. Update Results §6.6 with summary table (Empirical vs. Synthetic)
3. Verify all citations correct and present
4. Ensure figures are embedded/referenced correctly

**Before Final Submission:**
1. Full coherence pass (read paper end-to-end once more)
2. Verify all section files are in consolidated main paper if using single-file format
3. Citation formatting pass (ensure BibTeX matches references)
4. Coordinate Phase 2 results status (when parameter isolation testing begins)

---

**Status**: ✅ Architecture sound, Results documented, Main issues identified
**Priority**: Fix parameter isolation mismatch before advancing to Phase 2 testing
**Estimated effort**: 2-3 hours for fixes + 1 hour verification

