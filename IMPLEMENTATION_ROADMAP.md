# Implementation Roadmap: Paths A, B, C

**Status**: ✅ Complete — All three paths planned and documented

**Timeline**: ~8 weeks total (can run Paths B & C in parallel after Path A paper updates)

---

## Summary: What Changed

### Path A: Paper Restructure (COMPLETED)

**Updated files**:
1. `paper/sections/01-abstract.md` — Phase 2 results integrated
2. `paper/sections/04-methods-updated.md` — Reservoir Sampling algorithm + tower-gradient explanation
3. `paper/sections/06-results-updated.md` — Phase 1 vs Phase 2 structure (4.86% → 19.54% CIL)
4. `paper/sections/07-discussion-updated.md` — Tower-gradient insight + Phase 3/4 roadmap

**Key narrative shift**:
- Old: "Feature-only replay insufficient; parameter isolation needed" (unclear why)
- New: "Raw-pixel replay essential because gradient signals must flow through PerceiverTowers" (mechanistic insight)

**Paper is now**:
- ✅ Scientifically sound (explains failure mode + solution)
- ✅ Submission-ready for Path A (Phase 2 results alone)
- ❌ Not yet competitive (19.54% CIL < EWC/GEM 30–40%)
- ❌ No multimodal validation (claims 8+ modalities, only tested on CIFAR-100)

### Path B: Phase 3 Plan (READY TO EXECUTE)

**Deliverable**: `PHASE3_PARAMETER_ISOLATION_PLAN.md`

**Objective**: Combine Phase 2 robust replay + task-specific adapters → 30–35% CIL (SOTA-competitive)

**Timeline**: 2–3 weeks

**Key steps**:
1. Extend `SensoryPerceiverWithIsolation` class with adapter freezing
2. Run Phase 3 training (5 seeds) on Split-CIFAR-100
3. Benchmark against EWC, GEM, PackNet
4. Update paper Results §6.3 with Phase 3 findings

**Success criteria**:
- CIL ≥ 30% (vs. 19.54% Phase 2)
- BWT ≥ −25% (vs. −34.91% Phase 2)
- Task 0 retention ≥ 40% (vs. 33% Phase 2)

**Post-Phase 3**: Paper ready for top-tier venues (with multimodal work in progress)

### Path C: Multimodal Roadmap (READY TO EXECUTE)

**Deliverable**: `PHASE4_MULTIMODAL_ROADMAP.md`

**Objective**: Validate MIME-Type encoding + cross-modal attention on real multimodal data (AudioSet, Ego4D, MSCOCO)

**Timeline**: 4–6 weeks (can overlap with Phase 3 weeks 2–3)

**Key datasets**:
- **AudioSet** (must-have): 2M audio clips, validate audio-only learning
- **Ego4D** (must-have): 1000+ hours video+audio+IMU, validate multimodal fusion
- **MSCOCO** (must-have): 330K images+captions, validate vision-language
- **AV-MNIST** (nice-to-have): Audio-visual digit recognition

**Success criteria**:
- Single-modality ≥ published baselines
- Multimodal fusion improves by ≥ 5% vs. single-modality
- Cross-modal attention is visualizable and interpretable

**Post-Phase 4**: Paper ready for publication (validated on 3–4 modalities, SOTA-competitive on vision-only)

---

## Execution Plan: Recommended Sequence

### Immediate (Today/Tomorrow)

1. ✅ **Read updated paper sections** (A-B-C-D are ready)
2. ✅ **Review Phase 3 plan** (`PHASE3_PARAMETER_ISOLATION_PLAN.md`)
3. ✅ **Review Phase 4 plan** (`PHASE4_MULTIMODAL_ROADMAP.md`)
4. ⏭️ **Decision**: Which path to prioritize?

### Next Week: Weeks 1–2 (Phase 3 Priority)

| Day | Task | Owner | Output |
|-----|------|-------|--------|
| Mon | Extend `SensoryPerceiverWithIsolation` class | Engineer | Code PR ready |
| Tue | Implement adapter freezing + verify gradients | Engineer | Unit tests pass |
| Wed–Thu | Run Phase 3 seed 42 (baseline) | Engineer | Single-seed results |
| Fri | Run Phase 3 remaining seeds (1–4) | Engineer | 5-seed ensemble |

**Week 2**:
| Day | Task | Owner | Output |
|-----|------|-------|--------|
| Mon–Tue | Benchmark vs. EWC/GEM/PackNet | Engineer | Comparison table |
| Wed | Ablation: isolation vs. replay | Engineer | 3-config comparison |
| Thu | Generate figures (confusion matrix, BWT chart) | Engineer | Figures 7–9 |
| Fri | Write Results §6.3 + update Discussion §7.9 | Author (Zach) | Paper draft ready |

**Week 1–2 Deliverables**:
- ✅ Phase 3 empirical results (5 seeds, error bars)
- ✅ Paper Results §6.3 integration
- ✅ Confirmation: CIL ≥ 30% (success) or fallback analysis

### Parallel: Weeks 2–3 (Phase 4 Infrastructure)

While Phase 3 benchmarking runs (which is compute-bound), start Phase 4 infrastructure:

| Day | Task | Owner | Output |
|-----|------|-------|--------|
| Mon | Download/prepare AudioSet subset | Engineer | 50K AudioSet clips ready |
| Tue | Build AudioPerceiverTower + data loader | Engineer | Audio tower tested |
| Wed | Download MSCOCO + BERT tower | Engineer | MSCOCO loader + TextPerceiverTower |
| Thu | Download/request Ego4D access | Engineer | Ego4D registered + queued |
| Fri | MIME-Type routing extended | Engineer | 4 modalities routing tested |

**Weeks 2–3 Deliverables**:
- ✅ Phase 4 infrastructure ready (data loaders, modality towers)
- ✅ Single-modality baselines running (AudioSet, MSCOCO)

### Week 3: Phase 4 Benchmarking

Once Phase 3 seed runs complete and infrastructure is ready:

| Day | Task | Owner | Output |
|-----|------|-------|--------|
| Mon–Tue | AudioSet single-modality baseline | Engineer | Results: 25% CIL |
| Wed–Thu | MSCOCO single-modality baseline | Engineer | Results: 42% CIL |
| Fri | Ego4D video-only baseline (if access ready) | Engineer | Results: 35% CIL |

**Week 4**: Ego4D multimodal (video+audio+IMU), cross-modal attention heatmaps

**Week 5**: Paper integration (Results §6.4, Discussion §7.5 update), figures

---

## Decision Tree: Which Path First?

**Question 1: Is Phase 3 required before publication?**

| Answer | Path | Timeline | Recommendation |
|--------|------|----------|---|
| Yes, SOTA competitive | B→A | 2 weeks | **Do Phase 3 first** |
| No, Phase 2 alone sufficient | A→C | 4 weeks | Publish Phase 2, work on multimodal |
| Want to maximize impact | B→C→A | 6 weeks | Phase 3 + Phase 4 in parallel |

**My recommendation**: **B→C (parallel) → A (publish when both ready)**

**Rationale**:
- Phase 2 (19.54% CIL) alone is NOT competitive with EWC/GEM (30–40%)
- Phase 3 (expected 30–35% CIL) IS competitive
- Phase 4 (multimodal validation) adds novelty and differentiates from other CL work
- Running B & C in parallel (infrastructure setup while Phase 3 computes) saves 2 weeks

**Timeline with B→C parallel**:
- Week 1: Phase 3 implementation + Phase 4 infrastructure
- Week 2: Phase 3 benchmarking + Phase 4 AudioSet/MSCOCO
- Week 3: Phase 3 paper integration + Phase 4 Ego4D
- Week 4: Paper finalization (Results §6.3–6.4, Discussion §7)
- Week 5: Ready for submission

**Total**: 5 weeks to submission-ready (vs. 8 weeks sequential)

---

## File Inventory: What's Ready

### Paper Sections (Path A)

| File | Status | Usage |
|------|--------|-------|
| `01-abstract.md` | ✅ Updated | Replace old abstract with new (19.54% CIL) |
| `04-methods-updated.md` | ✅ New | Rename to `04-methods.md`, replace old §4 |
| `06-results-updated.md` | ✅ New | Rename to `06-results.md`, replace old §6 |
| `07-discussion-updated.md` | ✅ New | Rename to `07-discussion.md`, replace old §7 |

### Planning Documents (Path B & C)

| File | Status | Usage |
|------|--------|-------|
| `PHASE3_PARAMETER_ISOLATION_PLAN.md` | ✅ New | Implementation guide for Phase 3 |
| `PHASE4_MULTIMODAL_ROADMAP.md` | ✅ New | Implementation guide for Phase 4 |
| `IMPLEMENTATION_ROADMAP.md` | ✅ New | This file (executive summary) |

### Code Status

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| `TaskAdapter` class | ✅ Exists | `src/models.py:45–78` | Ready for Phase 3 |
| `freeze_task()` method | ⏳ Needs implementation | `src/models.py` | Add in Phase 3, Week 1 |
| `ReservoirSamplingBuffer` | ✅ Exists | `scripts/run_local.py` | Used in Phase 2 results |
| `AudioPerceiverTower` | ⏳ Template exists | `PHASE4_MULTIMODAL_ROADMAP.md` | Implement in Phase 4, Week 2 |
| `TextPerceiverTower` | ⏳ Template exists | `PHASE4_MULTIMODAL_ROADMAP.md` | Implement in Phase 4, Week 2 |
| `IMUPerceiverTower` | ⏳ Template exists | `PHASE4_MULTIMODAL_ROADMAP.md` | Implement in Phase 4, Week 2 |
| Test suite (`test_parameter_isolation.py`) | ✅ Exists | `tests/` | Run in Phase 3, Week 1 to verify freezing |

---

## Next Immediate Actions

### For Zach (Paper Author)

1. **Read** all three updated paper sections (`01-abstract.md`, `04-methods-updated.md`, `06-results-updated.md`, `07-discussion-updated.md`)
2. **Decide**: Phase 3 first, or Phase 4 first, or parallel?
3. **Integrate**: Once Phase 3 completes, merge `04-methods-updated.md` → `04-methods.md` and regenerate `.docx` file

### For Engineer

1. **Phase 3 ready** (weeks 1–2):
   - Implement `freeze_task()` + `activate_task()` methods
   - Run seed 42 → confirm CIL ≥ 25%
   - Run 4 additional seeds → error bars

2. **Phase 4 infrastructure ready** (week 2–3):
   - AudioPerceiverTower + data loader
   - TextPerceiverTower (BERT) + MSCOCO loader
   - MIME-Type routing extension

### For Shadow Agent (if continuing)

- **Monitor** Phase 3 benchmarking for convergence
- **Prepare** AudioSet/MSCOCO data if Phase 4 starting in week 2

---

## Success Metrics: Submission Readiness

### Path A Alone (Phase 2 Results)

**Paper quality**: ⭐⭐⭐⭐ (scientifically sound, mechanistic insight)

**Competitiveness**: ⭐⭐ (19.54% CIL << SOTA 45–50%)

**Publishability**:
- ⭐⭐⭐ Local conference (accept)
- ⭐⭐ Top conference (borderline reject)

### Path A + B (Phase 2 + 3 Results)

**Paper quality**: ⭐⭐⭐⭐⭐ (sound + SOTA-competitive)

**Competitiveness**: ⭐⭐⭐⭐ (30–35% CIL ≈ EWC/GEM)

**Publishability**:
- ⭐⭐⭐⭐⭐ Top conference (accept)
- ⭐⭐⭐⭐⭐ Journal (accept with minor revisions)

### Path A + B + C (Phase 2 + 3 + 4 Results)

**Paper quality**: ⭐⭐⭐⭐⭐ (sound + SOTA + multimodal validated)

**Competitiveness**: ⭐⭐⭐⭐⭐ (30–35% vision-only + 45%+ multimodal)

**Publishability**:
- ⭐⭐⭐⭐⭐ Top venue (ICML/NeurIPS/ICCV accept)
- ⭐⭐⭐⭐⭐ Journal (Nature Machine Intelligence candidate)

---

## Risk Mitigation

### Phase 3 Risk: CIL doesn't reach 30%

**Probability**: 30% (adapter freezing may not help as much as expected)

**Fallback**: Implement logit replay (Phase 3b) for +2–5% CIL boost

**Mitigation**: Start logit replay coding in parallel with Phase 3 seed runs (weeks 1–2)

### Phase 4 Risk: Ego4D access denied

**Probability**: 10% (requires registration)

**Fallback**: Use UCF101 instead (action recognition, single-modality)

**Mitigation**: Register immediately (Week 2); request API access in parallel

### Phase 4 Risk: Multimodal fusion doesn't improve performance

**Probability**: 20% (late fusion may be sufficient)

**Fallback**: Ablate attention (freeze cross-modal layer); if no degradation, rethink MIME-Type value prop

**Mitigation**: Pre-test on Ego4D early (week 2–3); pivot if needed

---

## Final Checklist

### Before starting Phase 3:

- [ ] Read updated paper sections A–D
- [ ] Review Phase 3 plan
- [ ] Confirm code locations (TaskAdapter, freeze_task implementation)
- [ ] Schedule compute resources (GPU/TPU for 5-seed runs, ~40 GPU-hours)

### Before starting Phase 4:

- [ ] Download/prepare AudioSet (or FSD50K backup)
- [ ] Register for Ego4D access
- [ ] Download MSCOCO
- [ ] Review modality-specific tower templates

### Before publishing:

- [ ] Phase 3 complete (CIL ≥ 30%)
- [ ] Phase 4 at least tier 1 benchmarks done (Ego4D video+audio+IMU)
- [ ] Paper sections merged and proofread
- [ ] Code reproducibility verified (one clean run from scratch)

---

## Open Questions

1. **Which path to prioritize?** (B first, or B+C parallel?)
2. **Publication target?** (Venue determines urgency of Phase 4)
3. **Code release timeline?** (Affects supplementary material prep)
4. **Multimodal data availability?** (Ego4D access timeline?)

**Action**: Discuss with Zach; make explicit decision.

---

**Status**: ✅ All paths documented, ready for execution.

**Next step**: Execute Phase 3 implementation (weeks 1–2) with Phase 4 infrastructure in parallel.

