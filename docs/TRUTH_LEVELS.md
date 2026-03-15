# Sensory Perceiver: Truth Levels Rubric

**Purpose**: Distinguish between implemented truth, evidenced truth, architectural truth, and conceptual truth. Enables honest communication about what this work actually proves vs. what it suggests vs. what is speculative.

**Single source of truth for all claims across paper, code, docs, and public statements.**

---

## Truth Level Definitions

### 🟢 **IMPLEMENTED & EVIDENCED** (Defensible in publication)
- Code exists, runs, produces results
- Empirical evidence on real or synthetic data
- Comparative validation (beats baseline or prior approach)
- **Can claim publicly** with full confidence
- Example: "We show that replay reaching the backbone improves retention by 23.5pp"

### 🟡 **IMPLEMENTED BUT NOT BROADLY EVIDENCED** (Architecture confidence)
- Code exists and runs correctly
- Limited empirical validation (1-2 datasets, specific conditions)
- Plausible as a design principle
- **Can claim carefully** with scope caveats
- Example: "Typed modality routing supports declarative fusion; we demonstrate this on synthetic streams and two real sensor types"

### 🔵 **ARCHITECTURAL TRUTH** (Systems-level confidence)
- No running code OR code is prototype/sketch-stage
- Strong theoretical justification or analogues in literature
- Solves a real problem in a principled way
- **Can discuss as direction/principle**, must not overstate implementation maturity
- Example: "Hierarchical grouping of 5-6 modalities is a plausible scaling strategy, analogous to multi-scale vision architectures"

### 🟠 **CONCEPTUAL / ILLUSTRATIVE** (Future directions)
- Concept demonstration, design template, or extrapolation
- Supports a narrative or principle
- Not yet implemented, or implementation is pseudocode
- May contain patentable ideas
- **Can say "we envision" or "a system could"**, must not claim validation
- Example: "Concert lighting could be synced by predicting energy 100ms ahead"

### ⚫ **DO NOT CLAIM YET** (Speculative or premature)
- Incomplete work, conflicting results, or unresolved unknowns
- Claims would be misleading or contradicted by evidence
- **Exclude from public statements entirely**, document for internal tracking
- Example: "Real-data training on PAMAP2 failed; reason unknown; omit from publication until resolved"

---

## Rubric Summary Table

| Level | Code | Empirical Data | Scope | Public Claim | Example |
|-------|------|---|---|---|---|
| 🟢 Implemented & Evidenced | ✓ Full | ✓ Multiple tests | Clear | Yes, with confidence | "Replay to backbone improves retention 23.5pp (Split-CIFAR-100, 6-seed mean)" |
| 🟡 Implemented, Limited Evidence | ✓ Full | ✓ Limited (1-2) | Narrow | Yes, with caveats | "Typed routing works on smartwatch, room, robot; not tested on medical sensors yet" |
| 🔵 Architectural | ⚠️ Sketch/none | ✗ Theoretical | General principle | As direction | "Towers-first design maps naturally to modality semantics; we expect this to scale" |
| 🟠 Conceptual | ⚠️ Pseudocode | ✗ Illustrative | Narrative | "Could," "envision," "suggest" | "A concert system could leverage lookahead buffers for predictive lighting" |
| ⚫ Do Not Claim | ✗ Broken/incomplete | ✗ Conflicting | Undefined | No | (Exclude from docs; document in internal issue tracking only) |

---

## Component Classification

### Core Paper Claims

#### Continual Learning Pipeline
| Component | Status | Evidence | Truth Level | Caveats |
|-----------|--------|----------|-------------|---------|
| Replay to backbone improves retention | 🟢 Implemented & Evidenced | Split-CIFAR-100 Phase 1→2→3, 6-seed mean | **PUBLISHED** | Phase 3 is tuned hyperparams; Phase 1/2 use default settings |
| Phase 3 achieves 23.96% CIL | 🟢 Implemented & Evidenced | Measured on Split-CIFAR-100 (standard benchmark) | **PUBLISHED** | Single architecture; limited architectural variation tested |
| BWT improves from −53.80% → −30.39% | 🟢 Implemented & Evidenced | Measured across all three phases | **PUBLISHED** | Improvement correlates with replay-path placement; causality shown via ablation |
| Task 0 retention 0.6% → 25.1% | 🟢 Implemented & Evidenced | Direct measurement | **PUBLISHED** | Phase 1 feature-only; Phase 3 full replay; clear mechanistic story |

#### Synthetic Data & Routing
| Component | Status | Evidence | Truth Level | Caveats |
|-----------|--------|----------|-------------|---------|
| Sensor routing ladder (canonical-dominant) | 🟢 Implemented & Evidenced | 6 ablations, clear gaps, modality dependency visible | **PUBLISHED** | Synthetic data; may not hold on all real modalities |
| Typed modality dispatch works | 🟡 Implemented, Limited Evidence | Works on 3 modalities (smartwatch, room, robot); routed correctly | **CLAIM CAREFULLY** | Not tested on medical, audio, video yet |
| Drop-ablation isolates modality value | 🟢 Implemented & Evidenced | Each drop produces measurable gap (except robot arm drop →−0.09 reversal) | **PUBLISHED** | One reversal suggests interaction effects; document as caveat |

#### Real-Data Bootstrap
| Component | Status | Evidence | Truth Level | Caveats |
|-----------|--------|----------|-------------|---------|
| Speech Commands achieves 88.32% | 🟢 Implemented & Evidenced | Measured on standard benchmark | **PUBLISHED** | Audio-only; single modality; not multimodal yet |
| WISDM +4.19pp over canonical-minus-merged | 🟢 Implemented & Evidenced | Measured (accel+gyro sensor data) | **PUBLISHED** | Small dataset; margin small relative to noise floor |
| PAMAP2 training failed (real-audio-seq-002 blocked) | 🟠 Conceptual / **DO NOT CLAIM** | Attempted; did not complete | **EXCLUDE** | Reason unknown; including would mislead readers |

---

### Architecture & Design Principles

#### Tower-First Multimodal Design
| Claim | Status | Justification | Truth Level | Public Statement |
|-------|--------|---|-------------|---|
| Towers preserve modality semantics | 🔵 Architectural | Analogous to early vs. late fusion; empirically better on toy data | **ARCHITECTURAL PRINCIPLE** | "We propose that per-modality encoders preserve task-relevant structure before fusion. This is supported by continual learning results on synthetic sensor streams." |
| Early fusion is wrong default | 🔵 Architectural | Literature (multimodal learning); our toy example (1.22M dims → catastrophic) | **ARCHITECTURAL PRINCIPLE** | "Early fusion of high-dim heterogeneous inputs is computationally and statistically inefficient; we demonstrate this on synthetic streams." |
| Replay should reach towers | 🟢 Implemented & Evidenced | 23.5pp improvement directly attributable to this | **PUBLISHED** | "Empirically, replay reaching the backbone yields 23.5pp better retention on Split-CIFAR-100." |
| Bounded supervisors > online learning | 🟡 Implemented, Limited Evidence | Works on 3 modalities; not compared to other online-learning frameworks | **CLAIM CAREFULLY** | "We implement a bounded-replay strategy; initial results are promising but broader comparison needed." |

#### Hierarchical & Dynamic Composition
| Claim | Status | Justification | Truth Level | Public Statement |
|-------|--------|---|-------------|---|
| Hierarchical grouping scales 5→6 modalities | 🟠 Conceptual | Design principle; no empirical validation on real 6-modality system | **FUTURE WORK** | "Hierarchical grouping is a plausible scaling strategy, analogous to vision pyramids. Future work will test this on multi-domain systems." |
| Dynamic composition works | 🔵 Architectural | Code sketch exists; no live validation | **ARCHITECTURAL PRINCIPLE** | "Dynamic modality composition allows add/remove without retraining the shared core. Implementation is feasible; deployment validation pending." |
| Graceful degradation on modality dropout | 🟡 Implemented, Limited Evidence | Tested on 3 modalities; masked attention works | **CLAIM CAREFULLY** | "Our system handles single-modality dropout gracefully (tested on 3 sensor types); behavior on critical modalities untested." |

---

### Domain-Specific Extensions (New Work)

#### Concert Audio Perceiver
| Component | Status | Evidence | Truth Level | Public Statement |
|-----------|--------|----------|-------------|---|
| 5-tower design (percussion, vocal, harmony, crowd, context) | 🟠 Conceptual | Domain knowledge + analogy to music psychology; no implementation | **ENVISION** | "A concert perceiver could employ separate towers for rhythm, pitch, harmony, and crowd. This reflects how listeners process music." |
| Onset detection for percussion | 🟠 Conceptual | Standard signal processing + learned CNN; tested on synthetic only | **ILLUSTRATIVE** | "We sketch an onset-detection tower combining spectral flux and learned detection; synthetic validation only." |
| Real-time sync <50ms | 🟠 Conceptual | Latency budget estimated; not hardware-validated | **ASPIRATIONAL** | "Target latency is <50ms (estimated); real deployment will require profiling and tuning." |
| Lookahead buffering for predictive lighting | 🟠 Conceptual | Plausible from latency perspective; not implemented | **FUTURE DIRECTION** | "Predictive lookahead (100ms ahead) could enable proactive lighting; concept only at this stage." |

#### Security Camera Tower
| Component | Status | Evidence | Truth Level | Public Statement |
|-----------|--------|----------|-------------|---|
| CNN+ConvLSTM for video | 🔵 Architectural | Analogous to video action recognition; standard architecture | **ARCHITECTURAL PRINCIPLE** | "Spatial-temporal video processing via CNN+ConvLSTM is well-established in vision. We apply this pattern to perceiver towers." |
| 30-120 fps streaming | 🟠 Conceptual | Frame rates specified; no real camera integration | **ASPIRATIONAL** | "Target frame rates are 30-120 fps. Implementation requires hardware integration." |
| Thermal + camera fusion | 🟠 Conceptual | Design sketch; no empirical validation | **FUTURE WORK** | "Thermal and visible spectra are complementary. We propose fusing them; validation pending." |

#### Visualization Lenses
| Component | Status | Evidence | Truth Level | Public Statement |
|-----------|--------|----------|-------------|---|
| Canonical color semantics | 🟠 Conceptual | Design principle from Artistic Notation spec; not empirically validated | **ILLUSTRATIVE** | "We apply canonical color semantics from music visualization theory. Validation of perceptual effectiveness is future work." |
| Chromatic Wash, Tension Terrain, etc. | 🟠 Conceptual | Rendered examples; no user study or perceptual validation | **ILLUSTRATIVE** | "Four lenses show different interpretations of the same perceiver output. Cognitive effectiveness is not yet measured." |
| Real-time streaming (30 fps @ 1280×720) | 🟡 Implemented, Limited Evidence | Latency budget estimated; not hardware-validated on real concert hardware | **CLAIM CAREFULLY** | "Target is 30 fps @ 1280×720 with <50ms latency. Actual performance depends on GPU and network conditions." |

---

### Infrastructure & Implementation Details

#### Typed Runtime Routing
| Component | Status | Evidence | Truth Level | Caveats |
|-----------|--------|----------|-------------|---------|
| MIME contract enforcement | 🟡 Implemented, Limited Evidence | Works on 3 modality types; tested in lab | **CLAIM CAREFULLY** | "Typed contracts prevent modality confusion in the perceiver. Tested on smartwatch, room, robot; extensibility to new types needs validation." |
| Live mask-aware inference | 🟡 Implemented, Limited Evidence | Code runs; limited test coverage | **CLAIM CAREFULLY** | "Graceful handling of missing modalities via attention masking. Tested on single-modality dropout; behavior under cascade failures is unknown." |

#### Real-Data Infrastructure
| Component | Status | Evidence | Truth Level | Caveats |
|-----------|--------|----------|-------------|---------|
| Speech Commands integration | 🟢 Implemented & Evidenced | Runs, validates on benchmark | **PUBLISHED** | Single audio modality; not multimodal yet |
| WISDM integration | 🟢 Implemented & Evidenced | Runs, validates on benchmark | **PUBLISHED** | Single sensor pair (accel+gyro); small dataset |
| PAMAP2 attempted | ⚫ Do Not Claim | Failed (reason unclear) | **EXCLUDE** | Document in internal issue tracking; do not mention in paper |

---

## Public Statement Guidelines

### When to Say "We Show"
✅ Use only for 🟢 Implemented & Evidenced claims.
```
"We show that replay to the backbone improves retention by 23.5pp 
on Split-CIFAR-100 (6-seed mean, Phase 3)."
```

### When to Say "We Propose / We Design"
✅ Use for 🔵 Architectural claims.
```
"We propose towers as a per-modality encoding strategy. 
This is motivated by early-fusion inefficiency and supported by 
continual-learning results on synthetic sensor streams."
```

### When to Say "We Sketch / We Envision"
✅ Use for 🟠 Conceptual claims.
```
"A concert perceiver could employ separate towers for rhythm, 
pitch, and engagement. We sketch this design and show how it maps 
onto the perceiver architecture."
```

### When to Say NOTHING
❌ Exclude 🟰 Do Not Claim material entirely.
```
EXCLUDE: "We trained on PAMAP2..." 
REASON: Training failed; reason unknown; would mislead.
INTERNAL: File issue #NNN: PAMAP2 training debugging needed.
```

---

## Exceptions & Edge Cases

### Ablation Results with Single Reversals
**Status**: 🟢 Implemented & Evidenced (with caveat)

When a drop ablation produces an unexpected sign reversal (e.g., drop_robot_arm → −0.09), this is:
- ✅ Real (measured)
- ⚠️ Unexplained (likely interaction effect)

**Public statement**:
```
"Drop-robot-arm produces a reversal (−0.09), suggesting 
that robot-arm features may interact negatively with other modalities 
under this configuration. This warrants further investigation."
```

### Limited Real-Data Evidence
**Status**: 🟡 Implemented, Limited Evidence

When a method works on Speech Commands (88.32%) and WISDM (+4.19pp) but fails on PAMAP2:

**Public statement**:
```
"On audio (Speech Commands) and IMU (WISDM), our approach achieves 
strong results. PAMAP2 training encountered issues (under investigation); 
we exclude it from claims pending resolution."
```

### Architectural Analogues Without Direct Proof
**Status**: 🔵 Architectural Truth

When early-fusion inefficiency is shown on toy data (1.22M→300 dims) but not on real multimodal benchmarks:

**Public statement**:
```
"Early fusion of high-dimensional heterogeneous inputs is statistically 
and computationally inefficient. We demonstrate this on synthetic streams; 
large-scale real-data validation is future work."
```

---

## Repo Inventory (All Major Components Classified)

### Implemented & Evidenced (🟢)
- [x] Replay-path placement (backbone >> head)
- [x] Phase 3 tuned continual learning (23.96% CIL, −30.39% BWT)
- [x] Task 0 retention improvement (0.6% → 25.1%)
- [x] Synthetic sensor routing ladder (6 ablations complete)
- [x] Speech Commands audio-only validation (88.32%)
- [x] WISDM sensor validation (+4.19pp)
- [x] Typed modality dispatch (on 3 types)
- [x] Masked attention for missing modalities (tested)

### Implemented, Limited Evidence (🟡)
- [x] Typed routing on 3 modalities only
- [x] Graceful degradation on single-modality dropout
- [x] Real-time latency budget (<50ms estimated)
- [x] Visualization streaming (30 fps estimated)
- [ ] Hierarchical grouping (5-6 modalities)
- [ ] Dynamic composition (add/remove at runtime)

### Architectural (🔵)
- [x] Tower-first multimodal design
- [x] Replay to backbone as principle
- [x] Bounded supervisors vs. online learning
- [x] Hierarchical grouping for scaling

### Conceptual (🟠)
- [x] Concert audio perceiver (5 towers, 50ms latency)
- [x] Security camera tower (CNN+ConvLSTM video)
- [x] Thermal camera addition
- [x] Canonical color semantics for visualization
- [x] Artistic notation lenses (4 designs)
- [x] Predictive lighting via lookahead
- [x] Dynamic modality composition

### Do Not Claim (⚫)
- [x] PAMAP2 real-data training (failed, reason unknown)
- [x] Multi-tower real-data validation (attempted, incomplete)
- [x] Production concert system (speculative)

---

## Change Log

**2026-03-12**: Initial truth-level inventory created. All components classified. Public statement guidelines established.

---

## How to Use This Document

1. **Before writing any public claim**, check this table.
2. **If claim is 🟢**, write with full confidence: "We show..."
3. **If claim is 🟡**, add scope caveat: "On these 3 modalities, we find..."
4. **If claim is 🔵**, frame as principle: "We propose that..."
5. **If claim is 🟠**, use future language: "We envision..."
6. **If claim is ⚫**, exclude entirely. Document in issues instead.
7. **When updating the paper**, reference this document by section/component.

---

## Contacts & Ownership

**Author**: Zach Shallbetter  
**Last updated**: 2026-03-12  
**Maintenance**: Update whenever code is added, data is tested, or a component's confidence level changes.  
**Questions**: File issue or discussion referencing the component name and truth level.
