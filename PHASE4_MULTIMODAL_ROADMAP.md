# Phase 4: Multimodal Real-World Benchmarking Roadmap

**Objective**: Validate Sensory Perceiver architecture on real multimodal datasets spanning vision, audio, sensor fusion, and temporal streams.

**Current State**: Phase 2 validates on Split-CIFAR-100 (vision only, 19.54% CIL). Phase 3 extends with parameter isolation (expected 30–35% CIL, vision only).

**Gap**: No validation of MIME-Type encoding, cross-modal attention, or multimodal fusion on real data.

**Timeline**: 4–6 weeks (parallel track to Phase 3)

---

## Why Phase 4 Now?

The abstract claims the architecture is designed for "8+ sensory modalities" and positions MIME-Type encoding as the primary contribution. However:

- ✅ Phase 2 validates continual learning mechanism (raw-pixel replay)
- ✅ Phase 3 validates parameter isolation
- ❌ **Phase 2–3 lack multimodal validation** (Split-CIFAR-100 is vision-only)
- ❌ **No AudioSet, Ego4D, AV-MNIST results**
- ❌ **No cross-modal attention analysis**

Before publication, the paper must demonstrate that MIME-Type encoding and cross-modal attention work on real multimodal data. Phase 4 fills this gap.

---

## Benchmark Selection

### Tier 1: Must-Have (for publication)

#### 1A. AudioSet: Large-Scale Audio Classification

**Dataset**: 2M audio clips (10-second segments), 527 audio classes (musical instruments, animal sounds, vehicles, etc.)

**Modalities**: Mono/stereo audio at 16kHz or 128kbps compressed

**Continual learning setup**:
- 10-task sequential: 52–53 classes per task
- Task boundaries by semantic similarity (e.g., T0: animal sounds, T1: music, T2: vehicles)
- Benchmark: Split-AudioSet-10 (define formally)

**PerceiverTower for audio**:
```python
class AudioPerceiverTower(nn.Module):
    """Converts audio waveforms to embeddings."""
    def __init__(self, sample_rate=16000, duration=10, d_model=768):
        super().__init__()
        # Mel-spectrogram: (1, 128, 625) for 10-second 16kHz audio
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            n_mels=128
        )
        # Conv2d feature extraction (mirroring visual tower)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.mlp = nn.Linear(64, d_model)

    def forward(self, waveform):
        mel = self.mel(waveform)  # (B, 1, 128, T)
        features = self.conv(mel)  # (B, 64, 1, 1)
        features = features.flatten(1)  # (B, 64)
        return self.mlp(features)  # (B, 768)
```

**Expected outcome**:
- Baseline (no multimodal fusion): ~25% CIL on audio-only
- Sensory Perceiver on audio: 25–30% CIL (validates single-modality operation)

#### 1B. Ego4D: First-Person Multimodal Video

**Dataset**: 1,000+ hours of egocentric video with synchronized audio, hand annotations, and environment labels.

**Modalities**:
- Video (RGB frames at 30fps)
- Audio (synchronized mono audio)
- IMU/accelerometer (hand motion, device motion)
- Text annotations (scene descriptions)

**Continual learning setup**:
- 5-task sequential: 100 action classes split into 5 groups (e.g., T0: eating, T1: cooking, T2: sports, T3: manual work, T4: miscellaneous)
- Multimodal continual learning: video + audio + IMU (text optional)
- Benchmark: Ego4D-CL-5 (define formally)

**PerceiverTowers for Ego4D**:
```python
# Video tower (similar to CIFAR visual tower)
class VideoPerceiverTower(nn.Module):
    def __init__(self, frame_size=224, d_model=768):
        # Frame-level CNN + temporal pooling
        self.backbone = torchvision.models.resnet18(pretrained=True)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)  # Aggregate frames
        self.mlp = nn.Linear(512, d_model)

# Audio tower (Mel-spectrogram Conv2d like AudioSet)
class AudioPerceiverTower(nn.Module):
    # (same as above)

# IMU tower (MLP on accelerometer readings)
class IMUPerceiverTower(nn.Module):
    def __init__(self, d_model=768):
        self.mlp = nn.Sequential(
            nn.Linear(6, 128),  # 3-axis accel + 3-axis gyro
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, d_model)
        )
```

**Expected outcome**:
- Video-only baseline: 35–40% CIL
- Video + Audio: 42–45% CIL (audio provides complementary information)
- Video + Audio + IMU: 45–50% CIL (best multimodal fusion)
- **Validates that MIME-Type routing + cross-modal attention improves multimodal learning**

#### 1C. MSCOCO: Vision-Language Multimodal

**Dataset**: 330K images with 5 captions each (~1.5M image-caption pairs)

**Modalities**:
- Image (RGB)
- Text (captions, tokenized)

**Continual learning setup**:
- 5-task sequential: 80 object categories split into 5 groups (e.g., T0: animals, T1: vehicles, T2: furniture, T3: food, T4: outdoor)
- Task: Image classification with caption-based task descriptions
- Benchmark: MSCOCO-CL-5 (define formally)

**PerceiverTowers for MSCOCO**:
```python
# Visual tower (ResNet18)
# Text tower (BERT embeddings pooled)
class TextPerceiverTower(nn.Module):
    def __init__(self, d_model=768):
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.mlp = nn.Linear(768, d_model)

    def forward(self, input_ids, attention_mask):
        embeddings = self.bert(input_ids, attention_mask)
        pooled = embeddings.pooler_output
        return self.mlp(pooled)
```

**Expected outcome**:
- Vision-only baseline: 40–45% CIL
- Vision + Text: 45–50% CIL
- **Validates text modality support + vision-language integration**

### Tier 2: Nice-to-Have (if time permits)

#### 2A. AV-MNIST: Audio-Visual Digit Recognition

Synthetic dataset: 10-digit classes, 10 audio speakers → 100 combinations

- Digit 0: natural 0 spoken
- Digit 0: synthetic 0 spoken
- (etc. for all 10 digits × 10 speakers)

**Modalities**: Audio (spoken digit), Image (MNIST handwritten digit)

**Continual learning setup**:
- 10-task sequential: 1 digit per task (T0: 0, T1: 1, ..., T9: 9)
- Audio-visual pairing: test generalization to mismatch

**Expected outcome**: 70–80% CIL (relatively simple benchmark, validates audio-visual sync)

#### 2B. Smart Home Sensor Fusion (Private Dataset)

**Modalities**: Motion sensors (PIR), Temperature/Humidity, Sound (audio classification), RGB video

**Continual learning setup**:
- 5-task sequential: 5 home activities (T0: cooking, T1: eating, T2: sleeping, T3: working, T4: exercising)
- Multimodal: all 4 modalities present and relevant

**Expected outcome**: Demonstrates real-world sensor fusion application (robotics, IoT context)

---

## Implementation Plan

### Phase 4a: Infrastructure (Week 1)

- [ ] **Task 1.1**: Download/prepare datasets
  - AudioSet: 200GB+ (may need subset; coordinate with infrastructure)
  - Ego4D: Request access (requires registration)
  - MSCOCO: 13GB (quick download)

- [ ] **Task 1.2**: Build data loaders
  - `DataLoader` for each dataset
  - Task-sequential sampling logic
  - Train/test split with task boundaries

- [ ] **Task 1.3**: Implement modality-specific towers
  - AudioPerceiverTower (Mel-spectrogram Conv2d)
  - VideoPerceiverTower (ResNet18 or ViT backbone)
  - TextPerceiverTower (BERT or RoBERTa)
  - IMUPerceiverTower (MLP on time-series)

- [ ] **Task 1.4**: MIME-Type routing
  - Extend MIME-Type parser to map to new towers
  - Test routing logic

### Phase 4b: Benchmarking (Weeks 2–3)

- [ ] **Task 2.1**: Single-modality baselines
  - AudioSet audio-only: expected ~25% CIL
  - Ego4D video-only: expected ~35–40% CIL
  - MSCOCO vision-only: expected ~40–45% CIL
  - Compare to published baselines for each dataset

- [ ] **Task 2.2**: Multimodal results
  - Ego4D video+audio: expected +5–10% over video-only
  - Ego4D video+audio+IMU: expected +10–15% over video-only
  - MSCOCO vision+text: expected +5–10% over vision-only
  - Report cross-modal attention weights (Figure 10)

- [ ] **Task 2.3**: Cross-modal attention analysis
  - Generate heatmaps: which modalities attend to which?
  - Ego4D: does video attend to audio during speech-heavy tasks?
  - MSCOCO: does text attend to image during caption-dense classes?

- [ ] **Task 2.4**: Comparison to multimodal baselines
  - Early fusion (concatenation): baseline
  - Late fusion (separate towers, concat outputs): baseline
  - Perceiver IO: multimodal baseline (if available)
  - Sensory Perceiver: evaluate relative performance

### Phase 4c: Analysis & Paper (Week 4)

- [ ] **Task 3.1**: Write Results §6.4 (Multimodal Benchmarking)
  - Table: AudioSet, Ego4D, MSCOCO results
  - Figures: attention heatmaps, modality contribution breakdown

- [ ] **Task 3.2**: Update Discussion §7.5
  - Interpret cross-modal attention findings
  - Discuss when multimodal fusion helps vs. hurts
  - Relate to biological multisensory integration

- [ ] **Task 3.3**: Update Abstract
  - Provide concrete numbers from Phase 4 (replace hypothetical claims)
  - "Validated on AudioSet, Ego4D, MSCOCO with 45–50% CIL"

- [ ] **Task 3.4**: Figures
  - Figure 10: Cross-modal attention heatmaps (Ego4D)
  - Figure 11: Modality contribution (ablation: video vs. audio vs. both)
  - Figure 12: Phase 2 (CIFAR-100), Phase 4 (Ego4D) comparison

---

## Expected Results

### Tier 1 Benchmarks

| Benchmark | Modalities | Single-Modality | Multimodal | Improvement |
|-----------|-----------|-----------------|-----------|------------|
| AudioSet | Audio | 25% | — | — |
| Ego4D | Video | 35% | Video+Audio: 40%, Video+Audio+IMU: 45% | +10% |
| MSCOCO | Vision | 42% | Vision+Text: 47% | +5% |

### Success Criteria

**Minimum**: Single-modality performance ≥ published baselines
**Target**: Multimodal fusion improves by ≥ 5% vs. single-modality
**Stretch**: Achieves SOTA on multimodal continual learning (unlikely; would require novel comparison)

---

## Fallback Plans

### If AudioSet unavailable:
Use **FSD50K** (Freesound Dataset) instead: 50K audio clips, 50 sound classes. Smaller but sufficient.

### If Ego4D access denied:
Use **UCF101** (101 action recognition classes, ~13k videos). Single-modality; run video-only test.

### If infrastructure bottleneck:
Implement **incremental evaluation**: Run Phase 4 in stages (1 benchmark per week).

---

## Integration with Paper

Once Phase 4 completes:

1. **Title**: Consider updating to emphasize multimodal validation
   - Old: "Perception Is How We Experience"
   - New: "Sensory Perceiver: Multimodal Continual Learning via Experience Replay"

2. **Abstract**: Include concrete multimodal numbers
   - "Validated on AudioSet (25% CIL), Ego4D (45% CIL video+audio+IMU), and MSCOCO (47% CIL vision+text)"

3. **Introduction**: Reference real multimodal validation (not hypothetical)
   - Shift from "designed for 8+ modalities" to "demonstrated on 3–4 modalities"

4. **Related Work**: Add real multimodal continual learning papers

5. **Results**: New major section (§6.4 Multimodal Benchmarking)
   - Tables with cross-dataset results
   - Heatmaps of cross-modal attention weights

6. **Discussion**: Expand §7.5 (Generalization Across Domains)
   - AudioSet shows audio-specific learning
   - Ego4D shows multimodal fusion benefits
   - MSCOCO shows vision-language integration

---

## Publication Readiness Post-Phase 4

### Conference Ready (Top-tier: ICML, NeurIPS, ICCV)

Requires:
- ✅ Vision-only SOTA or near-SOTA (Phase 3: 30–35% CIL expected)
- ✅ Multimodal validation (Phase 4: Ego4D 45% CIL expected)
- ✅ Reproducibility (code, hyperparameters, seeds)
- ⚠️ Novel contribution (raw-pixel replay + multimodal = novel)

**Verdict**: Publishable at top conference if Phase 3 + Phase 4 meet targets

### Journal Ready (Nature Machine Intelligence, JMLR)

Additional requirements:
- [ ] Extended baselines (EWC, GEM, PackNet, etc.)
- [ ] Theoretical analysis of tower-gradient insight
- [ ] Ablation studies (Table: which component contributes how much?)
- [ ] Long-term continual learning (50+ tasks)

**Timeline**: 6+ months after Phase 4 completion

---

## Risk Assessment

### High Risk

**AudioSet/Ego4D access or size**: 200GB+ datasets may exceed local storage/bandwidth.
- **Mitigation**: Use smaller subsets; leverage cloud infrastructure

**Cross-modal attention interpretability**: Heatmaps may not be informative if architecture is not interpretable.
- **Mitigation**: Supplement with ablation (freeze cross-attention, observe degradation)

### Medium Risk

**Multimodal fusion not helping**: Concatenation baseline may match Sensory Perceiver, undermining the multimodal motivation.
- **Mitigation**: Pre-test on Ego4D early (week 1); pivot to logit replay if needed

**Modality alignment**: Audio and video may be misaligned (e.g., Ego4D audio lags video). Requires temporal alignment preprocessing.
- **Mitigation**: Use dataset-provided synchronization; add alignment loss if needed

### Low Risk

**MSCOCO and AV-MNIST**: Well-established benchmarks, known to work.

---

## Timeline & Milestones

| Week | Phase | Milestone | Deliverable |
|------|-------|-----------|------------|
| 1 | 4a | Infrastructure | Data loaders + modality towers ready |
| 2 | 4b | Single-modality | AudioSet/Ego4D/MSCOCO baselines complete |
| 3 | 4b | Multimodal | Fusion results + cross-modal attention heatmaps |
| 4 | 4c | Paper integration | Results §6.4 written, figures generated |

**Total**: 4 weeks (can overlap with Phase 3 weeks 2–3)

---

## Success Declaration

Phase 4 is successful when:

1. ✅ Single-modality results ≥ published baselines
2. ✅ Multimodal fusion improves performance by ≥ 5%
3. ✅ Cross-modal attention is visualizable and interpretable
4. ✅ Paper integrates findings into Sections 4–7
5. ✅ All code is reproducible and documented

**Post-Phase 4**: Paper is ready for submission to top-tier venue.

