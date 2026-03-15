# Sensory Modality Framework

**Status**
- Label: **Conceptual**
- Relation to current repo: aligns with typed modality routing, tower-first structure, and domain-expansion planning
- Evidence status: partially adjacent to the synthetic routing ladder plus completed real-audio and real-sensor sequences
- Patent relevance: medium

## Concept

Sensory Perceiver should be read as a general sensory-integration architecture rather than as a framework tied only to one benchmark split such as `smartwatch`, `room_env`, and `robot_arm`.

The underlying idea is:

- any modality family has its own semantics,
- those semantics should be preserved before fusion,
- and the architecture should scale across modality families without pretending they are interchangeable.

## Why It Matters

This framing improves two things at once:

1. it clarifies the scientific scope of the architecture
2. it makes domain expansion more principled

Instead of asking only “what sensors do we already have?”, the better question is:

- what kind of sensory structure does this domain contain,
- and which distinctions deserve their own towers?

## Modality Families

The most useful current taxonomy is by structure, not by marketing label.

### Visual

- examples: images, video, optical flow, thermal imagery
- structure: high-dimensional spatial arrays, often with temporal continuity
- common bias: spatial and spatial-temporal processing

### Auditory

- examples: waveforms, spectrograms, MFCC-like features, ambient audio
- structure: temporal plus frequency-domain organization
- common bias: time-frequency processing

### Proprioceptive / Kinematic

- examples: joint angles, velocities, inertial state, robot arm telemetry
- structure: temporally ordered state evolution with mechanical constraints
- common bias: temporal and structured-state processing

### Interoceptive / Physiological

- examples: heart rate, ECG, PPG, respiration, temperature, fatigue signals
- structure: often low-dimensional but semantically rich and multi-timescale
- common bias: temporal processing with attention to calibration and subject drift

### Ambient / Environmental

- examples: room temperature, humidity, CO2, light, pressure, building state
- structure: sparse contextual state, slower temporal dynamics, nuisance sensitivity
- common bias: lightweight contextual encoding rather than heavy spatial processing

### Tactile / Contact

- examples: pressure arrays, force sensors, gripper contact, vibration
- structure: spatially distributed or event-like contact signals with fast local transitions
- common bias: local spatiotemporal or structured event processing

### Magnetic / Orientation

- examples: magnetometer, compass field, WLAN / localization-side ambient vectors
- structure: low-dimensional, contextual, often complementary rather than dominant
- common bias: lightweight auxiliary encoding

### Chemical

- examples: olfactory sensors, gas sensors, taste or compound-profile sensors
- structure: low-dimensional or moderate-dimensional composition vectors with slower dynamics and strong context dependence
- common bias: contextual encoding with calibration and drift awareness

## Overlap Patterns

Not all good multimodal systems are built from completely independent senses. Some of the most useful domain splits come from complementary overlap:

- visual + auditory
- tactile + proprioceptive
- physiological + ambient context
- magnetic/orientation + proprioceptive state
- chemical + environmental context

The point is not to eliminate overlap. The point is to separate streams that carry different kinds of evidence before fusion, then let the shared core learn how those overlaps matter.

This is the key refinement:

- the right question is not only “are these the same modality?”
- the right question is “do these inputs carry non-overlapping information?”

That is why:

- left and right ears can justify separation for localization
- left and right eyes can justify separation for depth
- several nearby accelerometers may benefit from merging if they mostly average the same body motion signal

## Practical Split Heuristics

When choosing whether two signals should share a tower or be separated, the best current heuristics are:

1. keep them separate if their temporal scales differ materially
2. keep them separate if their nuisance regimes differ materially
3. keep them separate if they arise from different physical mechanisms
4. consider merging only when they are strongly redundant and structurally similar

Examples:

- camera and microphone should not share a tower
- ECG and room temperature should not share a tower
- accelerometer and gyroscope may be grouped only if the task and noise model justify it
- two nearly identical scalar environmental channels may not deserve full tower separation

## Architectural Reading

This taxonomy does not mean every modality family already has a different implemented tower class in the repo.

Current truth:

- the repo uses one `PerceiverTower` family with modality-specific configuration
- the scientific claim is semantic distinction first
- richer per-family tower specialization remains future-facing

So the framework should be used as:

- a concept map for future design,
- a way to choose domains,
- and a way to reason about when typed routing should help

not as a claim that the current code already implements every modality-specific bias listed here.

## Relation to Distinction

This framework is one way to operationalize the distinction hypothesis.

If typed routing helps when streams carry genuinely different structure, then a good modality split should often track families like:

- visual vs auditory
- proprioceptive vs ambient
- physiological vs contextual

If a domain is split into streams that are mostly redundant or weakly differentiated, typed routing may add cost without benefit.

That is exactly why this framework is useful for the real-sensor follow-up analysis and for future domain expansion.

## Domain-Expansion Use

The most promising future domains are the ones where modality families are clearly heterogeneous:

- audio + video
- robot state + environment sensing
- physiological waveforms + slower vital/context streams
- tactile/contact + proprioception

These are stronger candidates than domains where the split is mostly arbitrary or redundant.

## Best Use In This Repo

Use this framework to:

- justify tower splits in future domain work
- rank future domains by heterogeneity and semantic separation
- interpret why typed routing helps on some datasets and not others

Do not use it to:

- imply unimplemented specialized towers already exist
- claim every sensory taxonomy branch is evidenced in the current paper

## Source Artifacts

- [/Users/zachshallbetter/Downloads/SENSORY_MODALITY_FRAMEWORK.md](/Users/zachshallbetter/Downloads/SENSORY_MODALITY_FRAMEWORK.md)
- [/Users/zachshallbetter/Projects/Perception/reference/sensor-types/Sensor References.md](/Users/zachshallbetter/Projects/Perception/reference/sensor-types/Sensor%20References.md)
- [/Users/zachshallbetter/Projects/Perception/reference/future-concepts/towers-and-modality-semantics.md](/Users/zachshallbetter/Projects/Perception/reference/future-concepts/towers-and-modality-semantics.md)
- [/Users/zachshallbetter/Projects/Perception/reference/future-concepts/distinction-and-separation.md](/Users/zachshallbetter/Projects/Perception/reference/future-concepts/distinction-and-separation.md)
