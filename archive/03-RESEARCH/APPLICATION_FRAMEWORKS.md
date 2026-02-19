# Application Domain Frameworks: Experimental Design

Complete experimental designs for 3 application papers (Robotics, Medical, Smart Home). Use as blueprint for Week 7-10 experiments.

---

## Application 1: Robotics & Autonomous Systems

### Problem Statement
Robotic systems require fusing 4+ modalities in real-time:
- **Visual** (camera): 640×480, 30 FPS
- **Proprioception** (joint angles): 6-DOF, 100 Hz
- **Force/Torque** (end-effector): 3-axis force + 3-axis torque, 100 Hz
- **Kinematics** (forward kinematics): Derived from joint angles

**Challenge**: Current approaches use hand-coded fusion logic (brittle, hard to generalize).

**Hypothesis**: SensoryPerceiver's MIME-Type routing handles this automatically, generalizing across robot morphologies.

---

### Experimental Design

#### Task 1: Pick-and-Place Reaching
**Environment**: PyBullet simulator (Franka Panda manipulator)
**Objective**: Reach target position with end-effector
**Modalities**:
- Vision: RGB image of scene (cropped to 64×64 for efficiency)
- Proprioception: 7 joint angles (normalized -180° to +180°)
- Force/Torque: 6-axis F/T sensor (normalized to [-1, 1])
- Target: XYZ position of goal (3 scalars)

**Inputs**:
```
Vision:       (batch, 3, 64, 64)    → PerceiverTower(Vision)
Proprioception: (batch, 7)          → PerceiverTower(IoT) [treated as 1D sequence]
Force/Torque:   (batch, 6)          → PerceiverTower(IoT)
Target:         (batch, 3)          → Scalar input

Output: (batch, 7) → Joint velocity commands (τ = dq/dt)
```

**Success Metric**: Reach target within 5cm AND maintain contact force < 10N

**Dataset**:
- Generate 1000 reaching trajectories (random targets)
- Split: 700 train, 300 test
- 3 robot morphologies: 6-DOF, 7-DOF, kinematically redundant

#### Task 2: Grasping
**Environment**: Pybullet with object physics
**Objective**: Grasp and lift object without dropping
**Modalities**: Same as above + contact sensor binary (true/false = grasped/not grasped)
**Output**: (batch, 8) → Joint velocity + gripper command
**Success Metric**: Grasp success rate AND object not dropped after 3 seconds

#### Task 3: Generalization to Unseen Morphology
**Test**: Train on 6-DOF robot, test on 7-DOF robot (unseen morphology)
**Metric**: Does SensoryPerceiver generalize better than hand-coded fusion?
**Expected**: Yes, because MIME-Type routing is morphology-agnostic

---

### Baselines

| Baseline | Method | Expected Accuracy |
|----------|--------|------------------|
| **Hand-Coded Fusion** | Linear combination: vision features + joint angles + F/T | 75% |
| **EKF Fusion** | Extended Kalman Filter for sensor fusion | 78% |
| **LSTM** | Sequence-to-sequence on concatenated modalities | 80% |
| **Perceiver IO** | Generic latent bottleneck | 82% |
| **SensoryPerceiver** | Modality-specific routing | 87% |

---

### Implementation Details

**Simulation Environment**:
```python
import pybullet as p
import pybullet_data

p.connect(p.GUI)
robot = p.loadURDF("franka_panda.urdf")
object = p.loadURDF("object.urdf", [0, 0, 1])

for episode in range(1000):
    # Generate random target
    target = [random(), random(), 1.0]

    # Run SensoryPerceiver for 1000 steps (10 seconds at 100 Hz)
    state = initial_state
    for t in range(1000):
        # Collect observations
        vision = render_camera()
        joints = p.getJointStates(robot, joint_indices)
        f_t = read_force_torque_sensor()

        # Forward through SensoryPerceiver
        action = model(vision, joints, f_t, target)

        # Execute action
        p.setJointMotorControlArray(robot, joints, p.VELOCITY_CONTROL, targetVelocities=action)

        # Check success
        ee_position = p.getLinkState(robot, ee_index)[4]
        success = distance(ee_position, target) < 0.05
```

**Metrics**:
- Success rate (% of episodes where goal reached)
- Time to success (how many steps needed)
- Force smoothness (variance of force over time, lower is better)
- Generalization (% accuracy on unseen morphology)

---

### Experiments (Week 7)

| Exp | Task | Robot | Runs | Duration | GPU |
|-----|------|-------|------|----------|-----|
| 1 | Reaching | 6-DOF | 10 | 2h | GPU 2 |
| 2 | Reaching | 7-DOF | 10 | 2h | GPU 2 |
| 3 | Grasping | 6-DOF | 10 | 3h | GPU 2 |
| 4 | Grasping | 7-DOF | 10 | 3h | GPU 2 |
| 5 | Morphology Generalization | 6→7 DOF transfer | 1 | 1h | GPU 2 |

**Total**: ~11 hours GPU time (parallelized across episodes)

**Result Expected**: 87% ± 2% success for SensoryPerceiver on both morphologies, beating hand-coded fusion significantly.

---

## Application 2: Medical Monitoring & Continual Learning

### Problem Statement
Hospital and home monitoring collect multimodal physiological data:
- **ECG** (electrocardiogram): 1D timeseries, 250 Hz
- **SpO₂** (oxygen saturation): Scalar continuous, 1 Hz
- **Accelerometer** (motion): 3-axis, 100 Hz
- **Temperature**: Scalar continuous, 0.1 Hz

**Challenge**: Patient-specific baselines drift over time (electrode shift, device recalibration). Models must adapt without retraining.

**Hypothesis**: SensoryPerceiver's continual learning prevents false alarms from baseline drift; replay buffer learns patient-specific patterns.

---

### Experimental Design

#### Task 1: Arrhythmia Detection
**Dataset**: PhysioNet MIMIC-III (ICU records)
**Objective**: Classify ECG as Normal, Atrial Fibrillation, Ventricular Tachycardia, or Bradycardia
**Modalities**:
```
ECG:           (batch, 250, 1)     → PerceiverTower(Timeseries)
SpO₂:          (batch, 1)          → PerceiverTower(Scalar)
Accelerometer: (batch, 100, 3)     → PerceiverTower(Timeseries)
Temperature:   (batch, 1)          → PerceiverTower(Scalar)

Output: (batch, 4) → Logits for 4 arrhythmia classes
```

**Success Metric**:
- Sensitivity (TP / TP+FN) > 95%
- Specificity (TN / TN+FP) > 90%
- F1 score

#### Task 2: Patient Adaptation (Continual Learning)
**Setup**: Simulate patient monitored for 30 days
- First 5 days: Train on labeled examples
- Days 6-30: New examples arrive (unlabeled), model must adapt via continual learning

**Baseline Drift**: Simulate electrode impedance increase (5% per day)
- Makes ECG waveforms gradually noisier
- Naive model accuracy drops 2% per day
- With replay: Minimal accuracy drop (0.1% per day)

**Metric**: How much does accuracy drop over 30 days?
- Without continual learning: -60% accuracy (breaks completely)
- With replay: -5% accuracy (robust adaptation)

#### Task 3: Generalization to New Patient
**Test**: Train on 50 patients, test on new (51st) patient
**Metric**: Accuracy without retraining
**Expected**: SensoryPerceiver generalizes better because modality-specific towers learn robust features

---

### Datasets & Preprocessing

**PhysioNet MIMIC-III**:
- 46,520 ICU patients with continuous monitoring
- Use subset: 1000 patients with clean ECG + SpO₂ + accelerometer data
- Ground truth: Manual annotation by cardiologists

**Preprocessing**:
```python
# ECG: Bandpass filter (0.5-40 Hz) to remove noise and baseline drift
ecg_clean = butter_bandpass_filter(ecg_raw, 0.5, 40, 250)

# SpO₂: Smooth with 30-second moving average (physiological dynamics)
spo2_smooth = moving_average(spo2_raw, window=30)

# Accelerometer: Compute magnitude and smooth
accel_mag = sqrt(ax^2 + ay^2 + az^2)
accel_smooth = moving_average(accel_mag, window=10)

# Temperature: Raw (changes slowly)

# Normalize all to [-1, 1]
ecg_norm = (ecg_clean - mean) / std
spo2_norm = (spo2_smooth - 95) / 5  # Typically 90-100
accel_norm = (accel_smooth - 0.2) / 0.3  # Typical range
temp_norm = (temperature - 37) / 2  # Typical range 35-39°C
```

---

### Experiments (Weeks 8-9)

| Exp | Task | Dataset | Runs | Duration |
|-----|------|---------|------|----------|
| 1 | Arrhythmia Detection | MIMIC-III | 10 | 4h |
| 2 | Baseline Drift (30-day) | Synthetic | 1 | 2h |
| 3 | Patient Generalization | 50→51 | 1 | 1h |
| 4 | Robustness to Sensor Dropout | MIMIC-III + dropout | 5 | 2h |

**Total**: ~9 hours GPU time

**Expected Results**:
- Arrhythmia: 96% sensitivity, 91% specificity (state-of-the-art clinical accuracy)
- Continual learning: 5% accuracy drop over 30 days (vs 60% without replay)
- Generalization: 92% accuracy on new patient (vs 75% for baselines)
- Dropout robustness: Minimal loss even with 30% sensor dropout

---

## Application 3: Smart Home IoT Orchestration

### Problem Statement
Smart homes collect heterogeneous sensor data:
- **Environmental**: Temperature, humidity, CO₂, light level (scalars, 0.5 Hz)
- **Occupancy**: Motion sensors, door contacts, occupant detected (binary, 1 Hz)
- **Appliances**: Power usage per device, on/off state (scalars, 1 Hz)
- **Temporal**: Time of day, day of week, seasonality (categorical, discrete)

**Challenge**: Systems use hard-coded rules ("if temp > 24°C AND occupancy, then AC on"). Brittle, doesn't adapt to user preferences.

**Hypothesis**: SensoryPerceiver learns user preferences automatically, enabling context-aware automation without manual rules.

---

### Experimental Design

#### Task 1: Occupancy Detection
**Objective**: Predict which rooms are occupied (binary classification)
**Modalities**:
```
Environmental: (batch, 4)           → PerceiverTower(Scalar)
Occupancy:     (batch, 5)           → PerceiverTower(Scalar)
Time:          (batch, 2)           → PerceiverTower(Scalar) [hour, weekday]

Output: (batch, 5, 2) → Logits for occupied/unoccupied per room
```

**Success Metric**:
- Accuracy > 95%
- Precision > 90% (don't waste energy with false positives)
- Recall > 95% (don't miss actual occupancy)

#### Task 2: Energy Optimization
**Objective**: Predict optimal HVAC setpoint given current state
**Inputs**: Same as above + historical comfort feedback (simulated)
**Output**: (batch, 1) → Optimal temperature (17-28°C)

**Metric**: Energy savings vs naive baseline
- Baseline: Fixed 22°C year-round
- SensoryPerceiver: Learned setpoint per context (occupancy, time, weather)
- Expected: 20-30% energy savings

#### Task 3: Anomaly Detection
**Objective**: Detect unusual patterns (e.g., water leak, intrusion)
**Setup**: Normal operation for 30 days, then inject anomaly
**Metric**: Time to detect (should be < 1 hour)

---

### Simulation Environment

**Smart Home Simulator** (custom Python):
```python
class SmartHome:
    def __init__(self, n_rooms=5):
        self.temp = [22] * n_rooms  # Celsius
        self.humidity = [40] * n_rooms
        self.occupancy = [False] * n_rooms
        self.time = 6 * 3600  # 6 AM
        self.weather_temp = 15  # Outdoor temp

    def step(self, hvac_actions):
        # Update environment
        for room in range(self.n_rooms):
            # Heat flow (simplified physics)
            dT = hvac_actions[room] - 0.1 * (self.temp[room] - self.weather_temp)
            self.temp[room] += dT * 0.01

        # Occupancy changes (time-based + random)
        if 7 <= self.hour < 9:  # Morning: Living room, kitchen
            self.occupancy[0] = True  # Living room
            self.occupancy[1] = True  # Kitchen

        self.time += 60  # Step = 1 minute
        return self.get_observation()

    def get_observation(self):
        return {
            'temp': self.temp,
            'humidity': self.humidity,
            'occupancy': self.occupancy,
            'time': self.time,
        }
```

---

### Experiments (Weeks 9-10)

| Exp | Task | Setting | Duration |
|-----|------|---------|----------|
| 1 | Occupancy Detection | Simulated 30 days | 2h |
| 2 | Energy Optimization | Simulated 90 days (seasonal) | 3h |
| 3 | Anomaly Detection | Normal + injected anomaly | 1h |
| 4 | Real Home Deployment | 2 weeks in actual smart home | 2h |

**Total**: ~8 hours GPU time

**Expected Results**:
- Occupancy: 96% accuracy (vs 85% for rule-based system)
- Energy: 25% savings (vs fixed-setpoint baseline)
- Anomaly detection: < 30 minutes to detect water leak
- Real home: User satisfaction > 8/10 (preference learning works)

---

## Paper Writing Integration

### Robotics Paper (Week 7-8)
Use experiments above as **Results** section:
- Table 1: Success rates by task and morphology
- Figure 1: Success rate curves over training episodes
- Figure 2: Generalization comparison (hand-coded vs SensoryPerceiver)
- Discussion: Why MIME-Type routing helps robotics

### Medical Monitoring Paper (Week 8-9)
Use experiments above as **Results** section:
- Table 1: Arrhythmia detection metrics
- Figure 1: Accuracy over 30-day patient trajectory
- Figure 2: Continual learning advantage (with/without replay)
- Figure 3: Sensor dropout robustness curve
- Discussion: Clinical implications and FDA considerations

### Smart Home Paper (Week 9-10)
Use experiments above as **Results** section:
- Table 1: Task-wise performance metrics
- Figure 1: Energy savings over 90 days (seasonal)
- Figure 2: Occupancy detection confusion matrix (per room)
- Figure 3: Anomaly detection latency distribution
- Discussion: Real-world deployment challenges and user acceptance

---

## Summary: Application Papers Timeline

```
WEEK 7:
  Mon-Tue: Run robotics experiments (reaching, grasping)
  Wed-Thu: Draft robotics paper (methods, results, discussion)
  Fri:     Publish to arXiv

WEEK 8:
  Mon-Tue: Run medical experiments (arrhythmia, continual learning)
  Wed-Thu: Draft medical paper
  Fri:     Publish to arXiv

WEEK 9:
  Mon-Tue: Run smart home experiments (occupancy, energy, anomaly)
  Wed-Thu: Draft smart home paper
  Fri:     Publish to arXiv

WEEK 10:
  Mon-Tue: Polish all papers
  Wed-Thu: Create supplementary materials (extended results, proofs)
  Fri:     Final review before conference/journal submission
```

---

## Key Success Criteria

**Robotics**:
- ✅ 87% success rate on both 6-DOF and 7-DOF robots
- ✅ Generalization works: 85%+ on unseen morphology
- ✅ Beats hand-coded fusion by >10%

**Medical**:
- ✅ 96% sensitivity for arrhythmia detection
- ✅ Continual learning: < 5% accuracy drop over 30 days
- ✅ Robust to sensor dropout (30% dropout = 2% accuracy loss)

**Smart Home**:
- ✅ 96% occupancy detection accuracy
- ✅ 20-30% energy savings vs baseline
- ✅ Anomaly detection < 30 minutes
- ✅ User satisfaction > 8/10 in real home

---

## Next Steps

**Before Week 7 (Robotics)**:
- [ ] Set up PyBullet environment
- [ ] Download MIMIC-III dataset (medical)
- [ ] Build smart home simulator

**During Week 7-10**:
- [ ] Run experiments as scheduled
- [ ] Draft papers using outlines from PAPER_OUTLINES.md
- [ ] Create figures using templates from FIGURE_TEMPLATES.md
- [ ] Publish to arXiv as completed

All frameworks are ready to implement. Begin with robotics (Week 7) while main paper is at publishers.

