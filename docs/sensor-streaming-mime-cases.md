# Sensor-Streaming Input Cases and Naming

## Decision

- Use `sensor_streaming` as the canonical runtime experiment ID.
- Do not maintain legacy `iot` aliases.

## Canonical Sensor-Streaming MIME Cases

These are defined in `src/runtime/mime_cases.py` and used as the source of truth for sensor-streaming modality dimensions:

1. `smartwatch`  
   - MIME: `application/vnd.sensory.smartwatch.timeseries+tensor`  
   - Feature width: `7`

2. `room_env`  
   - MIME: `application/vnd.sensory.environment.timeseries+tensor`  
   - Feature width: `4`

3. `robot_arm`  
   - MIME: `application/vnd.sensory.robot-arm.timeseries+tensor`  
   - Feature width: `9`

## Why this shape

- One canonical mapping prevents modality/schema drift across training, evaluation, and orchestration paths.
- Runtime modality setup is derived from this registry to keep code and docs synchronized.
