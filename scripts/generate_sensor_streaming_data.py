"""
Generate IoT sensor streams for validation.

Simulates:
- Smartwatch: HR, accelerometer (3-axis), skin conductance
- Environmental: temperature, humidity, CO2, light
- Robot arm: 6-DOF joint angles, end-effector F/T (3-axis)
"""

import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_smartwatch_data(
    n_sequences: int,
    sequence_length: int,
    seed: int = 42,
) -> np.ndarray:
    """Generate smartwatch data (HR, acc, GSR)."""
    np.random.seed(seed)

    data = np.zeros((n_sequences, sequence_length, 7))

    for seq_id in range(n_sequences):
        # Heart rate: realistic variation (60-100 bpm)
        hr = 70 + 15 * np.sin(np.linspace(0, 4 * np.pi, sequence_length))
        hr += np.random.normal(0, 2, sequence_length)
        data[seq_id, :, 0] = np.clip(hr, 40, 180)

        # Accelerometer (3-axis): random walk around 0g
        for ax in range(3):
            acc = np.cumsum(np.random.normal(0, 0.05, sequence_length))
            data[seq_id, :, 1 + ax] = np.clip(acc, -2, 2)

        # Skin conductance resistance: slow variation (0.5-2 µS)
        gsr = 1.0 + 0.5 * np.sin(np.linspace(0, 2 * np.pi, sequence_length))
        gsr += np.random.normal(0, 0.1, sequence_length)
        data[seq_id, :, 4] = np.clip(gsr, 0, 5)

        # Dummy features for alignment
        data[seq_id, :, 5:7] = np.random.normal(0, 1, (sequence_length, 2))

    return data


def generate_environmental_data(
    n_sequences: int,
    sequence_length: int,
    seed: int = 42,
) -> np.ndarray:
    """Generate room environmental data."""
    np.random.seed(seed)

    data = np.zeros((n_sequences, sequence_length, 4))

    for seq_id in range(n_sequences):
        # Temperature: 18-26°C with daily variation
        temp = 22 + 3 * np.sin(np.linspace(0, 2 * np.pi, sequence_length))
        temp += np.random.normal(0, 0.5, sequence_length)
        data[seq_id, :, 0] = np.clip(temp, 15, 30)

        # Humidity: 30-70% with inverse temp correlation
        humidity = 50 - 10 * np.sin(np.linspace(0, 2 * np.pi, sequence_length))
        humidity += np.random.normal(0, 3, sequence_length)
        data[seq_id, :, 1] = np.clip(humidity, 20, 90)

        # CO2: 400-600 ppm, increases during activity
        co2 = 450 + 100 * np.abs(np.sin(np.linspace(0, 2 * np.pi, sequence_length)))
        co2 += np.random.normal(0, 10, sequence_length)
        data[seq_id, :, 2] = np.clip(co2, 300, 1000)

        # Light: 0-1000 lux with periodic variation (day/night)
        light = 500 + 400 * np.abs(np.sin(np.linspace(0, np.pi, sequence_length)))
        light += np.random.normal(0, 50, sequence_length)
        data[seq_id, :, 3] = np.clip(light, 0, 2000)

    return data


def generate_robot_arm_data(
    n_sequences: int,
    sequence_length: int,
    seed: int = 42,
) -> np.ndarray:
    """Generate robot arm kinematics data."""
    np.random.seed(seed)

    data = np.zeros((n_sequences, sequence_length, 9))

    for seq_id in range(n_sequences):
        # 6-DOF joint angles (radians): -π to π
        for joint in range(6):
            angles = np.sin(np.linspace(0, 2 * np.pi, sequence_length)) * np.pi
            angles += np.random.normal(0, 0.1, sequence_length)
            data[seq_id, :, joint] = np.clip(angles, -np.pi, np.pi)

        # End-effector F/T (3-axis): -10 to +10 N
        for ft_ax in range(3):
            ft = np.random.normal(0, 2, sequence_length)
            data[seq_id, :, 6 + ft_ax] = np.clip(ft, -10, 10)

    return data


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--n_sequences", type=int, default=100, help="Number of sequences to generate"
    )
    parser.add_argument(
        "--sequence_length", type=int, default=500, help="Length of each sequence"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/sensor-streaming",
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating IoT data to {output_dir}")

    # Generate each modality
    logger.info("Generating smartwatch data...")
    smartwatch_data = generate_smartwatch_data(
        args.n_sequences, args.sequence_length, seed=args.seed
    )
    np.save(output_dir / "smartwatch.npy", smartwatch_data)
    logger.info(f"  Shape: {smartwatch_data.shape}")

    logger.info("Generating environmental data...")
    env_data = generate_environmental_data(
        args.n_sequences, args.sequence_length, seed=args.seed
    )
    np.save(output_dir / "room_env.npy", env_data)
    logger.info(f"  Shape: {env_data.shape}")

    logger.info("Generating robot arm data...")
    robot_data = generate_robot_arm_data(
        args.n_sequences, args.sequence_length, seed=args.seed
    )
    np.save(output_dir / "robot_arm.npy", robot_data)
    logger.info(f"  Shape: {robot_data.shape}")

    # Save metadata
    metadata = {
        "n_sequences": args.n_sequences,
        "sequence_length": args.sequence_length,
        "modalities": {
            "smartwatch": {
                "shape": list(smartwatch_data.shape),
                "features": [
                    "HR",
                    "Acc-X",
                    "Acc-Y",
                    "Acc-Z",
                    "GSR",
                    "Dummy1",
                    "Dummy2",
                ],
            },
            "room_env": {
                "shape": list(env_data.shape),
                "features": ["Temperature", "Humidity", "CO2", "Light"],
            },
            "robot_arm": {
                "shape": list(robot_data.shape),
                "features": ["J1", "J2", "J3", "J4", "J5", "J6", "Fx", "Fy", "Fz"],
            },
        },
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Done! Data saved to {output_dir}")
    logger.info(f"Metadata: {metadata}")


if __name__ == "__main__":
    main()
