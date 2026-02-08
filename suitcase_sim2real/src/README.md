# Sim2Real Bridge — Source Code

This directory contains the Python bridge that deploys the Isaac Lab-trained RL policy on real hardware at 50 Hz.

## Files

| File | Description |
|------|-------------|
| `run_policy.py` | CLI entry point (`calibrate`, `episode`, `continuous` modes) |
| `bridge.py` | Main 50 Hz control loop orchestrator |
| `policy_interface.py` | RSL-RL ActorCritic model loading and inference |
| `motor_interface.py` | Dynamixel XL430 motor control (4 motors, 2 serial ports) |
| `sensor_interface.py` | HX711 load cell sensor interface (5 channels via Arduino) |
| `config_manager.py` | YAML configuration loader with platform overrides |
| `config/interface_config.yml` | Hardware and policy configuration |

## Quick Start

```bash
# Install dependencies
pip install -r ../requirements.txt

# Calibrate load cells (first time only)
python run_policy.py --mode calibrate

# Run single episode (4000 steps = 8 seconds)
python run_policy.py --mode episode

# Run continuously until Ctrl+C
python run_policy.py --mode continuous
```

See the [project README](../../README.md) for full documentation.
