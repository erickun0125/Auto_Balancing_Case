# Sim2Real Deployment

Deploys the Isaac Lab-trained PPO policy on real Auto Balancing Case hardware at 50 Hz.

## Directory Structure

```
suitcase_sim2real/
├── src/                             # Python bridge code
│   ├── run_policy.py                #   CLI entry point (calibrate/episode/continuous)
│   ├── bridge.py                    #   50 Hz control loop orchestrator
│   ├── policy_interface.py          #   RSL-RL model loading & inference
│   ├── motor_interface.py           #   Dynamixel XL430 motor control (4 motors, 2 ports)
│   ├── sensor_interface.py          #   HX711 load cell interface (5 channels via Arduino)
│   ├── config_manager.py            #   YAML config with platform overrides
│   └── config/interface_config.yml  #   All hardware/policy settings
├── firmware/                        # Arduino firmware
│   └── hx711_load_cells/
│       └── hx711_load_cells.ino     #   6× HX711 ADC readout, CSV at 10 Hz
└── requirements.txt                 # Python dependencies
```

## Hardware

| Component | Details |
|-----------|---------|
| Motors | 4× Dynamixel XL430-W250, Protocol 2.0, 57600 baud, 2 serial ports |
| Sensors | 5× HX711 load cells (4 wheel + 1 handle), Arduino Mega, 115200 baud |
| Control | 50 Hz real-time loop, ±0.5 rad action range, 0.51 rad emergency stop |

## Quick Start

```bash
cd src
pip install -r ../requirements.txt

python run_policy.py --mode calibrate    # First-time load cell calibration
python run_policy.py --mode episode      # Single episode (8 seconds)
python run_policy.py --mode continuous   # Run until Ctrl+C
```

## Platform Configuration

Serial ports are auto-configured via `platform_overrides` in `config/interface_config.yml`:
- **Windows**: COM7 (sensor), COM11/COM12 (motors)
- **Linux**: /dev/ttyACM0 (sensor), /dev/ttyUSB0/ttyUSB1 (motors)

## Safety

- Emergency stop at 0.51 rad tilt threshold
- Watchdog timeout at 1.0 second
- Motors return to center position on shutdown
- Torque disabled after safe stop
