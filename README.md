# Auto Balancing Case: End-to-End Robotics System

A comprehensive robotics project implementing an **Auto Balancing Suitcase** from CAD design to real-world deployment, featuring hardware design, physics simulation, reinforcement learning, and low-level motor control.

**Note: This project is currently under active development. Components and documentation will be continuously updated.**

## Project Overview

This repository demonstrates a complete end-to-end robotics development pipeline:

- **Hardware Design & Manufacturing**: CAD-based mechanical design with URDF models for a self-balancing wheeled suitcase
- **Physics Simulation**: High-fidelity simulation environment using NVIDIA Isaac Lab
- **Reinforcement Learning**: Policy training for balance control using PPO algorithm  
- **Sim2Real Transfer**: Bridge between simulation and real hardware deployment
- **Low-Level Control**: Real-time motor control using Dynamixel SDK

The system learns to maintain balance of a wheeled suitcase against external disturbances through a single actuated joint between the base platform and luggage case.

## System Architecture

```
Auto Balancing Case System
├── Hardware Design (suitcase_model/)
│   ├── CAD Models & URDF
│   ├── Mechanical Assembly
│   └── ROS Integration
├── Simulation Environment (IsaacLab/)
│   ├── Physics Simulation
│   ├── RL Environment
│   └── Policy Training
├── Sim2Real Pipeline (suitcase_sim2real/)
│   ├── Hardware Bridge
│   ├── Policy Deployment
│   └── Real-time Control
└── Motor Control (DynamixelSDK/)
    ├── Low-level Communication
    ├── Motor Interface
    └── Hardware Abstraction
```

## Repository Structure

### Hardware Design (`suitcase_model/`)
Contains CAD-based hardware design and manufacturing specifications:
- **`Auto_Balancing_Case/`**: Primary URDF model with joint definitions
- **`assembly_final_donot_export_description/`**: ROS package with meshes and launch files
- **`assembly_final_urdf_description/`**: Complete URDF description with Gazebo integration

**Key Components:**
- 4-wheel mobile platform with individual wheel control
- Single balance actuator (Revolute_motor) with ±30° range
- Luggage case body with handle assembly
- Contact sensors for ground interaction

### Simulation Environment (`IsaacLab/`)
Physics simulation and reinforcement learning environment built on NVIDIA Isaac Sim:

- **Environment**: `Isaac-Suitecase-Flat-v0` - Training environment with external disturbances
- **Task**: Balance control through single actuated joint while maintaining wheel ground contact
- **Algorithm**: PPO (Proximal Policy Optimization) with custom reward functions
- **Observations**: Joint positions, velocities, contact forces, body orientation
- **Actions**: Target position for balance actuator

**Key Features:**
- High-fidelity physics simulation with PhysX
- Domain randomization for robustness
- Multiple disturbance patterns (force, velocity, torque)
- Contact-aware reward design

### Sim2Real Pipeline (`suitcase_sim2real/`)
Bridge between simulation and real hardware deployment:

#### `rl_hardware_bridge.py`
- Loads trained Isaac Lab policies
- Manages real-time control loop at 50Hz
- Handles observation preprocessing and action post-processing
- Provides episode-based and continuous control modes

#### `dynamixel_xl430_interface.py`
- Real-time hardware interface for Dynamixel XL430 motors
- Multi-threaded reading for low-latency sensor feedback
- Position, velocity, and current control modes
- Observation normalization and action denormalization

### Motor Control (`DynamixelSDK/`)
Official Dynamixel SDK for low-level motor communication:
- Protocol 2.0 communication
- Multiple language bindings (Python, C++, Java, etc.)
- Hardware abstraction layer
- Real-time control capabilities

## Getting Started

### Prerequisites
- Ubuntu 20.04/22.04
- Python 3.8+
- NVIDIA GPU (for Isaac Lab simulation)
- Dynamixel XL430 motors (for hardware deployment)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd auto-balancing-case
```

2. **Setup Isaac Lab Environment**
```bash
cd IsaacLab
./isaaclab.sh --install
```

3. **Install Sim2Real Dependencies**
```bash
cd suitcase_sim2real
pip install torch numpy pyyaml
```

4. **Setup Dynamixel SDK**
```bash
cd DynamixelSDK/python
pip install .
```

### Usage Examples

#### 1. Training in Simulation
```bash
cd IsaacLab
./isaaclab.sh -p scripts/rsl_rl/train.py --task Isaac-Suitecase-Flat-v0 --num_envs 4096
```

#### 2. Testing Trained Policy
```bash
./isaaclab.sh -p scripts/rsl_rl/play.py --task Isaac-Suitecase-Flat-Play-v0 --checkpoint path/to/model.pt
```

#### 3. Hardware Deployment
```bash
cd suitcase_sim2real
python run_policy.py --model_path path/to/trained_model.pt --motor_ids 1 2 3 4
```

#### 4. Real-time Hardware Bridge
```python
from rl_hardware_bridge import RLHardwareBridge, PolicyConfig

# Configure policy
config = PolicyConfig(
    model_path="trained_policy.pt",
    observation_space={'joint_pos': {'shape': [4]}, 'joint_vel': {'shape': [4]}},
    action_space={'shape': [1]},  # Single balance actuator
    control_frequency=50.0
)

# Initialize and run
bridge = RLHardwareBridge(config, motor_ids=[1, 2, 3, 4])
bridge.run_continuous()
```

## Technical Specifications

### Hardware Specifications
- **Platform**: 4-wheel mobile base (36cm × 23cm)
- **Actuator**: Single balance joint with ±30° range
- **Motors**: Dynamixel XL430 series
- **Weight**: Approximately 6.5kg total system mass
- **Control Frequency**: 50Hz real-time control

### Simulation Environment
- **Physics Engine**: NVIDIA PhysX
- **Simulation Frequency**: 120Hz physics, 50Hz control
- **Training**: Multiple parallel environments
- **Algorithm**: PPO with custom reward shaping
- **Observation Dim**: Variable (joint states + contact forces)
- **Action Dim**: 1 (balance actuator target)

## Key Features

### Advanced Simulation
- **Domain Randomization**: Mass, friction, motor parameters
- **Disturbance Modeling**: External forces, velocity perturbations
- **Contact Dynamics**: Wheel-ground interaction modeling
- **Sensor Simulation**: Realistic noise and latency modeling

### Robust Control
- **Multi-Modal Disturbances**: Force, torque, and velocity-based
- **Contact-Aware Policy**: Maintains wheel ground contact
- **Real-time Performance**: 50Hz control loop
- **Fault Tolerance**: Motor error detection and recovery

### Production Ready
- **Modular Architecture**: Separate simulation, training, and deployment
- **Hardware Abstraction**: Easy integration with different motor types
- **Monitoring & Logging**: Comprehensive system state tracking
- **Safety Features**: Emergency stops and limit checking

## Research Applications

This project serves as a comprehensive example for:
- **Sim2Real Transfer**: Bridging simulation and reality gaps
- **Contact-Rich Manipulation**: Learning with complex contact dynamics  
- **Real-time RL Deployment**: Production-ready policy deployment
- **Hardware-Software Co-design**: Integrated system development
- **Robotic Product Development**: End-to-end development pipeline

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **NVIDIA Isaac Lab**: Physics simulation and RL framework
- **Robotis Dynamixel**: Motor hardware and SDK
- **ROS Community**: URDF standards and tools
- **PyTorch**: Deep learning framework

---

*This project demonstrates the complete robotics development lifecycle from concept to deployment, showcasing modern tools and techniques in robotics, simulation, and machine learning.*