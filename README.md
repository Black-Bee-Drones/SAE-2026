# SAE EletroQuad 2026 - Black Bee Drones

**2nd Place**

This repository contains the autonomous mission code developed by [Black Bee Drones](https://github.com/Black-Bee-Drones) for the Competição EletroQuad SAE BRASIL – AXIA Energia 2026 (14–17 May 2026, Univap, São José dos Campos).

## Competition Overview

EletroQuad 2026 required every team to fly three outdoor GPS missions with a standardized quadrotor. Vision and onboard compute identify targets; the flight controller executes the motion.

- **Bouncing 2.0** — Find the landing pad whose geometric figure and number match the takeoff marker, then land inside that figure.
- **Hang the Right Wire** — Two suspended ropes; an orange sphere marks the correct one. Hang a hook on that rope and return.
- **Faulty or Not** — Read analog manometers at three waypoints. A reading above the expected pressure is faulty; the ground station reports the result.

## Technical Stack

### Hardware

- **Flight controller**: [ArduPilot](https://ardupilot.org/) with GPS pose
- **Onboard computer**: [Jetson Orin Nano](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)
- **Down camera**: Arducam IMX662 USB (hook: 1920×1080; bouncing: 1280×720 OpenCV webcam)
- **Altitude**: Benewake TFLuna lidar (hook via Nectar `rangefinder_node`; faulty_or_not has its own node)

### Software

- **[ROS 2](https://docs.ros.org/)** Humble
- **[MAVROS](https://github.com/mavlink/mavros)** (hook, bouncing) or direct MAVLink UDP `14550` (faulty_or_not)
- **[Nectar SDK](https://github.com/Black-Bee-Drones/nectar-sdk)** — drone, vision, and AI for hook and bouncing ([docs](https://black-bee-drones.github.io/nectar-sdk/))
- **zaxis** — MAVLink drone API used by faulty_or_not
- **[Yasmin](https://github.com/uleroboticsgroup/yasmin)** — state machines
- **[Ultralytics YOLO](https://docs.ultralytics.com/)** — detection, segmentation, and classification

### Models

Datasets and trained weights: [blackbeedrones/sae-2026](https://huggingface.co/collections/blackbeedrones/sae-2026).

## Missions

### Bouncing 2.0

**Objective**: Climb, find the pad whose shape (hexagon / star / triangle) plus number (3 / 4 / 5) match the takeoff ArUco, and land with the gear inside the figure.

**Implementation**:
- Climb to 6 m and photograph four relative waypoints
- YOLO detector plus classifier; ArUco `DICT_5X5_*` for the takeoff number
- PID centering, then descent to 1.1 m

**Documentation**: [bouncing/README.md](bouncing/README.md)

### Hang the Right Wire

**Objective**: Take off from the arena center, find the orange sphere, fly along the longer rope arm, align perpendicular to the hose, descend on lidar, release the hook with a servo, and land.

**Implementation**:
- YOLO-seg classes `sphere` and `rose` at 960 px
- Metric PIDs park the **hook** (not the camera nadir) a fixed distance from the sphere
- Side choice from instance lengths; predictive yaw; stand-off align then lidar descent
- Servo RC AUX 3 (`HOLD_PWM=1000`, `RELEASE_PWM=2000`)
- Gazebo Harmonic + ArduCopter SITL world in-package

**Documentation**: [hook/README.md](hook/README.md)

### Faulty or Not

**Objective**: Visit three manometers, classify the needle (20–120 psi), and tell the ground station whether the reading is above the expected value.

**Implementation**:
- Two processes: air mission and Ground Monitor, handshake on `/gauge/reading` and `/gauge/ack`
- Coarse detect (`coarse.pt`) then classify (`best.pt`)
- Faulty if actual psi > expected; plays `alarm.mp3` or `not_faulty.mp3`

**Documentation**: [faulty_or_not/README.md](faulty_or_not/README.md)

## File Structure

```
SAE-2026/
├── bouncing/           # Bouncing 2.0
├── hook/               # Hang the Right Wire
├── faulty_or_not/      # Faulty or Not
├── Regulamento_EletroQuad_2026_portugues.pdf
├── ProcedimentosOperacionaisEletroQuadSAEBRASIL-AXIAEnergia2026.pdf
└── README.md
```

Each mission is a separate `ament_python` ROS 2 package.

## Running Missions

Build the packages you need (hook also needs Nectar):

```bash
cd ~/ros2_ws
colcon build --packages-select bouncing hook faulty_or_not nectar nectar_interfaces
source install/setup.bash
```

- **Bouncing 2.0**: `ros2 run bouncing mangalarga`
- **Hang the Right Wire**: `ros2 run hook mangalarga`
- **Faulty or Not**: `ros2 run faulty_or_not faulty_or_not`

Per-package READMEs cover MAVROS/lidar, CLI flags, and simulation.

## References

### Competition

- [EletroQuad SAE BRASIL](https://saebrasil.org.br/programas-estudantis/eletroquad/)
- [Rules (PDF)](Regulamento_EletroQuad_2026_portugues.pdf)
- [Operational procedures (PDF)](ProcedimentosOperacionaisEletroQuadSAEBRASIL-AXIAEnergia2026.pdf)

### Software

- [Nectar SDK](https://github.com/Black-Bee-Drones/nectar-sdk)
- [Nectar documentation](https://black-bee-drones.github.io/nectar-sdk/)
- [ROS 2](https://docs.ros.org/)
- [MAVROS](https://github.com/mavlink/mavros)
- [Yasmin](https://github.com/uleroboticsgroup/yasmin)

### Models

- [SAE 2026 collection](https://huggingface.co/collections/blackbeedrones/sae-2026)

## Team

**Black Bee Drones** - Latin America's first academic autonomous drone team  
Federal University of Itajubá (UNIFEI), Brazil
