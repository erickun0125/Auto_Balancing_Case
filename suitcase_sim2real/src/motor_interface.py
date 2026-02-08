#!/usr/bin/env python3
"""
Dynamixel XL430 Motor Interface

Hardware interface for 4 Dynamixel XL430-W250 motors across 2 serial ports,
implementing position control for the Auto Balancing Case balance joint.
Motors [1,2] on port 1 receive +theta, motors [3,4] on port 2 receive -theta.
Uses Dynamixel Protocol 2.0.
"""

import numpy as np
import time
import threading
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS
except ImportError:
    logger.warning("dynamixel_sdk not installed. Install with: pip install dynamixel-sdk")
    logger.warning("Running in mock mode for development without hardware")

    class PortHandler:
        """Mock PortHandler for development without Dynamixel hardware."""
        def __init__(self, port: str) -> None: pass
        def openPort(self) -> bool: return True
        def setBaudRate(self, baudrate: int) -> bool: return True
        def closePort(self) -> None: pass

    class PacketHandler:
        """Mock PacketHandler for development without Dynamixel hardware."""
        def __init__(self, protocol: float) -> None: pass
        def write1ByteTxRx(self, port, id, addr, val): return 0, 0
        def write4ByteTxRx(self, port, id, addr, val): return 0, 0
        def read4ByteTxRx(self, port, id, addr): return 0, 0, 0
        def getTxRxResult(self, result: int) -> str: return "Mock result"
        def getRxPacketError(self, error: int) -> str: return "Mock error"

    COMM_SUCCESS = 0


class DynamixelXL430Interface:
    """Quad-motor interface for the Auto Balancing Case balance joint.

    Controls 4 Dynamixel XL430-W250 motors across 2 serial ports.
    Port 1 motors [1,2] rotate in +theta direction,
    port 2 motors [3,4] rotate in -theta direction (mirrored).

    All positions use a 12-bit range (0-4095) with center at 2048 (0 rad).
    """

    # XL430 Control Table Addresses (Protocol 2.0)
    ADDR_OPERATING_MODE = 11
    ADDR_TORQUE_ENABLE = 64
    ADDR_GOAL_POSITION = 116
    ADDR_GOAL_VELOCITY = 104
    ADDR_GOAL_CURRENT = 102
    ADDR_PRESENT_POSITION = 132
    ADDR_PRESENT_VELOCITY = 128
    ADDR_PRESENT_CURRENT = 126
    ADDR_HARDWARE_ERROR_STATUS = 70

    # XL430 Specifications
    POSITION_RANGE = 4095       # 0-4095 (12-bit encoder resolution)
    VELOCITY_RANGE = 1023       # 0-1023
    CURRENT_RANGE = 1193        # -1193 to 1193

    # Control Modes
    CURRENT_CONTROL_MODE = 0
    VELOCITY_CONTROL_MODE = 1
    POSITION_CONTROL_MODE = 3
    EXTENDED_POSITION_CONTROL_MODE = 4

    # Balance joint constants
    CENTER_POSITION = 2048                                  # Center position (0 rad)
    MAX_ANGLE_RAD = 0.5                                     # Max angle (+/-0.5 rad = +/-28.6 deg)
    POSITION_PER_RAD = POSITION_RANGE / (2 * np.pi)        # Encoder ticks per radian

    # XL430 velocity unit: 0.229 rev/min per tick
    VELOCITY_UNIT_TO_RAD_S = 0.229 * 2 * np.pi / 60.0      # ~0.024 rad/s per tick

    def __init__(self, motor_ids: List[int] = None,
                 device_names: List[str] = None,
                 baudrate: int = 57600,
                 control_mode: int = POSITION_CONTROL_MODE) -> None:
        """Initialize the quad-motor interface.

        Args:
            motor_ids: Motor ID list [1,2,3,4]. First two on port 1 (+theta),
                       last two on port 2 (-theta).
            device_names: Serial port names. [port1, port2].
            baudrate: Serial communication speed (default: 57600 baud).
            control_mode: Dynamixel operating mode (default: position control).
        """
        if motor_ids is None:
            motor_ids = [1, 2, 3, 4]
        if device_names is None:
            device_names = ['/dev/ttyUSB0', '/dev/ttyUSB1']

        self.motor_ids = motor_ids
        self.control_mode = control_mode

        # Split motors by port: port1 = [1,2] (+theta), port2 = [3,4] (-theta)
        self.port1_motors = motor_ids[:2]
        self.port2_motors = motor_ids[2:]

        self.port_lock = threading.Lock()

        # SDK initialization — two ports
        self.port_handler1 = PortHandler(device_names[0])
        self.port_handler2 = PortHandler(device_names[1])
        self.packet_handler = PacketHandler(2.0)

        # State storage for all 4 motors
        self.current_positions: Dict[int, int] = {mid: 0 for mid in motor_ids}
        self.current_velocities: Dict[int, int] = {mid: 0 for mid in motor_ids}
        self.current_currents: Dict[int, int] = {mid: 0 for mid in motor_ids}
        self.goal_positions: Dict[int, int] = {mid: self.CENTER_POSITION for mid in motor_ids}

        # Real-time reading thread
        self.is_reading: bool = False
        self.read_thread: Optional[threading.Thread] = None
        self.read_frequency: int = 50  # Hz

        self._initialize_connection(baudrate)
        self._setup_motors()

    def _initialize_connection(self, baudrate: int) -> None:
        """Open serial ports and set baud rate.

        Args:
            baudrate: Target baud rate for both ports.
        """
        if not self.port_handler1.openPort():
            raise RuntimeError(f"Failed to open port 1: {self.port_handler1}")
        if not self.port_handler1.setBaudRate(baudrate):
            raise RuntimeError("Failed to set baud rate for port 1")

        if not self.port_handler2.openPort():
            raise RuntimeError(f"Failed to open port 2: {self.port_handler2}")
        if not self.port_handler2.setBaudRate(baudrate):
            raise RuntimeError("Failed to set baud rate for port 2")

        logger.info("Dynamixel 2-port connection established")

    def _setup_motors(self) -> None:
        """Configure all 4 motors: disable torque, set mode, enable torque, center."""
        port_groups = [
            (self.port1_motors, self.port_handler1, "port1"),
            (self.port2_motors, self.port_handler2, "port2"),
        ]
        for motor_ids, port_handler, port_name in port_groups:
            for motor_id in motor_ids:
                # 1. Disable torque (required before mode change)
                self._write_1byte(motor_id, self.ADDR_TORQUE_ENABLE, 0, port_handler)
                time.sleep(0.01)
                # 2. Set control mode
                self._write_1byte(motor_id, self.ADDR_OPERATING_MODE, self.control_mode, port_handler)
                time.sleep(0.01)
                # 3. Enable torque
                self._write_1byte(motor_id, self.ADDR_TORQUE_ENABLE, 1, port_handler)
                time.sleep(0.01)
                # 4. Move to center position
                self._write_4byte(motor_id, self.ADDR_GOAL_POSITION, self.CENTER_POSITION, port_handler)
                logger.info("Motor %d (%s) initialized at center position %d",
                            motor_id, port_name, self.CENTER_POSITION)

    def _get_port_handler(self, motor_id: int) -> 'PortHandler':
        """Return the appropriate port handler for a given motor ID."""
        return self.port_handler1 if motor_id in self.port1_motors else self.port_handler2

    def _write_1byte(self, motor_id: int, address: int, value: int,
                     port_handler: Optional['PortHandler'] = None) -> None:
        """Write 1 byte to a motor register (thread-safe).

        Args:
            motor_id: Target motor ID.
            address: Control table address.
            value: Value to write.
            port_handler: Port handler override. Auto-detected if None.
        """
        if port_handler is None:
            port_handler = self._get_port_handler(motor_id)
        with self.port_lock:
            result, error = self.packet_handler.write1ByteTxRx(port_handler, motor_id, address, value)
            if result != COMM_SUCCESS:
                logger.error("Write error (motor %d): %s", motor_id, self.packet_handler.getTxRxResult(result))
            if error != 0:
                logger.error("Hardware error (motor %d): %s", motor_id, self.packet_handler.getRxPacketError(error))

    def _write_4byte(self, motor_id: int, address: int, value: int,
                     port_handler: Optional['PortHandler'] = None) -> None:
        """Write 4 bytes to a motor register (thread-safe).

        Args:
            motor_id: Target motor ID.
            address: Control table address.
            value: Value to write.
            port_handler: Port handler override. Auto-detected if None.
        """
        if port_handler is None:
            port_handler = self._get_port_handler(motor_id)
        with self.port_lock:
            result, error = self.packet_handler.write4ByteTxRx(port_handler, motor_id, address, value)
            if result != COMM_SUCCESS:
                logger.error("Write error (motor %d): %s", motor_id, self.packet_handler.getTxRxResult(result))
            if error != 0:
                logger.error("Hardware error (motor %d): %s", motor_id, self.packet_handler.getRxPacketError(error))

    def _read_4byte(self, motor_id: int, address: int,
                    port_handler: Optional['PortHandler'] = None) -> int:
        """Read 4 bytes from a motor register (thread-safe).

        Args:
            motor_id: Target motor ID.
            address: Control table address.
            port_handler: Port handler override. Auto-detected if None.

        Returns:
            Register value as unsigned 32-bit integer.
        """
        if port_handler is None:
            port_handler = self._get_port_handler(motor_id)
        with self.port_lock:
            value, result, error = self.packet_handler.read4ByteTxRx(port_handler, motor_id, address)
            if result != COMM_SUCCESS:
                logger.error("Read error (motor %d): %s", motor_id, self.packet_handler.getTxRxResult(result))
                return 0
            if error != 0:
                logger.error("Hardware error (motor %d): %s", motor_id, self.packet_handler.getRxPacketError(error))
            return value

    def start_real_time_reading(self) -> None:
        """Start background thread for continuous motor state polling."""
        if self.is_reading:
            return
        self.is_reading = True
        self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.read_thread.start()
        logger.info("Real-time motor reading started")

    def stop_real_time_reading(self) -> None:
        """Stop background reading thread."""
        self.is_reading = False
        if self.read_thread:
            self.read_thread.join()
        logger.info("Real-time motor reading stopped")

    def _read_loop(self) -> None:
        """Background loop: poll position, velocity, and current from all motors."""
        while self.is_reading:
            start_time = time.time()

            for motor_id in self.motor_ids:
                # Position (unsigned 32-bit, 0-4095)
                pos = self._read_4byte(motor_id, self.ADDR_PRESENT_POSITION)
                self.current_positions[motor_id] = pos

                # Velocity (signed 32-bit)
                vel = self._read_4byte(motor_id, self.ADDR_PRESENT_VELOCITY)
                if vel > 2147483647:
                    vel -= 4294967296
                self.current_velocities[motor_id] = vel

                # Current (signed 16-bit, read as 4-byte)
                curr = self._read_4byte(motor_id, self.ADDR_PRESENT_CURRENT)
                if curr > 32767:
                    curr -= 65536
                self.current_currents[motor_id] = curr

            # Maintain read frequency
            elapsed = time.time() - start_time
            sleep_time = (1.0 / self.read_frequency) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def set_command(self, action: float) -> None:
        """Send a unified angle command to all 4 motors.

        Motors [1,2] receive +theta, motors [3,4] receive -theta (mirrored).
        The action is expected to be pre-clipped by the bridge.

        Args:
            action: Target angle in radians (from RL policy, typically [-0.5, 0.5]).
        """
        for i, motor_id in enumerate(self.motor_ids):
            if i < 2:  # Port 1 motors: +theta
                position = int(self.CENTER_POSITION + action * self.POSITION_PER_RAD)
            else:       # Port 2 motors: -theta (mirrored)
                position = int(self.CENTER_POSITION - action * self.POSITION_PER_RAD)

            # Hardware safety clamp
            if position < 0 or position > self.POSITION_RANGE:
                logger.warning("Position %d out of range [0, %d] for motor %d",
                               position, self.POSITION_RANGE, motor_id)
                position = max(0, min(position, self.POSITION_RANGE))

            self._write_4byte(motor_id, self.ADDR_GOAL_POSITION, position)
            self.goal_positions[motor_id] = position

    def get_state(self) -> Dict[str, float]:
        """Get unified joint state averaged from all 4 motors.

        Converts -theta motors back to +theta frame before averaging.

        Returns:
            Dictionary with 'position' (radians) and 'velocity' (rad/s).
        """
        positions: List[float] = []
        velocities: List[float] = []

        for i, motor_id in enumerate(self.motor_ids):
            pos = self.current_positions[motor_id]
            vel = self.current_velocities[motor_id]

            if i < 2:
                # Port 1 motors (+theta): use as-is
                positions.append(pos)
                velocities.append(vel)
            else:
                # Port 2 motors (-theta): convert to +theta frame
                angle_from_center = (self.CENTER_POSITION - pos) / self.POSITION_PER_RAD
                converted_pos = self.CENTER_POSITION + angle_from_center * self.POSITION_PER_RAD
                positions.append(converted_pos)
                velocities.append(-vel)

        avg_position = np.mean(positions)
        avg_velocity = np.mean(velocities)

        angle_rad = (avg_position - self.CENTER_POSITION) / self.POSITION_PER_RAD
        # XL430 velocity unit: 0.229 rev/min per tick
        velocity_rad_s = avg_velocity * self.VELOCITY_UNIT_TO_RAD_S

        return {
            'position': angle_rad,
            'velocity': velocity_rad_s
        }

    def shutdown(self) -> None:
        """Safely shut down: center motors, disable torque, close ports."""
        try:
            self.stop_real_time_reading()

            # Move all motors to center position
            for motor_id in self.motor_ids:
                self._write_4byte(motor_id, self.ADDR_GOAL_POSITION, self.CENTER_POSITION)
            time.sleep(1.0)

            # Disable torque
            for motor_id in self.motor_ids:
                self._write_1byte(motor_id, self.ADDR_TORQUE_ENABLE, 0)

        except Exception as e:
            logger.error("Shutdown error: %s", e)
        finally:
            if hasattr(self, 'port_handler1') and self.port_handler1:
                self.port_handler1.closePort()
            if hasattr(self, 'port_handler2') and self.port_handler2:
                self.port_handler2.closePort()
            logger.info("Dynamixel quad-motor interface shut down (2 ports)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    interface = DynamixelXL430Interface(
        motor_ids=[1, 2, 3, 4],
        device_names=['/dev/ttyUSB0', '/dev/ttyUSB1'],
        control_mode=DynamixelXL430Interface.POSITION_CONTROL_MODE
    )

    try:
        interface.start_real_time_reading()

        test_angles = [0.0, 0.3, -0.3, 0.0]
        for angle in test_angles:
            logger.info("Target angle: %.2f rad (%.1f deg)", angle, np.degrees(angle))
            interface.set_command(angle)

            for _ in range(20):
                state = interface.get_state()
                logger.info("  Position: %.3f rad, Velocity: %.3f rad/s",
                            state['position'], state['velocity'])
                time.sleep(0.1)
            time.sleep(1.0)
    finally:
        interface.shutdown()
