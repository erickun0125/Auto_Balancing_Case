#!/usr/bin/env python3
"""
HX711 Load Cell Interface (Arduino-based)

Reads force data from 5 HX711 load cells (4 wheel + 1 handle) via Arduino serial
connection. Supports calibration, real-time background reading, and force unit
conversion (grams -> Newtons). Communication uses CSV format at 115200 baud.
"""

import numpy as np
import time
import threading
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    logger.warning("pyserial not installed. Install with: pip install pyserial")
    logger.warning("Running in mock mode for development without Arduino hardware")
    SERIAL_AVAILABLE = False

    class serial:  # noqa: N801
        """Mock serial module for development without Arduino hardware."""
        class Serial:
            def __init__(self, port: str, baudrate: int, timeout: int = 1) -> None:
                self.is_open = True
                self.in_waiting = 0
            def close(self) -> None: pass
            def readline(self) -> bytes: return b'0.0,0.0,0.0,0.0,0.0,0.0\n'
            def write(self, data: bytes) -> None: pass
            def flush(self) -> None: pass


class HX711LoadCellInterface:
    """Arduino-based load cell interface for the Auto Balancing Case.

    Reads 6 HX711 channels (5 active: 4 wheel + 2 handle) via Arduino serial.
    Handles calibration (zero offset + scale factor), real-time background reading,
    and conversion from raw ADC values to Newtons.

    Arduino outputs CSV: FR,RR,FL,RL,H1,H2 at 10Hz over 115200 baud serial.
    """

    # Load cell channel names matching Arduino CSV output order
    LOAD_CELL_NAMES = ['wheel_FR', 'wheel_RR', 'wheel_FL', 'wheel_RL', 'handle_1', 'handle_2']

    def __init__(self, arduino_port: str = 'COM7', baudrate: int = 115200) -> None:
        """Initialize the load cell interface.

        Args:
            arduino_port: Serial port for Arduino (Linux: /dev/ttyACM0, Windows: COM7).
            baudrate: Serial communication speed (default: 115200 baud).
        """
        self.arduino_port = arduino_port
        self.baudrate = baudrate

        # Serial connection
        self.serial_connection: Optional[serial.Serial] = None
        self._initialize_serial_connection()

        # Force values storage
        self.current_forces: Dict[str, float] = {name: 0.0 for name in self.LOAD_CELL_NAMES}
        self.calibrated_forces: Dict[str, float] = {name: 0.0 for name in self.LOAD_CELL_NAMES}

        # Real-time reading thread
        self.is_reading: bool = False
        self.read_thread: Optional[threading.Thread] = None
        self.read_frequency: int = 50  # Hz

        # Calibration parameters
        self.zero_offsets: Dict[str, float] = {name: 0.0 for name in self.LOAD_CELL_NAMES}
        self.calibration_factors: Dict[str, float] = {name: 1.0 for name in self.LOAD_CELL_NAMES}

    def _initialize_serial_connection(self) -> None:
        """Establish serial connection with the Arduino."""
        if not SERIAL_AVAILABLE:
            logger.info("Mock mode: creating mock serial connection")
            self.serial_connection = serial.Serial(self.arduino_port, self.baudrate)
            return

        try:
            self.serial_connection = serial.Serial(
                port=self.arduino_port,
                baudrate=self.baudrate,
                timeout=1
            )
            time.sleep(2)  # Wait for Arduino initialization
            logger.info("Arduino connected: %s @ %d baud", self.arduino_port, self.baudrate)
        except Exception as e:
            logger.warning("Arduino connection failed: %s. Switching to mock mode.", e)
            self.serial_connection = serial.Serial(self.arduino_port, self.baudrate)

    def calibrate_single_load_cell(self, cell_name: str, known_weight: float = 200.0) -> bool:
        """Calibrate a single load cell using a known weight.

        Args:
            cell_name: Name of the load cell to calibrate (e.g., 'wheel_FR').
            known_weight: Reference weight in grams.

        Returns:
            True if calibration succeeded.
        """
        if cell_name not in self.LOAD_CELL_NAMES:
            logger.error("Invalid load cell name: '%s'. Valid names: %s",
                         cell_name, self.LOAD_CELL_NAMES)
            return False

        print(f"\n=== '{cell_name}' Load Cell Calibration ===")
        print(f"Remove all weight from '{cell_name}' and wait 5 seconds...")
        time.sleep(5)

        # Tare (zero) the specific channel
        self._send_command(f"TARE_{cell_name.upper()}")
        time.sleep(2)
        self.zero_offsets[cell_name] = 0.0
        print(f"'{cell_name}' tared successfully")

        print(f"\nPlace {known_weight}g weight on '{cell_name}'...")
        input("Press Enter when ready...")

        # Measure calibration values
        print("Measuring calibration values...")
        readings: List[float] = []
        for i in range(10):
            data = self._read_sensor_data()
            if data and cell_name in data:
                readings.append(data[cell_name])
                print(f"  Reading {i + 1}/10: {data[cell_name]:.2f}")
            time.sleep(0.1)

        if not readings:
            logger.error("'%s' calibration failed: no data received", cell_name)
            return False

        avg_reading = np.mean(readings)
        if avg_reading == 0:
            logger.error("'%s' calibration failed: zero reading", cell_name)
            return False

        self.calibration_factors[cell_name] = avg_reading / known_weight
        print(f"'{cell_name}' calibration successful!")
        print(f"  Average reading: {avg_reading:.2f}")
        print(f"  Calibration factor: {self.calibration_factors[cell_name]:.6f}")
        return True

    def calibrate_all_load_cells(self, known_weight: float = 200.0) -> tuple:
        """Calibrate all load cells sequentially with user prompts.

        Args:
            known_weight: Reference weight in grams.

        Returns:
            Tuple of (successful_names, failed_names).
        """
        print("=== Sequential Load Cell Calibration ===")
        print(f"Calibrating {len(self.LOAD_CELL_NAMES)} load cells one by one.")

        successful: List[str] = []
        failed: List[str] = []

        for i, cell_name in enumerate(self.LOAD_CELL_NAMES, 1):
            print(f"\n[{i}/{len(self.LOAD_CELL_NAMES)}] '{cell_name}' calibration")

            while True:
                response = input(f"Calibrate '{cell_name}'? (y/n/s to skip all): ").lower().strip()
                if response in ('y', 'yes'):
                    break
                elif response in ('n', 'no'):
                    failed.append(cell_name)
                    break
                elif response in ('s', 'skip'):
                    failed.extend(self.LOAD_CELL_NAMES[i - 1:])
                    break
                else:
                    print("Enter 'y' (yes), 'n' (no), or 's' (skip all).")

            if response in ('s', 'skip'):
                break
            if response in ('n', 'no'):
                continue

            success = self.calibrate_single_load_cell(cell_name, known_weight)
            (successful if success else failed).append(cell_name)

            if i < len(self.LOAD_CELL_NAMES):
                print("\nMoving to next load cell...")
                time.sleep(2)

        print("\n=== Calibration Summary ===")
        print(f"Successful: {len(successful)} ({', '.join(successful) if successful else 'none'})")
        print(f"Failed/Skipped: {len(failed)} ({', '.join(failed) if failed else 'none'})")

        return successful, failed

    def start_real_time_reading(self) -> None:
        """Start background thread for continuous load cell reading."""
        if self.is_reading:
            return
        self.is_reading = True
        self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.read_thread.start()
        logger.info("Load cell real-time reading started")

    def stop_real_time_reading(self) -> None:
        """Stop background reading thread."""
        self.is_reading = False
        if self.read_thread:
            self.read_thread.join()
        logger.info("Load cell real-time reading stopped")

    def _read_loop(self) -> None:
        """Background loop: read and calibrate sensor data continuously."""
        while self.is_reading:
            start_time = time.time()

            try:
                data = self._read_sensor_data()
                if data:
                    for name in self.LOAD_CELL_NAMES:
                        if name in data:
                            raw_value = data[name]
                            self.current_forces[name] = raw_value

                            # Apply calibration: raw -> grams -> Newtons
                            if self.calibration_factors[name] != 0:
                                grams = (raw_value - self.zero_offsets[name]) / self.calibration_factors[name]
                                self.calibrated_forces[name] = grams * 9.81 / 1000.0  # N
                            else:
                                self.calibrated_forces[name] = 0.0
                        else:
                            self.current_forces[name] = 0.0
                            self.calibrated_forces[name] = 0.0

            except Exception as e:
                logger.error("Sensor read error: %s", e)
                for name in self.LOAD_CELL_NAMES:
                    self.current_forces[name] = 0.0
                    self.calibrated_forces[name] = 0.0

            # Maintain read frequency
            elapsed = time.time() - start_time
            sleep_time = (1.0 / self.read_frequency) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_state(self) -> Dict[str, np.ndarray]:
        """Get current force state formatted for the policy observation.

        Wheel forces use absolute values. Handle force is computed as the
        difference between handle_1 and the average of both handle sensors
        (detecting external force applied to the handle).

        Returns:
            Dictionary with 'wheel_forces' (4-element array, N) in Isaac Lab
            order [FR, RR, FL, RL] and 'handle_force' (1-element array, N).
        """
        wheel_names = ['wheel_FR', 'wheel_RR', 'wheel_FL', 'wheel_RL']
        wheel_forces = [abs(self.calibrated_forces.get(name, 0.0)) for name in wheel_names]

        # Handle force: deviation of handle_1 from the mean of both handle sensors
        h1 = self.calibrated_forces.get('handle_1', 0.0)
        h2 = self.calibrated_forces.get('handle_2', 0.0)
        handle_force = abs(h1 - (h1 + h2) / 2.0)

        return {
            'wheel_forces': np.array(wheel_forces),
            'handle_force': np.array([handle_force])
        }

    def save_calibration(self, filename: str = 'load_cell_calibration.npz') -> None:
        """Save calibration data to a NumPy archive.

        Args:
            filename: Output file path.
        """
        np.savez(filename,
                 zero_offsets=self.zero_offsets,
                 calibration_factors=self.calibration_factors)
        logger.info("Calibration data saved to %s", filename)

    def load_calibration(self, filename: str = 'load_cell_calibration.npz') -> None:
        """Load calibration data from a NumPy archive.

        Args:
            filename: Input file path.

        Raises:
            FileNotFoundError: If calibration file doesn't exist.
        """
        try:
            data = np.load(filename, allow_pickle=True)
            self.zero_offsets = data['zero_offsets'].item()
            self.calibration_factors = data['calibration_factors'].item()
            logger.info("Calibration data loaded from %s", filename)
        except FileNotFoundError:
            logger.warning("Calibration file not found: %s", filename)
            raise
        except Exception as e:
            logger.error("Calibration load error: %s", e)
            raise

    def shutdown(self) -> None:
        """Stop reading and close serial connection."""
        try:
            self.stop_real_time_reading()
            if self.serial_connection and self.serial_connection.is_open:
                self.serial_connection.close()
                logger.info("Arduino serial connection closed")
        except Exception as e:
            logger.error("Shutdown error: %s", e)
        finally:
            self.serial_connection = None

    def _send_command(self, command: str) -> None:
        """Send a text command to the Arduino.

        Args:
            command: Command string (e.g., 'TARE_ALL', 'STATUS').
        """
        if self.serial_connection and self.serial_connection.is_open:
            try:
                self.serial_connection.write(f"{command}\n".encode())
                self.serial_connection.flush()
            except Exception as e:
                logger.error("Command send error: %s", e)

    def _read_sensor_data(self) -> Optional[Dict[str, float]]:
        """Read and parse a single CSV line from the Arduino.

        Expected format: FR,RR,FL,RL,H1,H2 (6 comma-separated floats).

        Returns:
            Dictionary mapping channel names to raw float values, or None.
        """
        if not self.serial_connection or not self.serial_connection.is_open:
            return None

        try:
            if self.serial_connection.in_waiting > 0:
                line = self.serial_connection.readline().decode('utf-8', errors='ignore').strip()

                # Filter Arduino system messages
                if line in ('READY',) or line.startswith(('STATUS:', 'ERROR:', 'TARE_', 'UNKNOWN_')):
                    logger.debug("Arduino system message: %s", line)
                    return None

                # Parse CSV data
                if line and ',' in line:
                    try:
                        values = line.split(',')
                        if len(values) == 6:
                            return {
                                'wheel_FR': float(values[0]),
                                'wheel_RR': float(values[1]),
                                'wheel_FL': float(values[2]),
                                'wheel_RL': float(values[3]),
                                'handle_1': float(values[4]),
                                'handle_2': float(values[5])
                            }
                        else:
                            logger.warning("Unexpected CSV field count (%d): %s", len(values), line)
                    except ValueError as e:
                        logger.warning("CSV parse error: %s, data: '%s'", e, line)
                elif line:
                    logger.debug("Unrecognized format: '%s'", line)

            return None

        except Exception as e:
            logger.error("Data read error: %s", e)
            return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    interface = HX711LoadCellInterface(arduino_port='COM7', baudrate=115200)

    try:
        interface.load_calibration()
        interface.start_real_time_reading()

        for i in range(100):
            state = interface.get_state()
            logger.info("Step %d: Wheels=%s, Handle=%.2f N",
                        i, state['wheel_forces'], state['handle_force'][0])
            time.sleep(0.1)
    finally:
        interface.shutdown()
