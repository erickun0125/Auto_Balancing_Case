#!/usr/bin/env python3
"""
Auto Balancing Case Sim2Real Policy Runner

CLI entry point for running the trained RL policy on real hardware.
Supports three modes: calibrate (load cell setup), episode (single run),
and continuous (indefinite operation).
"""

import argparse
import os
import sys
import logging
from pathlib import Path

from bridge import AutoBalancingCaseBridge
from config_manager import ConfigManager

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging with console and optional file output.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
        ]
    )


def validate_config_file(config_path: str) -> bool:
    """Check if the configuration file exists.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        True if the file exists.
    """
    if not os.path.exists(config_path):
        logger.error("Config file not found: %s", config_path)
        logger.error("Ensure config/interface_config.yml exists in the run directory.")
        return False
    return True


def main() -> None:
    """Parse arguments and run the selected mode."""
    parser = argparse.ArgumentParser(description='Auto Balancing Case Sim2Real Policy Runner')
    parser.add_argument('--config', '-c', type=str, default='config/interface_config.yml',
                        help='Path to config file (default: config/interface_config.yml)')
    parser.add_argument('--mode', '-m', type=str,
                        choices=['episode', 'continuous', 'calibrate'],
                        default='calibrate',
                        help='Run mode: episode | continuous | calibrate')

    args = parser.parse_args()

    # Setup logging
    setup_logging("INFO")

    # Resolve config path
    config_dir = Path(__file__).parent / 'config'
    if args.config == 'config/interface_config.yml':
        config_path = config_dir / 'interface_config.yml'
    else:
        config_path = Path(args.config)

    if not validate_config_file(str(config_path)):
        return

    # Load and validate configuration
    try:
        logger.info("Loading config: %s", config_path)
        config_manager = ConfigManager(str(config_path))

        if not config_manager.validate_config():
            logger.error("Config validation failed")
            return

        logger.info("Config loaded successfully")
        logger.info("Config summary: %s", config_manager.get_config_summary())
    except Exception as e:
        logger.error("Config load error: %s", e)
        return

    # Verify policy file exists
    if not os.path.exists(config_manager.policy.model_path):
        logger.error("Policy file not found: %s", config_manager.policy.model_path)
        logger.error("Check model_path in config file.")
        return

    # Initialize bridge
    try:
        logger.info("Initializing Auto Balancing Case Bridge...")
        bridge = AutoBalancingCaseBridge(str(config_path))
        logger.info("Bridge initialized")
    except Exception as e:
        logger.error("Bridge initialization error: %s", e)
        return

    try:
        if args.mode == 'calibrate':
            print("=== Load Cell Calibration Mode ===")
            print("Follow the prompts to place/remove weights on each sensor.")
            input("Press Enter to begin...")
            bridge.calibrate_load_cells()

        elif args.mode == 'episode':
            print("=== Episode Mode ===")
            print(f"Running for up to {config_manager.system.max_episode_steps} steps.")
            print("Press Ctrl+C to stop early.")
            input("Press Enter to start...")
            bridge.run_episode()

        elif args.mode == 'continuous':
            print("=== Continuous Mode ===")
            print("Running until Ctrl+C is pressed.")
            input("Press Enter to start...")
            bridge.run_continuous()

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    except Exception as e:
        logger.error("Runtime error: %s", e)

    finally:
        logger.info("Cleaning up...")
        bridge.shutdown()
        logger.info("Done.")


if __name__ == "__main__":
    main()
