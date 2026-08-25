"""
Base interface object and system-level environment access: serial port
discovery, .env lookups, and shared runtime configuration
"""
import json
import os
import time
import serial
import serial.tools.list_ports
from pathlib import Path

from behavior_rig.config import CONFIG_DIR, PHASE_CONFIG, BAUDRATE
from behavior_rig.link import ArduinoLink


class InterfaceObject:
    """Small base class for hardware/API/operator boundaries used by
    the training system"""
    interface_name = "generic"

    @property
    def ready(self):
        """Return whether this interface is ready for use."""
        return True


class SystemInterface(InterfaceObject):
    interface_name = "environment"

    def __init__(self, config_dir=CONFIG_DIR):
        """
        Initialize paths and load environment settings for the script.

        Args:
            script_dir: Directory containing runtime assets and the .env file.
        """
        self.script_dir = Path(config_dir)
        self.animal_map_path = self.script_dir / "animal_map.json"
        self.credentials_path = self.script_dir / "credentials.json"
        self.env_path = self.script_dir / ".env"

    @property
    def animal_map(self):
        with open(self.animal_map_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError("animal_map.json must be a dict")

        for k, v in data.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise ValueError("animal_map.json keys and values must be strings")

        return data

    def require(self, name):
        """Read a required environment variable.

        Args:
            name: Environment variable name.

        Returns:
            The configured environment value.
        """
        v = os.getenv(name)
        if not v:
            raise RuntimeError(f"{name} not found in .env file")

        return v

    def _env_float(self, name):
        """Read a required environment variable as a float.

        Args:
            name: Environment variable name.

        Returns:
            Parsed floating-point value.
        """
        v = self.require(name).strip()

        if (len(v) >= 2) and (v[0] == v[-1]) and v[0] in {"'", '"'}:
            v = v[1:-1].strip()

        return float(v)

    def validate_assets(self):
        """Verify that required local asset files are present.

        Returns:
            True when validation succeeds.
        """
        if not self.animal_map_path.exists():
            raise FileNotFoundError("animal_map.json not found in the script directory")
        
        if not self.credentials_path.exists():
            raise FileNotFoundError("credentials.json not found in the script directory")

        return True

    @staticmethod
    def _cohort_tokens(map_key):
        return [t.strip() for t in str(map_key).split("_") if t.strip()]

    def animal_exists(self, animal_id, animal_map=None):
        """
        Check whether an animal ID exists in the animal map.

        Args:
            animal_id: Animal identifier to validate.
            animal_map: Optional preloaded animal map.

        Returns:
            True when the animal is present, otherwise False.
        """
        animal_id = str(animal_id).strip()
        animal_map = animal_map or self.animal_map

        return any(animal_id in self._cohort_tokens(key)
                   for key in animal_map.keys())

    def get_workbook_id(self, animal_id, animal_map=None):
        """
        Resolve the Google Sheets workbook ID for an animal.

        Args:
            animal_id: Animal identifier to resolve.
            animal_map: Optional preloaded animal map.

        Returns:
            Workbook ID string, or None if the DEV animal.
        """
        animal_id = str(animal_id).strip()
        animal_map = animal_map or self.animal_map

        if animal_id.upper() == "DEV":
            return None

        self.validate_assets()

        try:
            map_key = next(key for key in animal_map.keys()
                           if animal_id in self._cohort_tokens(key))
        except StopIteration:
            raise ValueError(f"No cohort assigned for animal {animal_id!r}")

        cohort_name = animal_map[map_key]
        workbook_id = f"{cohort_name}_ID"

        return self.require(workbook_id)

    def _find_arduino_port(self):
        ports = serial.tools.list_ports.comports()

        for port in ports:
            dsc = (port.description or "").lower()
            if "arduino" in dsc or "usb serial" in dsc:
                return port.device

        return None

    def serial_connect(self):
        port = self._find_arduino_port()
        if not port:
            print("\n[WARNING] No Arduino detected (continuing anyway)", flush=True)
            return None, False

        try:
            ser = serial.Serial(port, BAUDRATE, timeout=0.05)
            time.sleep(2)

            if not ser.is_open:
                print(f"\n[WARNING] {port} port is not open after initialization (continuing anyway)", flush=True)
                return None, False

            print(f"\nConnected to {port} port\n", flush=True)
            return ser, True
        except Exception as e:
            print(f"\n[WARNING] Could not open Arduino port: {e}", flush=True)
            return None, False

    def get_arduino(self, ser, exceptions=None):
        link = ArduinoLink(ser, exceptions=exceptions)

        try:
            link.start()
        except Exception:
            pass

        return link

    def get_config(self, phase_id):
        cfg = PHASE_CONFIG.get(str(phase_id))
        if cfg is None and str(phase_id) not in {"0", "1"}:
            raise ValueError(f"No PHASE_CONFIG entry for phase {phase_id}")
        
        return {
            "cfg": cfg,
            "engage_ms": self._env_float("BRAKE_ENGAGE_MS"),
            "release_ms": self._env_float("BRAKE_RELEASE_MS"),
            "pulse_ms": self._env_float("SPOUT_PULSE_MS"),
            "threshold": float(cfg.get('threshold', 0.0)) if cfg else 0.0,
            "side": str(cfg.get('side', 'B')).upper() if cfg else "B",
            "reverse": bool(cfg.get('reverse', False)) if cfg else False,
            }

    @property
    def client_id(self):
        """Return the CLIENT_ID environment value as a string."""
        return str(os.getenv("CLIENT_ID"))
