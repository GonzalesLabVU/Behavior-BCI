"""
TCP client interface to PrairieView: starts/stops imaging and relays
session timing to the microscopy acquisition software
"""

from behavior_rig.hardware.system import InterfaceObject
from behavior_rig.hardware.prairie_client import PrairieClient


class PrairieInterface(InterfaceObject):
    interface_name = "prairie"

    def connect(self, imaging_active):
        """
        Connect to Prairie View when imaging is active.

        Args:
            imaging_active: Whether imaging should be initialized.

        Returns:
            PrairieClient instance, or None when imaging is inactive/unavailable.
        """
        if not imaging_active:
            return None

        try:
            client = PrairieClient()
        except Exception as e:
            print(f"\n[WARNING] Imaging requested, but Prairie View connection could not be established "
                  f"({type(e).__name__}: {e}). Continuing without imaging...",
                  flush=True)
            return None

        try:
            configured = client.configure()
        except Exception as e:
            print(f"\n[WARNING] Prairie View CONFIG failed "
                  f"({type(e).__name__}: {e}). Continuing without imaging...",
                  flush=True)
            return None

        if not configured:
            print("\n[WARNING] Prairie View CONFIG returned false. Continuing without imaging...",
                  flush=True)
            return None

        return client

    def finish(self, client, session_data):
        """Finish Prairie imaging and copy timestamps into session data.

        Args:
            client: PrairieClient instance or None.
            session_data: SessionData instance to receive imaging timestamps.
        """
        if client is None:
            return
        
        client.finish()

        if session_data is not None:
            session_data.img["start_ts"] = list(getattr(client, "start_ts", []) or [])
            session_data.img["stop_ts"] = list(getattr(client, "stop_ts", []) or [])
