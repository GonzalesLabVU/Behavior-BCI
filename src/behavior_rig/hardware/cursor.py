"""
Interface to the cursor/wheel input hardware: reads and exposes live
position data during a running session
"""

from behavior_rig.config import PHASE_CONFIG, EVT_QUEUE, ENC_QUEUE
from behavior_rig.session import _get_easy
from behavior_rig.hardware.system import InterfaceObject
from behavior_rig.hardware.cursor_task import BCI


class CursorInterface(InterfaceObject):
    interface_name = "cursor"

    def connect(self, phase_id, side):
        """Start the cursor task for wheel phases.

        Args:
            phase_id: Training phase identifier.
            side: Target side configuration.

        Returns:
            Tuple of cursor instance or None, and the initial easy-trial flag.
        """
        if int(phase_id) <= 3:
            return None, True

        easy = _get_easy(phase=int(phase_id), trial_n=1, K=5)

        cursor = BCI(
            phase_id=phase_id,
            evt_queue=EVT_QUEUE,
            enc_queue=ENC_QUEUE,
            config=PHASE_CONFIG,
            display_idx=1,
            fullscreen=False,
            easy_threshold=15.0,
        )
        cursor.update_config(easy, side)
        cursor.start()

        return cursor, easy

    def update_trial(self, cursor, is_easy, side):
        """Update cursor task settings for the next trial.

        Args:
            cursor: Active cursor object or None.
            is_easy: Whether the next trial is easy.
            side: Target side for the next trial.
        """
        if cursor is not None:
            cursor.update_config(is_easy, side)

    def stop(self, cursor):
        """Stop the cursor task if it is running.

        Args:
            cursor: Active cursor object or None.

        Returns:
            Cursor stop result, or True when no cursor is active.
        """
        if cursor is not None:
            return cursor.stop()
        
        return True
