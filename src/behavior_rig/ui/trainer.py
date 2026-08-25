"""
Trainer-facing prompts and validation: collects and confirms session
metadata (animal, phase, targets) before a run starts
"""

import sys

from behavior_rig.config import _valid_phases
from behavior_rig.ui.console import _cmd_run
from behavior_rig.hardware.system import InterfaceObject, SystemInterface


class TrainerInterface(InterfaceObject):
    interface_name = "trainer"

    def __init__(self, system=None):
        """Initialize trainer prompts with an environment interface.

        Args:
            env: Optional SystemInterface instance.
        """
        self.system = system or SystemInterface()

    def prompt_flush(self):
        """Prompt whether to flush the spout before the session.

        Returns:
            True when the trainer confirms flushing, otherwise False.
        """
        flush_raw = input("\nFlush spout for 5 seconds? [y/N]:  ")
        flush_choice = _is_affirmative(flush_raw)

        if flush_choice:
            flush_raw = input("This operation will restart the program. Continue? [y/N]:  ")
            return _is_affirmative(flush_raw)

        return flush_choice

    def prompt_animal(self):
        """
        Prompt for and validate the animal ID.

        Returns:
            Tuple of the selected animal ID and loaded animal map.
        """
        animal_map = self.system.animal_map

        while True:
            print("\nAnimal ID:  ", end="", flush=True)

            animal_raw = sys.stdin.readline()
            if animal_raw == "":
                raise EOFError
            
            animal_raw = animal_raw.rstrip("\n").upper()

            if not animal_raw:
                sys.stdout.write('\x1b[1A')
                sys.stdout.write('\x1b[2K')
                sys.stdout.write('Animal ID:  DEV\n')
                sys.stdout.flush()
                animal_id = "DEV"
            else:
                if not self.system.animal_exists(animal_raw, animal_map):
                    print('Please enter a valid animal')
                    continue

                animal_id = animal_raw

            return animal_id, animal_map

    def prompt_phase(self):
        """Prompt for a valid training phase.

        Returns:
            Selected phase ID string.
        """
        valid_phases = _valid_phases()

        while True:
            phase_id = input('Training Phase:  ').strip()
            if phase_id in valid_phases:
                return phase_id

            print('Please enter a valid phase\n', flush=True)

    def prompt_side(self):
        """
        Prompt for target side to use (phase 4 only)
        
        Returns:
            'L' or 'R' selected by the trainer
        """
        while True:
            side_raw = input('Target side (L/R):  ').strip().upper()
            if side_raw in {'L', 'R'}:
                return side_raw

            print('Please enter L or R\n', flush=True)

    def prompt_imaging(self):
        """Prompt whether imaging is active.

        Returns:
            True when imaging is active, otherwise False.
        """
        imaging_raw = input('\nImaging active? [y/N]:  ')
        return _is_affirmative(imaging_raw)

    def prompt_ephys(self):
        """Prompt whether electrophysiology recording is active.

        Returns:
            True when ephys is active, otherwise False.
        """
        ephys_raw = input("Ephys active? [y/N]:  ")
        return _is_affirmative(ephys_raw)

    def confirm_meta(self, session_data):
        animal_map = self.system.animal_map

        while True:
            raw = input("Enter the correct animal/phase to use for this session:  ").strip().upper()
            if not raw:
                continue

            if raw.isdigit():
                if raw in _valid_phases():
                    session_data.meta['phase'] = raw
                    return True

                print("Please enter a valid phase")
                continue

            if raw.isalnum():
                if self.system.animal_exists(raw, animal_map):
                    session_data.meta['animal'] = raw
                    session_data.meta['workbook_id'] = self.system.get_workbook_id(raw, animal_map)
                    return True

                print("Please enter a valid animal")
                continue

            print("Please enter a valid animal or phase")

    def confirm_save(self):
        """Prompt whether to save the completed session.

        Returns:
            True when the trainer accepts saving, otherwise False.
        """
        save_choice = input('\nSave current session? [Y/n]:  ').strip().lower()
        _cmd_run('echo.')

        return save_choice in {"", "y", "yes"}


def _is_affirmative(text):
    """Interpret a user-entered yes/no response.

    Args:
        text: Raw response text.

    Returns:
        True for y/yes responses, otherwise False.
    """
    return str(text).strip().lower() in {"y", "yes"}
