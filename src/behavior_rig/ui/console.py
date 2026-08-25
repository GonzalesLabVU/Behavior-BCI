"""
Terminal-facing operator interface: prompts, confirmations, and console
output shown to the person running a session
"""

import os
import time
import keyboard
import traceback
from datetime import datetime

from behavior_rig.hardware.system import InterfaceObject
from behavior_rig.errors import EXC_STACK


class ConsoleInterface(InterfaceObject):
    interface_name = "console"

    INFO_CFG = {
        "labels": ["TRIAL", "ELAPSED", "SUCCESS", "FAILURE", "RATE"],
        "label_pads": (5, 5, 4, 4, 5),
        "value_pads": (6, 5, None, None, 3)
        }
    BLOCK = "\u2588"

    def clear(self):
        """Clear the console window."""
        _cmd_run('cls')

    def line(self, text="", **kwargs):
        """Print a line of text to the console.

        Args:
            text: Text to print.
            **kwargs: Additional keyword arguments passed to print().
        """
        print(text, **kwargs)

    def warning(self, text):
        """Print a formatted warning message.

        Args:
            text: Warning text to display.
        """
        print(f'[WARNING] {text}', flush=True)

    def show_start(self):
        """Display the session start time."""
        print(f'\nSession started at {datetime.now().strftime("%I:%M %p")}\n', flush=True)

    def show_header(self):
        """Display the trial status table header."""
        labels = self.INFO_CFG["labels"]
        label_pads = self.INFO_CFG["label_pads"]

        cells = [f'{" " * sz}{txt}{" " * sz}'
                 for txt, sz in zip(labels, label_pads)]
        header = "|".join(cells)
        hline = "|".join("—" * len(cell) for cell in cells)

        print(header)
        print(hline)

    def show_trial_info(self, dt, n_hit, n_miss, outcome):
        """Display a single row of trial progress information.

        Args:
            dt: Trial elapsed time in seconds.
            n_hit: Number of recent hit outcomes.
            n_miss: Number of recent miss outcomes.
            outcome: Outcome label for the current trial.
        """
        labels = self.INFO_CFG["labels"]
        label_pads = self.INFO_CFG["label_pads"]
        value_pads = self.INFO_CFG["value_pads"]

        n_total = n_hit + n_miss

        trial_str = n_total
        elapsed_str = f"{(dt - 1.5):.2f} s"

        col_w = [len(label) + (2 * pad)
                 for label, pad in zip(labels, label_pads)]
    
        success_w = col_w[labels.index("SUCCESS")]
        failure_w = col_w[labels.index("FAILURE")]

        success_str = (self.BLOCK * success_w if outcome == "hit"
                       else " " * success_w)
        failure_str = (self.BLOCK * failure_w if outcome == "miss"
                       else " " * failure_w)

        rate = 100.0 * (n_hit / n_total) if n_total else 0.0
        rate_str = f"{rate:.1f} %"

        all_str = [trial_str, elapsed_str, success_str, failure_str, rate_str]
        values = {label: s for label, s in zip(labels, all_str)}

        cells = []
        for i, label in enumerate(labels):
            val = str(values[label])
            total_w = col_w[i]
            rpad = value_pads[i]

            if rpad is None:
                cells.append(val)
                continue

            free_w = max(0, total_w - rpad - len(val))
            cell = (" " * free_w) + val + (" " * rpad)
            cells.append(cell)

        print("|".join(cells))

    def show_summary(self, session_data):
        """Display a brief session duration summary.

        Args:
            session_data: SessionData instance containing metadata.
        """
        if not session_data:
            return
        
        dur = session_data.meta.get('duration_sec')
        if dur is None:
            return
        
        m, s = divmod(int(max(0, dur)), 60)
        print(f"\nSession duration: {m}:{s:02d}\n", flush=True)

    def show_exceptions(self):
        hline = 100 * "—"

        if not EXC_STACK:
            _cmd_run("echo.")
            print(f"{hline}\n")
            print(f"{hline}\n")
            print("[Process exited with code 0]")
            return

        print(hline + "\nEXCEPTION STACK (in order of occurrence):\n" + hline, flush=True)

        for i, info in enumerate(EXC_STACK, start=1):
            print(f"\n[{i}] {info['type']} in {info['caller']}:", flush=True)

            exc = info['exc']
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

            print(f"\n{tb}", flush=True)
            print(hline, flush=True)

        print("\n[Process exited with code 1]\n")

    def wait_for_key(self):
        """Wait for one keyboard press before exiting."""
        print('\nPress any key to continue . . .', end="", flush=True)
        time.sleep(0.25)
        keyboard.read_key()
        _cmd_run('echo.', 'echo.')


def _cmd_run(*args):
    """Run one or more shell commands joined for Windows command execution.

    Args:
        *args: Command strings to execute in sequence.
    """
    cmd = " & ".join(args)
    os.system(cmd)
