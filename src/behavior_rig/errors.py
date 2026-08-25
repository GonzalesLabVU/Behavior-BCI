"""
Centralized exception capture and reporting: buffers exceptions during a
session and commits them to the error log / GitHub issue on exit
"""

import os
import shutil
import subprocess
import tempfile
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path

from behavior_rig.config import ERROR_LOG_PATH
from behavior_rig.hardware.system import InterfaceObject


REPO_SLUG = "GonzalesLabVU/Behavior-BCI"
REPO_BRANCH = "main"
REPO_REL_PATH = Path("pc") / "config" / "errors.log"

ERROR_LOGGED = False
LOG_COMMIT_FAIL = False

EXC_STACK: "deque[dict[str, object]]" = deque()


class ExceptionInterface(InterfaceObject):
    interface_name = "exception"

    def __init__(self, animal_id="UNKNOWN", phase_id="0"):
        """
        Initialize exception logging context

        Args:
            animal_id: Animal identifier for log records
            phase_id: Phase identifier for log records
        """
        self.animal_id = animal_id
        self.phase_id = phase_id

    def set_session(self, animal_id, phase_id):
        """
        Update the animal and phase used for exception logs

        Args:
            animal_id: Animal identifier for log records
            phase_id: Phase identifier for log records
        """
        self.animal_id = animal_id
        self.phase_id = phase_id

    def cache(self, exc, caller):
        """
        Cache an exception for console display

        Args:
            exc: Exception instance to cache
            caller: Name of the caller where the exception occurred
        """
        EXC_STACK.append({
            "type": type(exc).__name__,
            "caller": caller,
            "exc": exc
            })

    def log(self, exc):
        """
        Append an exception to the local error log

        Args:
            exc: Exception or error value to log
        """
        global ERROR_LOGGED
        ERROR_LOGGED = True

        try:
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")

            client = os.getenv("CLIENT_ID", "UNKNOWN_CLIENT")
            level = "UNKNOWN"

            if isinstance(exc, BaseException) and exc.__traceback__ is not None:
                tb = exc.__traceback__
                while tb.tb_next:
                    tb = tb.tb_next

                level = tb.tb_frame.f_code.co_name

            animal = str(self.animal_id)
            phase = str(self.phase_id)

            header = [
                f"TIMESTAMP={date_str} {time_str}",
                f"LEVEL={level}",
                f"CLIENT={client}",
                f"ANIMAL={animal}",
                f"PHASE={phase}"
                ]
            hline = ["-" * 40]
            body = []

            if isinstance(exc, BaseException):
                tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)

                for line in "".join(tb_lines).rstrip('\n').splitlines():
                    body.append(f"  {line}")
            else:
                body.append(f"  {type(exc).__name__}: {exc!r}")

            with open(ERROR_LOG_PATH, 'a', encoding='utf-8') as f:
                for line in hline + header + hline + body:
                    f.write(line + '\n')

                f.write('\n')
        except Exception:
            pass

    def commit(self):
        """
        Commit the local error log to the remote repository

        Returns:
            True when a commit and push occurred, otherwise False
        """
        global LOG_COMMIT_FAIL

        if not ERROR_LOGGED:
            return False

        token = os.getenv("GITHUB_TOKEN")
        if not token:
            print("[WARNING] GITHUB_TOKEN not set (skipping errors.log push)", flush=True)
            return False

        if not ERROR_LOG_PATH.exists():
            return False

        remote_url = f"https://x-access-token:{token}@github.com/{REPO_SLUG}.git"

        try:
            with tempfile.TemporaryDirectory(prefix='behavior_bci_repo_') as td:
                repo_dir = Path(td) / "repo"

                _git_run(['git', 'clone', '--depth', '1', '--branch', REPO_BRANCH, remote_url, str(repo_dir)])

                dest_path = repo_dir / REPO_REL_PATH
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                shutil.copy2(ERROR_LOG_PATH, dest_path)

                st = _git_run(['git', 'status', '--porcelain', str(REPO_REL_PATH)], cwd=repo_dir).stdout.strip()
                if not st:
                    return False

                _git_run(['git', 'config', 'user.name', 'behavior-bci-bot'], cwd=repo_dir)
                _git_run(['git', 'config', 'user.email', 'behavior-bci-bot@users.noreply.github.com'], cwd=repo_dir)

                _git_run(['git', 'add', str(REPO_REL_PATH)], cwd=repo_dir)

                msg = f"Update errors.log (animal={self.animal_id}, phase={self.phase_id})"
                c = subprocess.run(
                    ['git', 'commit', '-m', msg],
                    cwd=repo_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                    )

                if c.returncode != 0:
                    return False

                _git_run(['git', 'push', 'origin', REPO_BRANCH], cwd=repo_dir, check=True)

                return True
        except Exception as e:
            if not LOG_COMMIT_FAIL:
                LOG_COMMIT_FAIL = True
                print(f"[WARNING] Failed to commit errors.log: {type(e).__name__}", flush=True)

            return False

    def log_and_commit(self, exc):
        """
        Log an exception and attempt to commit the error log

        Args:
            exc: Exception or error value to log

        Returns:
            Result of the commit attempt when applicable
        """
        if isinstance(exc, KeyboardInterrupt):
            return
        
        try:
            self.log(exc)
        finally:
            try:
                self.commit()
            except Exception:
                pass


def _git_run(cmd, cwd=None, check=True):
    """
    Execute a git CLI command
    """
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
        )
