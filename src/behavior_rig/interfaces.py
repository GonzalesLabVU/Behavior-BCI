"""
Composition root: builds and wires every interface object once per run,
injecting shared dependencies (system, trainer, exceptions) between them
"""

from behavior_rig.errors import ExceptionInterface
from behavior_rig.hardware.system import SystemInterface
from behavior_rig.ui.trainer import TrainerInterface
from behavior_rig.ui.console import ConsoleInterface
from behavior_rig.notify.dashboard import DashboardInterface
from behavior_rig.notify.email import EmailInterface
from behavior_rig.io.sheets import SaveInterface
from behavior_rig.hardware.prairie import PrairieInterface
from behavior_rig.hardware.cursor import CursorInterface


class BehaviorInterfaces:
    def __init__(self):
        """
        Create the default runtime interfaces used by setup and main.
        """
        self.exceptions = ExceptionInterface()
        self.system = SystemInterface()
        self.user = TrainerInterface(system=self.system)
        self.console = ConsoleInterface()
        self.dashboard = DashboardInterface()
        self.email = EmailInterface(system=self.system)
        self.saving = SaveInterface(trainer=self.user, exceptions=self.exceptions)
        self.prairie = PrairieInterface()
        self.cursor = CursorInterface()
