"""
Email notifications: sends an end-of-session summary over SMTP using
credentials pulled from the system interface
"""

import json
import html
import smtplib
from datetime import datetime
from email.message import EmailMessage

from behavior_rig.hardware.system import InterfaceObject, SystemInterface


class EmailInterface(InterfaceObject):
    interface_name = "email"

    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587

    def __init__(self, system=None):
        self.system = system or SystemInterface()

    def send_session_summary(self, session_data):
        """
        Email a summary of the completed session.

        Args:
            session_data: SessionData instance containing event and timing data.
        """
        def format_subject(animal, phase):
            """Build the email subject from animal and phase identifiers.

            Args:
                animal: Animal identifier.
                phase: Training phase identifier.

            Returns:
                Formatted subject string.
            """
            animal_str = f'Animal {animal}'
            phase_str = f'Phase {phase}'
            return f'{animal_str}  |  {phase_str}'

        def format_body(date, t_start, t_stop, dur_s, evt):
            """Build the plain-text session summary email body.

            Args:
                date: Session date string in MM/DD/YYYY format.
                t_start: Human-readable start time.
                t_stop: Human-readable stop time.
                dur_s: Session duration in seconds.
                evt: Event data dictionary with values.

            Returns:
                Formatted email body string.
            """
            date_str = datetime.strptime(date, '%m/%d/%Y').strftime('%b-%d')

            m, s = divmod(int(dur_s or 0), 60)
            t_elapsed = f"{m}m {s}s"

            n_hits = sum(1 for e in evt['values'] if e == 'hit')
            n_total = sum(1 for e in evt['values'] if e == 'cue')
            hit_rate = ((n_hits / n_total) * 100) if n_total else 0.0

            lines = [
                ("Date", date_str),
                ("", ""),
                ("Started", str(t_start)),
                ("Finished", str(t_stop)),
                ("Duration", str(t_elapsed)),
                ("", ""),
                ("Total Trials", str(n_total)),
                ("Success Rate", f"{hit_rate:.1f}%"),
            ]

            out = []
            for label, value in lines:
                if not label and not value:
                    out.append("")
                else:
                    out.append(f"{label:<13}{value:>13}")

            return "\n".join(out)

        def ms_to_12h(ms):
            """Convert milliseconds since midnight to a 12-hour clock string.

            Args:
                ms: Milliseconds since midnight.

            Returns:
                Time string formatted as H:MM AM/PM.
            """
            ms = int(ms)
            total_s = ms // 1000
            h24 = (total_s // 3600) % 24
            m = (total_s % 3600) // 60

            am_pm = "AM" if h24 < 12 else "PM"
            h12 = h24 % 12
            if h12 == 0:
                h12 = 12

            return f'{h12}:{m:02d} {am_pm}'

        smtp_username = self.system.require("SMTP_USERNAME")
        smtp_password = self.system.require("SMTP_PASSWORD")
        smtp_to_addr = self.system.require("SMTP_TO_ADDR")

        date = session_data.meta['date']
        animal = session_data.meta['animal']
        phase = session_data.meta['phase']

        subject = format_subject(animal, phase)

        start_ms = session_data.meta.get('t_start')
        stop_ms = session_data.meta.get('t_stop')

        t_start = ms_to_12h(start_ms) if start_ms is not None else "?"
        t_stop = ms_to_12h(stop_ms) if stop_ms is not None else "?"
        dur_s = session_data.meta.get('duration_sec', 0)
        evt = session_data.evt

        body = format_body(date, t_start, t_stop, dur_s, evt)

        try:
            recipients = json.loads(smtp_to_addr)
            if isinstance(recipients, str):
                recipients = [recipients]
        except Exception:
            recipients = [r.strip() for r in smtp_to_addr.split(",") if r.strip()]

        to_addr = ", ".join(recipients)

        msg = EmailMessage()
        msg['From'] = smtp_username
        msg['To'] = to_addr
        msg['Subject'] = subject

        msg.set_content(body)
        msg.add_alternative(
            f"<pre style=\"font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;\">"
            f"{html.escape(body)}"
            f"</pre>",
            subtype="html",
        )

        with smtplib.SMTP(self.SMTP_SERVER, self.SMTP_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
