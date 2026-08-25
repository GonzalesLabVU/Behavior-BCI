"""
Command-line entry point: parses arguments, runs a full session via
runner.py, and reports the outcome (saved, aborted, or failed) to the user
"""

import os
import warnings

from behavior_rig.interfaces import BehaviorInterfaces
from behavior_rig.runner import setup, main as run_main
from behavior_rig.errors import ExceptionInterface


def main():
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    warnings.filterwarnings("ignore",
                            category=UserWarning,
                            message="pkg_resources is deprecated as an API.*")

    interfaces = BehaviorInterfaces()
    interfaces.console.clear()

    link = None
    session_data = None
    cursor = None
    prairie = None

    animal_id_for_log = "UNKNOWN"
    phase_id_for_log = "0"


    try:
        link, session_data, cursor, prairie, writer = setup(interfaces)

        if session_data is not None:
            animal_id_for_log = session_data.meta.get("animal", "UNKNOWN")
            phase_id_for_log = session_data.meta.get("phase", "0")

        run_main(link, session_data, cursor, prairie, writer, interfaces)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass
    except BaseException as e:
        interfaces.exceptions.cache(e, '__main__')
        interfaces.console.show_exceptions()
    finally:
        interfaces.console.show_summary(session_data)
        run_info = (animal_id_for_log, phase_id_for_log)

        if prairie is not None:
            try:
                interfaces.prairie.finish(prairie, session_data)
            except Exception as e:
                interfaces.exceptions.cache(e, "__main__.prairie_finish")
                ExceptionInterface(*run_info).log_and_commit(e)

        if cursor is not None:
            try:
                interfaces.cursor.stop(cursor)
            except Exception as e:
                interfaces.exceptions.cache(e, "__main__.cursor_stop")
                ExceptionInterface(*run_info).log_and_commit(e)
        
        if link is not None:
            try:
                link.close()
            except Exception as e:
                interfaces.exceptions.cache(e, '__main__.link_close')
                ExceptionInterface(*run_info).log_and_commit(e)

        if session_data is not None and session_data.is_finished:
            if session_data.meta.get('animal', None) not in {None, "DEV"}:
                try:
                    interfaces.email.send_session_summary(session_data)
                except Exception as e:
                    interfaces.exceptions.cache(e, '__main__.send_email')
                    ExceptionInterface(*run_info).log_and_commit(e)
        
        if session_data is not None and session_data.any_data():
            if session_data.meta.get('animal', None) not in {None, "DEV"}:
                try:
                    if interfaces.user.confirm_save():
                        if interfaces.saving.resolve_protocol(session_data):
                            ok = interfaces.saving.save_session(session_data)
                            if not ok:
                                interfaces.console.warning("Google Sheets save failed (local record already saved)")
                        else:
                            print("Session exited a Google Sheets upload (local record already saved)", flush=True)
                except Exception as e:
                    interfaces.exceptions.cache(e, '__main__.safe_save')

        interfaces.console.show_exceptions()
        interfaces.console.wait_for_key()


if __name__ == "__main__":
    main()
