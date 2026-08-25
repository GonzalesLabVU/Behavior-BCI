"""
Session lifecycle orchestration: builds hardware/session state in
setup(), then drives the main event loop and crash-safe cleanup
"""

import time
from pathlib import Path
from collections import deque
from queue import Empty

from behavior_rig.config import EARLY_STRING, DATA_DIR, PHASE_CONFIG, EVT_QUEUE, ENC_QUEUE
from behavior_rig.interfaces import BehaviorInterfaces
from behavior_rig.session import SessionData, SessionWriter, _get_easy, _update_easy_rate
from behavior_rig.timeutil import _get_date, _ts_to_ms, _get_ts
from behavior_rig.ui.console import _cmd_run
from behavior_rig.hardware.cursor_task import ABORT_EVT


def _cleanup(link, msg, timeout_s=30.0):
    """Request early termination, wait for Arduino END, and close the link.

    Args:
        link: ArduinoLink to stop and close.
        msg: Message printed when cleanup completes or times out.
        timeout_s: Maximum seconds to wait for END.
    """
    try:
        try:
            link.send(EARLY_STRING)
        except Exception:
            pass

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                typ, _, _ = link.msg_q.get(timeout=0.05)

                if typ == "END":
                    print(f'{msg}', flush=True)
                    return
            except Empty:
                pass
        
        print(f'{msg}', flush=True)
    finally:
        link.close()


def setup(interfaces=None):
    """
    Initialize hardware, prompts, and session state before running.

    Args:
        interfaces: Optional BehaviorInterfaces instance for runtime boundaries.

    Returns:
        Tuple of ArduinoLink, SessionData, cursor object, and Prairie client.
    """
    interfaces = interfaces or BehaviorInterfaces()

    system_proxy = interfaces.system
    user_proxy = interfaces.user
    prairie_proxy = interfaces.prairie
    cursor_proxy = interfaces.cursor
    dashboard_proxy = interfaces.dashboard

    ser = None
    link = None
    client = None
    cursor = None

    animal_id = "DEV"
    phase_id = "3"

    try:
        ser, arduino_found = system_proxy.serial_connect()
        link = system_proxy.get_arduino(ser, exceptions=interfaces.exceptions)

        flush_active = user_proxy.prompt_flush()
        if link.active:
            link.send_flush(flush_active)

        animal_id, animal_map = user_proxy.prompt_animal()
        workbook_id = system_proxy.get_workbook_id(animal_id, animal_map)

        if animal_id == "DEV":
            link.verbose = True

        phase_id = user_proxy.prompt_phase()

        if not arduino_found:
            raise RuntimeError(f'No Arduino detected (required for phase {phase_id})')

        side_override = user_proxy.prompt_side() if phase_id == "4" else None
        imaging_active = user_proxy.prompt_imaging()
        ephys_active = user_proxy.prompt_ephys()

        print('\nInitializing resources...', flush=True)

        settings = system_proxy.get_config(phase_id)
        cfg = settings['cfg']
        if side_override:
            settings['side'] = side_override
        side = settings['side']
            
        link.send_config(phase_id, settings)
        link.send_ephys(ephys_active)
        
        client = prairie_proxy.connect(imaging_active)

        is_easy = True
        if cfg and link.active:
            cursor, is_easy = cursor_proxy.connect(phase_id, side)
        
        if int(phase_id) > 1:
            try:
                link.send_and_wait(f"1 {'1' if is_easy else '0'}")
            except Exception as e:
                raise RuntimeError(f'[ERROR] Failed during initial trial config handshake: {e}') from e
        
        session_data = SessionData(animal_id, str(phase_id), _get_date())

        data_subdir = "dev" if animal_id == "DEV" else Path("sessions") / "pending"
        writer = SessionWriter(DATA_DIR / data_subdir)

        try:
            writer.open(session_data)
        except Exception as e:
            raise RuntimeError(f"Could not open local session record: {e}") from e

        session_data.meta['workbook_id'] = workbook_id
        session_data.meta['imaging_active'] = bool(client is not None)
        session_data.meta['ephys_active'] = bool(ephys_active)
        session_data.meta['side_override'] = side_override

        session_data.log_trial_config(trial_n=1, type=is_easy, side=side)

        if ephys_active:
            link.start_ephys()

        print('Running session...\n', flush=True)
        link.send_start()

        dashboard_proxy.notify_start(session_data)

        return link, session_data, cursor, client, writer
    except Exception as e:
        interfaces.exceptions.cache(e, 'setup')

        if link is not None:
            try:
                link.close()
            except Exception as e2:
                interfaces.exceptions.cache(e2, 'setup._cleanup')
        
        if ser is not None:
            try:
                ser.close()
            except Exception as e2:
                interfaces.exceptions.cache(e2, 'setup._cleanup')
        
        raise


def main(link, session_data, cursor, client=None, writer=None, interfaces=None):
    """
    Run the behavioral session event loop.

    Args:
        link: ArduinoLink providing serial messages and commands.
        session_data: SessionData instance to populate.
        cursor: Cursor task object, or None.
        client: Optional PrairieClient for imaging control.
        interfaces: Optional BehaviorInterfaces instance.
    """
    interfaces = interfaces or BehaviorInterfaces()

    console_proxy = interfaces.console
    dashboard_proxy = interfaces.dashboard
    cursor_proxy = interfaces.cursor

    do_calibration = int(session_data.meta['phase']) > 4
    imaging_active = (bool(session_data.meta.get('imaging_active', False))
                      and (client is not None))
    ephys_active = bool(session_data.meta.get('ephys_active', False))

    K = 5
    N = 20
    trial_n = 0
    phase_id = str(session_data.meta['phase'])

    trial_stack = []
    calibrated = not do_calibration
    last_outcome = None

    trial_start_ms = None
    trial_dt = 0.0
    recent_outcomes = deque()

    def _get_msg(timeout_s=0.05):
        """
        Read one message from the Arduino queue.

        Args:
            timeout_s: Maximum seconds to wait for a message.

        Returns:
            Tuple of message type, timestamp, and payload.
        """
        typ, ts, payload = link.msg_q.get(timeout=timeout_s)

        return typ, ts, payload

    started = False
    first_trial = True

    try:
        while link.ser and link.ser.is_open:
            if cursor is not None and ABORT_EVT.is_set():
                raise KeyboardInterrupt

            try:
                typ, ts, payload = _get_msg(timeout_s=0.05)
            except Empty:
                continue

            if not started:
                started = True
                session_data.meta["t_start"] = _ts_to_ms(ts)

                console_proxy.show_start()
                console_proxy.show_header()

            if typ == "ERR":
                if isinstance(payload, BaseException):
                    _cmd_run('echo.')
                    raise payload
                
                raise RuntimeError(f"\nArduinoLink reader error: {payload!r}")

            if typ == "END":
                if ephys_active:
                    session_data.meta['_ephys_stopped'] = True

                break

            if typ == "RAW":
                session_data.add_raw_cap(ts, payload)

            if typ == "EVT":
                p = str(payload)

                try:
                    EVT_QUEUE.put_nowait((ts, p))
                except Exception:
                    pass

                if p == "cue":
                    session_data.add_evt(ts, p)
                    if writer is not None:
                        writer.append({"type": "evt", "ts": ts, "value": p}, force_fsync=True)

                    if imaging_active and first_trial:
                        client_ok = client.start()
                        ttl_ok = link.start_imaging(delay_s=0.0)

                        if not (client_ok and ttl_ok):
                            raise RuntimeError('Initial START command failed')

                        first_trial = False
                    
                    trial_n += 1
                    last_outcome = None
                    trial_start_ms = _ts_to_ms(ts)
                
                if p == 'r_cue':
                    session_data.add_evt(ts, p)
                    if writer is not None:
                        writer.append({"type": "evt", "ts": ts, "value": p}, force_fsync=True)
                
                if p in {'hit', 'miss'}:
                    if last_outcome == p:
                        continue

                    last_outcome = p

                    session_data.add_evt(ts, p)
                    if writer is not None:
                        writer.append({"type": "evt", "ts": ts, "value": p}, force_fsync=True)

                    if imaging_active:
                        client_stop_ok = client.stop_after(delay_s=1.0)
                        ttl_stop_ok = link.stop_imaging(delay_s=1.0)

                        client_start_ok = client.start_after(delay_s=3.0)
                        ttl_start_ok = link.start_imaging(delay_s=3.0)

                        if not (client_stop_ok and ttl_stop_ok and client_start_ok and ttl_start_ok):
                            raise RuntimeError('Failed to schedule imaging restart')

                    end_ms = _ts_to_ms(ts)
                    if trial_start_ms is None or end_ms is None:
                        trial_dt = 0.0
                    else:
                        trial_dt = max(0.0, (end_ms - trial_start_ms) / 1000.0)

                    recent_outcomes.append(p)

                    n_hit = sum(1 for o in recent_outcomes if o == 'hit')
                    n_miss = sum(1 for o in recent_outcomes if o == 'miss')

                    console_proxy.show_trial_info(trial_dt, n_hit, n_miss, p)

                    if do_calibration:
                        trial_stack.insert(0, p)
                        if len(trial_stack) > N:
                            trial_stack.pop()
                        
                        if not calibrated:
                            if len(trial_stack) >= N:
                                K, N, calibration_hits = _update_easy_rate(session_data, trial_stack)

                                session_data.meta['K2'] = K
                                calibrated = True

                                print(f'\nCalibration finished [hits={calibration_hits}/20, K={K}, N={N}]\n', flush=True)
                                console_proxy.show_header()

                    if int(phase_id) >= 4:
                        if calibrated:
                            # early_exit = _is_early_exit(session_data.evt, trial_n, end_ms)
                            early_exit = False

                            if early_exit:
                                if imaging_active:
                                    client.stop()
                                    time.sleep(1)
                                    client.finish()

                                link.stop_ephys(session_data, safe=True)

                                _cleanup(link, 'Terminated by early exit')
                                break

                        next_trial_n = trial_n + 1
                        next_easy = _get_easy(int(phase_id), next_trial_n, K)
                        next_side = session_data.meta.get('side_override') or PHASE_CONFIG[phase_id]['side']

                        time.sleep(0.05)
                        link.send_and_wait(f'{next_trial_n} {"1" if next_easy else "0"}')
                        session_data.log_trial_config(next_trial_n, next_easy, next_side)

                        if cursor is not None:
                            cursor_proxy.update_trial(cursor, next_easy, next_side)

                if p in {"hit", "lick"}:
                    session_data.add_raw_evt(ts, p)

            if typ == "ENC":
                p = str(payload)
                
                try:
                    pos = float(p)

                    session_data.add_enc(ts, str(pos))
                    if writer is not None:
                        writer.append({"type": "enc", "ts": ts, "value": pos})

                    try:
                        ENC_QUEUE.put_nowait(("WHEEL", pos))
                    except Exception:
                        pass
                except Exception:
                    pass
    except KeyboardInterrupt:
        session_data.meta["aborted"] = True

        link.stop_ephys(session_data, safe=True)
        _cleanup(link, "\nTerminated by KeyboardInterrupt")
        raise
    except Exception as e:
        interfaces.exceptions.cache(e, 'main')
    finally:
        if session_data.meta["t_start"] is None:
            session_data.meta["t_start"] = _ts_to_ms(_get_ts())

        session_data.meta["t_stop"] = _ts_to_ms(_get_ts())

        t0 = session_data.meta['t_start']
        t1 = session_data.meta['t_stop']
        dt = 0 if (t0 is None or t1 is None) else max(0, (t1 - t0) // 1000)
        session_data.meta["duration_sec"] = int(dt)

        if writer is not None:
            try:
                writer.finalize(session_data)
            except Exception:
                writer.emergency_dump(session_data)

        link.stop_ephys(session_data, safe=True)

        if client is not None:
            client.stop()

        dashboard_proxy.notify_finish()
