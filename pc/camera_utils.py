import os
import time
import atexit
import threading
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import PySpin


FRAME_RATE = 30.0
EXPOSURE_TIME = 2000.0
GAIN = 15.0
BINNING_FACTOR = 4

PIXEL_FORMAT = "Mono8"
VIDEO_CODEC = "avc1"
VIDEO_EXTENSION = ".mp4"


def _ts_to_seconds(ts):
    hms, ms = ts.split(".")
    h, m, s = hms.split(":")

    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000


def _get_ts():
    t = time.time()
    base = time.strftime("%H:%M:%S", time.localtime(t))
    ms = int((t - int(t)) * 1000)

    return f"{base}.{ms:03d}"


class FLIRCamera:
    def __init__(self, output_dir):
        # system/camera handles
        self._system = None
        self._cam_list = None
        self._cam = None
        self._nodemap = None
        self._nodemap_tldevice = None

        # configuration state
        self._config = {}
        self._is_configured = False
        self._target_fov = None
        self._frame_rate = None
        self._pixel_format = None

        # per-trial acquisition/threading state
        self._acquisition_thread = None
        self._stop_event = threading.Event()
        self._is_recording = False
        self._lock = threading.Lock()

        self._video_writer = None
        self._output_dir = self._validate_output_dir(output_dir)
        self._current_output_path = None
        self._trial_number = 0
        self._trial_log = {}
        self._record_exc = None

        # session-level bookkeeping
        self._session_files = []

        # initialize camera
        self._initialize_camera()

        # safety net
        atexit.register(self.close)

    def __del__(self):
        try:
            self.close()

        except Exception:
            pass

    def _validate_output_dir(self, output_dir):
        output_dir = os.path.abspath(output_dir)

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        if not os.access(output_dir, os.W_OK):
            raise RuntimeError(f"Output directory {output_dir} is not writable")

        return output_dir

    def _detect_camera(self):
        if self._cam_list.GetSize() == 0:
            raise RuntimeError("No FLIR cameras detected")

        return self._cam_list.GetByIndex(0)

    def _initialize_camera(self):
        self._system = PySpin.System.GetInstance()
        self._cam_list = self._system.GetCameras()

        try:
            self._cam = self._detect_camera()
        except RuntimeError:
            self._cam_list.Clear()
            self._system.ReleaseInstance()

            self._system = None
            self._cam_list = None

            raise

        self._cam.Init()

        self._nodemap = self._cam.GetNodeMap()
        self._nodemap_tldevice = self._cam.GetTLDeviceNodeMap()

    def _warm_up(self):
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        width, height = self._target_fov

        tmp_path = os.path.join(self._output_dir, "_warmup.mp4")
        video = cv2.VideoWriter(tmp_path,
                                cv2.CAP_MSMF,
                                fourcc,
                                FRAME_RATE,
                                (width, height),
                                isColor=False
                                )
        video.release()

        try:
            os.remove(tmp_path)
        except OSError:
            pass

    def _record(self, output_path, trial_key):
        cam = self._cam
        width, height = self._target_fov
        video = None

        self._trial_log[trial_key] = {
            "frameRate": None,
            "frameCount": 0,
            "tStart": None,
            "tStop": None
            }

        try:
            fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
            video = cv2.VideoWriter(output_path,
                                    cv2.CAP_MSMF,
                                    fourcc,
                                    self._frame_rate,
                                    (width, height),
                                    isColor=False
                                    )

            if not video.isOpened():
                raise RuntimeError(f"Could not open video file '{output_path}' for writing")

            self._video_writer = video

            cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
            cam.BeginAcquisition()

            while not self._stop_event.is_set():
                try:
                    img = cam.GetNextImage(1000)
                except PySpin.SpinnakerException as e:
                    if self._stop_event.is_set():
                        break

                    if e.code == PySpin.SPINNAKER_ERR_TIMEOUT:
                        continue

                    raise

                if img.IsIncomplete():
                    img.Release()
                    continue

                try:
                    frame_ts = _get_ts()

                    arr = img.GetNDArray()
                    if arr.dtype != np.uint8:
                        arr = cv2.convertScaleAbs(arr)

                    video.write(arr)

                    trial_entry = self._trial_log[trial_key]
                    if trial_entry['tStart'] is None:
                        trial_entry['tStart'] = frame_ts
                    trial_entry['tStop'] = frame_ts
                    trial_entry['frameCount'] += 1
                finally:
                    img.Release()
        except Exception as e:
            self._record_exc = e
        finally:
            if video is not None:
                try:
                    video.release()
                except Exception:
                    pass

            self._video_writer = None

            try:
                if cam.IsStreaming():
                    cam.EndAcquisition()
            except Exception:
                pass

            trial_entry = self._trial_log[trial_key]
            if trial_entry['frameCount'] > 1 and trial_entry['tStart'] is not None:
                elapsed = _ts_to_seconds(trial_entry['tStop']) - _ts_to_seconds(trial_entry['tStart'])
                if elapsed > 0:
                    trial_entry['frameRate'] = round((trial_entry['frameCount'] - 1) / elapsed, 2)

            with self._lock:
                self._is_recording = False

                if output_path not in self._session_files:
                    self._session_files.append(output_path)

    def _save_trial_log(self):
        root = ET.Element("RecordingData")

        for trial_key, data in self._trial_log.items():
            tag_name = trial_key.replace(" ", "-")
            trial_elem = ET.SubElement(root, tag_name)

            ET.SubElement(trial_elem, "frameRate").text = str(data['frameRate'])
            ET.SubElement(trial_elem, "frameCount").text = str(data['frameCount'])
            ET.SubElement(trial_elem, "tStart").text = str(data['tStart'])
            ET.SubElement(trial_elem, "tStop").text = str(data['tStop'])

        tree = ET.ElementTree(root)
        ET.indent(tree, space="    ")

        xml_path = os.path.join(self._output_dir, "metadata.xml")
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    def configure(self):
        with self._lock:
            if self._is_recording:
                raise RuntimeError("Cannot configure camera while recording")

        cam = self._cam
        node_map = self._nodemap

        # pixel binning
        if BINNING_FACTOR > 1:
            selector_node = PySpin.CEnumerationPtr(node_map.GetNode("BinningSelector"))
            if PySpin.IsAvailable(selector_node) and PySpin.IsWritable(selector_node):
                all_entry = selector_node.GetEntryByName("All")
                if PySpin.IsAvailable(all_entry):
                    selector_node.SetIntValue(all_entry.GetValue())

            for mode_name in ("BinningHorizontalMode", "BinningVerticalMode"):
                mode_node = PySpin.CEnumerationPtr(node_map.GetNode(mode_name))
                if PySpin.IsAvailable(mode_node) and PySpin.IsWritable(mode_node):
                    mode_node.SetIntValue(mode_node.GetEntryByName("Average").GetValue())

            for axis_name in ("BinningHorizontal", "BinningVertical"):
                axis_node = PySpin.CIntegerPtr(node_map.GetNode(axis_name))
                if not (PySpin.IsAvailable(axis_node) and PySpin.IsWritable(axis_node)):
                    raise RuntimeError(f"Camera does not support {axis_name}")

                axis_node.SetValue(BINNING_FACTOR)

        # field of view
        width_node = PySpin.CIntegerPtr(cam.Width)
        height_node = PySpin.CIntegerPtr(cam.Height)
        offset_x_node = PySpin.CIntegerPtr(cam.OffsetX)
        offset_y_node = PySpin.CIntegerPtr(cam.OffsetY)

        offset_x_node.SetValue(0)
        offset_y_node.SetValue(0)

        width_node.SetValue(width_node.GetMax())
        height_node.SetValue(height_node.GetMax())

        self._target_fov = (width_node.GetValue(), height_node.GetValue())

        # exposure
        exp_auto = PySpin.CEnumerationPtr(node_map.GetNode("ExposureAuto"))
        if PySpin.IsAvailable(exp_auto) and PySpin.IsWritable(exp_auto):
            exp_auto.SetIntValue(exp_auto.GetEntryByName("Off").GetValue())

        exp_time = PySpin.CFloatPtr(node_map.GetNode("ExposureTime"))
        exp_time.SetValue(EXPOSURE_TIME)

        # gain
        gain_auto = PySpin.CEnumerationPtr(node_map.GetNode("GainAuto"))
        if PySpin.IsAvailable(gain_auto) and PySpin.IsWritable(gain_auto):
            gain_auto.SetIntValue(gain_auto.GetEntryByName("Off").GetValue())

        gain_node = PySpin.CFloatPtr(node_map.GetNode("Gain"))
        gain_node.SetValue(GAIN)

        # pixel format
        px_fmt = PySpin.CEnumerationPtr(node_map.GetNode("PixelFormat"))
        if PySpin.IsAvailable(px_fmt) and PySpin.IsWritable(px_fmt):
            px_fmt.SetIntValue(px_fmt.GetEntryByName(PIXEL_FORMAT).GetValue())

        self._pixel_format = PIXEL_FORMAT

        # USB bandwidth
        limit_node = PySpin.CIntegerPtr(node_map.GetNode("DeviceLinkThroughputLimit"))
        if PySpin.IsAvailable(limit_node) and PySpin.IsWritable(limit_node):
            limit_node.SetValue(limit_node.GetMax())

        # frame rate
        frame_rate_enable = PySpin.CBooleanPtr(node_map.GetNode("AcquisitionFrameRateEnable"))
        if PySpin.IsAvailable(frame_rate_enable) and PySpin.IsWritable(frame_rate_enable):
            frame_rate_enable.SetValue(True)

        fps_node = PySpin.CFloatPtr(node_map.GetNode("AcquisitionFrameRate"))
        fps_node.SetValue(FRAME_RATE)

        self._frame_rate = FRAME_RATE

        # save configuration settings
        self._config = {
            "fov": self._target_fov,
            "frame_rate": FRAME_RATE,
            "exposure_time": EXPOSURE_TIME,
            "gain": GAIN,
            "pixel_format": PIXEL_FORMAT,
            }
        self._is_configured = True

        # initialize video encoder
        self._warm_up()

    def start(self):
        if not self._is_configured:
            raise RuntimeError("Camera must be configured before recording")

        with self._lock:
            if self._is_recording:
                raise RuntimeError("A recording is already active")

            if self._acquisition_thread is not None and self._acquisition_thread.is_alive():
                raise RuntimeError("Previous acquisition thread is still running")

            self._trial_number += 1
            trial_key = f"Trial {self._trial_number}"
            output_path = os.path.join(self._output_dir, f"Trial {self._trial_number}{VIDEO_EXTENSION}")

            self._stop_event.clear()
            self._current_output_path = output_path
            self._is_recording = True

            self._acquisition_thread = threading.Thread(target=self._record,
                                                        args=(output_path, trial_key),
                                                        daemon=True)
            self._acquisition_thread.start()

    def stop(self, timeout=5.0):
        with self._lock:
            if not self._is_recording:
                return

            self._stop_event.set()

            acquisition_thread = (self._acquisition_thread)

        if acquisition_thread is not None and acquisition_thread.is_alive():
            acquisition_thread.join(timeout=timeout)

        if acquisition_thread is not None and not acquisition_thread.is_alive():
            self._acquisition_thread = None
        elif acquisition_thread is None:
            self._acquisition_thread = None

        exc = self._record_exc
        self._record_exc = None

        if exc is not None:
            raise exc

    def close(self):
        stop_exc = None

        if self._is_recording:
            try:
                self.stop()
            except Exception as e:
                stop_exc = e

        save_exc = None

        try:
            self._save_trial_log()
        except Exception as e:
            save_exc = e

        if self._cam is not None:
            try:
                if self._cam.IsStreaming():
                    self._cam.EndAcquisition()
            except Exception:
                pass

            try:
                self._cam.DeInit()
            except Exception:
                pass

            self._cam = None

        if self._cam_list is not None:
            try:
                self._cam_list.Clear()
            except Exception:
                pass

            self._cam_list = None

        if self._system is not None:
            try:
                self._system.ReleaseInstance()
            except Exception:
                pass

            self._system = None

        if stop_exc is not None and save_exc is not None:
            raise ExceptionGroup("Multiple errors encountered", [stop_exc, save_exc])
        if stop_exc is not None:
            raise stop_exc
        if save_exc is not None:
            raise save_exc


if __name__ == "__main__":
    cam = FLIRCamera(output_dir=r"C:\Users\Max\Documents\Classes\Gonzales Lab\Behavioral\Local\webapp\backend\camera-dev")
    cam.configure()

    n = 0

    # while n < 3:
    #     n += 1

    #     print(f"Starting capture {n}...", end="")
    #     cam.start()
    #     time.sleep(3)
    #     cam.stop()
    #     print("Done")
    #     time.sleep(1)

    capture_mins = 10

    print(f"Starting {capture_mins} minute capture...")
    cam.start()
    time.sleep(capture_mins*60)
    cam.stop()

    cam.close()