import argparse
import itertools
import json
import math
import os
import subprocess
import threading
import time
import traceback

import numpy as np
import scipy.ndimage

import cv2  # isort:ignore

parser = argparse.ArgumentParser()
parser.add_argument("--output-file", default=None, type=str)
parser.add_argument("--input-file", default=None, type=str)


def zone_argmax(m, zone):
    try:
        ind = np.argmax(m[zone[0][1]:zone[1][1], zone[0][0]:zone[1][0]])
    except ValueError as e:
        print(zone)
        raise e
    ind = np.unravel_index(ind, (zone[1][1] - zone[0][1], zone[1][0] - zone[0][0]))
    ind += np.array([zone[0][1], zone[0][0]])
    value = m[ind[0], ind[1]]
    return value, ind


def diff_image(background, img, filter_radius=20):
    diff = np.abs(img[:, :, 0].astype(np.int32) - background[:, :, 0].astype(np.int32)).astype(
        np.uint8)
    # diff = np.sqrt(np.sum(np.square(img.astype(np.int32) - background.astype(np.int32)),
    #                       axis=2)).astype(np.uint8)
    diff = scipy.ndimage.median_filter(diff, filter_radius)
    return diff


drag_zone = []
detection_zone = []
dragging = False
drag_lock = threading.Lock()
drag_start_time = None
min_drag_duration = 2.0
downscale_factor = 2


def handle_drag(event, x, y, *_):
    global detection_zone, drag_zone, dragging, drag_lock
    with drag_lock:
        if event == cv2.EVENT_LBUTTONDOWN:
            drag_zone = [(x, y)]
            dragging = True

        elif event == cv2.EVENT_LBUTTONUP:
            x1, y1 = drag_zone[0]
            x2, y2 = x, y
            x1, x2, y1, y2 = (int(x1 / downscale_factor), int(x2 / downscale_factor), int(
                y1 / downscale_factor), int(y2 / downscale_factor))
            drag_zone = [(min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2))]
            dragging = False
            print(drag_zone)
            if abs(x1 - x2) > 0 or abs(y1 - y2) > 0:
                detection_zone = drag_zone


class Settings:
    def __init__(self, window_name):
        self._window_name = window_name
        self._setting_names = []
        self.add('Threshold', 90, 255)
        self.add('On', 0, 1)
        self.add('RepeatDelay', 5, 20)
        self.add('RepeatInterval', 2, 20)
        self.add('DownVsPageDown', 0, 1)
        self.add('KeyRepeat', 1, 20)
        self.detection_zone = [(0, 60), (160, 120)]

    def add(self, name, default, max_value):
        self._setting_names.append(name)
        cv2.createTrackbar(name, self._window_name, default, max_value, lambda x: None)

    def get(self, name):
        if name not in self._setting_names:
            raise ValueError("Unknown setting name {}".format(name))
        return cv2.getTrackbarPos(name, self._window_name)

    def set(self, name, value):
        if name not in self._setting_names:
            raise ValueError("Unknown setting name {}".format(name))
        cv2.setTrackbarPos(name, self._window_name, value)

    def load(self, filename):
        try:
            with open(filename, 'r') as f:
                d = json.load(f)
                for name, value in d.items():
                    if name == "detection_zone":
                        self.detection_zone = detection_zone
                    self.set(name, value)
        except json.decoder.JSONDecodeError:
            traceback.print_exc()

    def save(self, filename):
        with open(filename, 'w') as f:
            d = {name: self.get(name) for name in self._setting_names}
            d["detection_zone"] = self.detection_zone
            json.dump(d, f)


if __name__ == "__main__":
    args = parser.parse_args()

    detection_zone = [(0, 60), (160, 120)]
    default_framerate = 30
    default_font = cv2.FONT_HERSHEY_SIMPLEX
    wait_duration = 1
    filter_radius = 10
    repeat_initial_delay = 8
    downscale_factor = 2
    max_active_keypresses = 20

    frameskip = 1
    if args.input_file is not None:
        cam = cv2.VideoCapture(args.input_file)
        wait_duration = int(math.floor(1000.0 / cam.get(cv2.CAP_PROP_FPS)))
    else:
        cam = cv2.VideoCapture(1)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
        frameskip = 5

    fgbg = cv2.createBackgroundSubtractorMOG2()

    cv2.namedWindow("Video")
    cv2.setMouseCallback("Video", handle_drag)
    settings = Settings("Video")
    settings_filename = "settings.json"
    if os.path.exists(settings_filename):
        settings.load(settings_filename)

    writer = None

    active_start_time = None
    last_keypress_time = time.time()
    active_keypresses = 0
    key_press_interval = 1.0
    for i in itertools.count():
        # Hit 'q' on the keyboard to quit!
        key = cv2.waitKey(wait_duration) & 0xFF
        if key == ord('s'):
            settings.save(settings_filename)
        if key == ord('q'):
            break
        elif key == ord('r'):
            fgbg = cv2.createBackgroundSubtractorMOG2()

        b = cam.grab()
        if b and i % frameskip == 0:
            retval, raw_img = cam.retrieve()
            if (raw_img is None and args.output_file is None and args.input_file is not None):
                # Loop to beginning
                cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
                fgbg = cv2.createBackgroundSubtractorMOG2()
                continue
            downscaled_img = raw_img[::downscale_factor, ::downscale_factor, :]
            downscaled_img = scipy.ndimage.filters.gaussian_filter(downscaled_img, 1.0)
            img = cv2.cvtColor(downscaled_img, cv2.COLOR_BGR2HSV)
            background = fgbg.getBackgroundImage()
            if background is not None:
                diff = diff_image(background, img, filter_radius)
                diff_bgr = np.repeat(diff[:, :, np.newaxis], 3, axis=2)
                annotated_img = np.concatenate(
                    (cv2.cvtColor(img, cv2.COLOR_HSV2BGR), cv2.cvtColor(
                        background, cv2.COLOR_HSV2BGR), diff_bgr),
                    axis=0)
                with drag_lock:
                    diffmax, ind = zone_argmax(diff, detection_zone)
                    cv2.rectangle(annotated_img, detection_zone[0], detection_zone[1], (0, 255, 0),
                                  2)
                    cv2.circle(annotated_img, (ind[1], ind[0]), filter_radius, (255, 0, 0))
                    cv2.putText(annotated_img, "{}".format(diffmax), (0, 20), default_font, 0.5,
                                (255, 255, 255), 1, cv2.LINE_AA)
                    if (diffmax > settings.get("Threshold") and settings.get("On") == 1):
                        repeat_delay = settings.get("RepeatDelay") * 0.1
                        repeat_interval = settings.get("RepeatInterval") * 0.1
                        key_repeat = max(settings.get("KeyRepeat"), 1)
                        t = time.time()
                        press = False
                        if active_start_time is None:
                            if t - last_keypress_time > repeat_interval:
                                active_start_time = t
                                active_keypresses = 0
                                last_keypress_time = None
                                press = True
                        elif last_keypress_time is None and t - active_start_time > repeat_delay:
                            press = True
                        elif last_keypress_time is not None and t - last_keypress_time > repeat_interval:
                            press = True
                        if active_start_time is not None:
                            print(t - active_start_time, press)
                        if press:
                            if active_keypresses < max_active_keypresses:
                                key = "Down"
                                if settings.get("DownVsPageDown") == 1:
                                    key = "Page_Down"
                                subprocess.run(["xdotool", "key"] + [key] * key_repeat)
                                last_keypress_time = t
                                active_keypresses += 1
                    else:
                        active_start_time = None

                cv2.imshow('Video',
                           cv2.resize(
                               annotated_img, None, fx=downscale_factor, fy=downscale_factor))

            fgmask = fgbg.apply(img)

            if not args.output_file is None:
                if writer is None:
                    (h, w) = img.shape[:2]
                    fourcc_out = cv2.VideoWriter_fourcc(* 'MJPG')
                    writer = cv2.VideoWriter(args.output_file, fourcc_out,
                                             default_framerate / frameskip, (w, h), True)
                writer.write(raw_img)

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
