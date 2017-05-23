import argparse
import itertools
import math
import subprocess

import numpy as np
import scipy.ndimage
import time
import threading

import cv2  # isort:ignore

parser = argparse.ArgumentParser()
parser.add_argument("--output-file", default=None, type=str)
parser.add_argument("--input-file", default=None, type=str)

dz = []
dragging = False
drag_lock = threading.Lock()


def zone_argmax(m, zone):
    ind = np.argmax(m[zone[0][1]:zone[1][1], zone[0][0]:zone[1][0]])
    ind = np.unravel_index(ind, (zone[1][1] - zone[0][1], zone[1][0] - zone[0][0]))
    ind += np.array([zone[0][1], zone[0][0]])
    value = m[ind[0], ind[1]]
    return value, ind


def diff_image(background, img, filter_radius=20):
    diff = np.sqrt(np.sum(np.square(img.astype(np.int32) - background.astype(np.int32)),
                          axis=2)).astype(np.uint8)
    diff = scipy.ndimage.median_filter(diff, filter_radius)
    return diff


def handle_drag(event, x, y, *_):
    global dz, dragging, drag_lock
    with drag_lock:
        if event == cv2.EVENT_LBUTTONDOWN:
            dz = [(x, y)]
            dragging = True

        elif event == cv2.EVENT_LBUTTONUP:
            dz.append((x, y))
            dragging = False
            print(dz)


if __name__ == "__main__":
    args = parser.parse_args()

    dz = [(0, 40), (160, 120)]
    default_framerate = 30
    default_font = cv2.FONT_HERSHEY_SIMPLEX
    wait_duration = 1
    filter_radius = 10
    repeat_initial_delay = 8
    downscale_factor = 2

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
    cv2.createTrackbar('Threshold', 'Video', 120, 255, lambda x: None)
    cv2.createTrackbar('On', 'Video', 0, 1, lambda x: None)
    cv2.createTrackbar('RepeatDelay', 'Video', 5, 20, lambda x: None)
    cv2.createTrackbar('RepeatInterval', 'Video', 2, 20, lambda x: None)

    writer = None

    active_start_time = None
    last_keypress_time = None
    key_press_interval = 1.0
    for i in itertools.count():
        # Hit 'q' on the keyboard to quit!
        key = cv2.waitKey(wait_duration) & 0xFF
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
            img = raw_img[::downscale_factor, ::downscale_factor, :]
            background = fgbg.getBackgroundImage()
            if background is not None:
                diff = diff_image(background, img, filter_radius)
                diff_bgr = np.repeat(diff[:, :, np.newaxis], 3, axis=2)
                annotated_img = np.concatenate((img, background, diff_bgr), axis=0)
                with drag_lock:
                    if len(dz) == 2 and not dragging:
                        diffmax, ind = zone_argmax(diff, dz)
                        cv2.rectangle(annotated_img, dz[0], dz[1], (0, 255, 0), 2)
                        cv2.circle(annotated_img, (ind[1], ind[0]), filter_radius, (255, 0, 0))
                        cv2.putText(annotated_img, "{}".format(diffmax), (0, 20), default_font, 0.5,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                        if (diffmax > cv2.getTrackbarPos("Threshold", "Video") and
                                cv2.getTrackbarPos("On", "Video") == 1):
                            repeat_delay = cv2.getTrackbarPos("RepeatDelay", "Video") * 0.1
                            repeat_interval = cv2.getTrackbarPos("RepeatInterval", "Video") * 0.1
                            t = time.time()
                            press = False
                            if active_start_time is None:
                                active_start_time = t
                                last_keypress_time = None
                                press = True
                            elif last_keypress_time is None and t - active_start_time > repeat_delay:
                                press = True
                            elif last_keypress_time is not None and t - last_keypress_time > repeat_interval:
                                press = True
                            print(t - active_start_time, press)
                            if press:
                                subprocess.run(["xdotool", "key", "Down"])
                                last_keypress_time = t
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
