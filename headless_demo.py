%%writefile /content/UCMCTrack/demo_headless.py

#!/usr/bin/env python3
"""
Head-less demo for UCMCTrack.
Reads an MP4, runs YOLO v8 + UCMCTrack on every frame,
and writes the annotated result to a new MP4.
"""

import os
import cv2
import argparse
import numpy as np
from ultralytics import YOLO

from tracker.ucmc import UCMCTrack
from detector.mapper import Mapper


class Detection:
    """Simple container matching the original repo’s Detection struct."""

    def __init__(self, det_id,
                 bb_left=0, bb_top=0,
                 bb_width=0, bb_height=0,
                 conf=0.0, det_class=0):
        self.id = det_id
        self.bb_left = bb_left
        self.bb_top = bb_top
        self.bb_width = bb_width
        self.bb_height = bb_height
        self.conf = conf
        self.det_class = det_class
        self.track_id = 0
        self.y = np.zeros((2, 1))
        self.R = np.eye(4)

    def __str__(self):
        return (f"d{self.id}, bb:[{self.bb_left},{self.bb_top},"
                f"{self.bb_width},{self.bb_height}], "
                f"conf={self.conf:.2f}, cls{self.det_class}, "
                f"uv:[{self.bb_left+self.bb_width/2:.0f},"
                f"{self.bb_top+self.bb_height:.0f}], "
                f"mapped:[{self.y[0,0]:.1f},{self.y[1,0]:.1f}]")

    __repr__ = __str__


class Detector:
    """Wrap YOLO-v8 inference + image-to-ground mapping."""

    def __init__(self):
        self.mapper: Mapper | None = None
        self.model: YOLO | None = None

    def load(self, cam_para_file: str):
        self.mapper = Mapper(cam_para_file, "MOT17")
        self.model = YOLO("pretrained/yolov8x.pt")

    def get_dets(self, img, conf_thresh=0.01, det_classes=(0,)):
        dets, det_id = [], 0

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        preds = self.model(rgb, imgsz=1088)

        for box in preds[0].boxes:
            conf = float(box.conf)
            cls_id = int(box.cls)
            if conf < conf_thresh or cls_id not in det_classes:
                continue

            x1, y1, x2, y2 = map(float, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            if w <= 10 or h <= 10:
                continue

            det = Detection(det_id, x1, y1, w, h, conf, cls_id)
            det.y, det.R = self.mapper.mapto(
                [det.bb_left, det.bb_top, det.bb_width, det.bb_height])
            dets.append(det)
            det_id += 1
        return dets


def run(args):
    # --------------------------------------------------
    # I/O
    # --------------------------------------------------
    if not os.path.isfile(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not os.path.isfile(args.cam_para):
        raise FileNotFoundError(f"Camera-parameter file not found: {args.cam_para}")

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    writer = cv2.VideoWriter(
        args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # --------------------------------------------------
    # init detector + tracker
    # --------------------------------------------------
    classes_of_interest = (2, 5, 7)       # car, bus, truck in COCO
    detector = Detector()
    detector.load(args.cam_para)

    tracker = UCMCTrack(args.a, args.a, args.wx, args.wy,
                        args.vmax, args.cdt, fps,
                        "MOT", args.high_score, False, None)

    # --------------------------------------------------
    # main loop
    # --------------------------------------------------
    frame_id = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        dets = detector.get_dets(frame, args.conf_thresh, classes_of_interest)
        tracker.update(dets, frame_id)

        # draw results
        for det in dets:
            if det.track_id <= 0:
                continue
            x1, y1 = int(det.bb_left), int(det.bb_top)
            x2, y2 = int(det.bb_left + det.bb_width), int(det.bb_top + det.bb_height)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(det.track_id), (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        writer.write(frame)
        frame_id += 1
        if frame_id % 10 == 0:
            print('***********************************************************')
            print(f"[UCMCTrack] processed {frame_id} frames…", flush=True)
            print('***********************************************************')

    cap.release()
    writer.release()
    print(f"Done. Output saved to {args.out}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Head-less demo for UCMCTrack (designed for Colab / non-GUI).")
    p.add_argument("--video",      type=str, default="demo/demo.mp4",
                   help="input video file")
    p.add_argument("--cam_para",   type=str, default="demo/cam_para.txt",
                   help="camera parameter file")
    p.add_argument("--out",        type=str, default="output/output.mp4",
                   help="output video file")
    p.add_argument("--wx",         type=float, default=5.0)
    p.add_argument("--wy",         type=float, default=5.0)
    p.add_argument("--vmax",       type=float, default=10.0)
    p.add_argument("--a",          type=float, default=100.0,
                   help="assignment threshold")
    p.add_argument("--cdt",        type=float, default=10.0,
                   help="coasted deletion time")
    p.add_argument("--high_score", type=float, default=0.5)
    p.add_argument("--conf_thresh", type=float, default=0.01)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
