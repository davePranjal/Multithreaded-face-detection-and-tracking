# -*- coding: utf-8 -*-
"""
Created on Fri May  8 23:48:40 2020

@author: pranj
"""
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import numpy as np
import cv2
from matplotlib import pyplot as plt
import face_recognition
from random import randint
import threading

faces = []
curr_frame = []
def isBoxtooclose(box,boxes):
    res = False
    for element in boxes:
        diff = tuple(np.subtract(element,box))
        res = res or  all(x < y for x, y in zip(diff, (5,5,5,5)))
    return res

## Detection Function ##
def detect_faces(detect_eve,read_eve):
    # lock.acquire()
    read_eve.wait()
    # if not detect_eve.is_set():
    detect_eve.set()
    print("Trying to detect a new face... ")
    if len(curr_frame) != 0:
        frame = curr_frame[0]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.5, 5)
        print(f" Got a face {faces}")
    # detect_eve.set()
    detect_eve.clear()
# if read_eve.is_set():
    read_eve.clear()
    # if lock.locked():
    #     lock.release()

## tracking function ##
def trackingfn(detect_eve,read_eve):
    # detect_eve.wait()
    read_eve.wait()
    # if len(faces)==0:
    #     if lock.locked():
    #         lock.release()
    count = 0
    while not detect_eve.is_set():
        count+=1
        for (x,y,w,h) in faces:
            box = tuple((x,y,w,h))
            tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
            if not isBoxtooclose(box, boxes) and len(curr_frame) != 0:
                trackers.add(tracker, curr_frame[0], tuple(box))
        if len(curr_frame) != 0:
            (success, boxes) = trackers.update(curr_frame[0])
            for box in boxes:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(curr_frame[0], (x, y), (x + w, y + h), (0, 255, 0), 2)
            curr_frame.append(curr_frame[0])
            cv2.imshow("Frame", curr_frame[0])
        # if read_eve.is_set():
        read_eve.clear()
        if count == 15:
            break

        # detect_eve.clear()
    print("reached here")
    # key = cv2.waitKey(1) & 0xFF
    # if lock.locked():
    #     lock.release()


def frame_reader(vs,detect_eve,read_eve):
    # while not read_eve.is_set():
    while True:
        frame = vs.read()
        # count = count +1
        frame = frame[1] if args.get("video", False) else frame
        if frame is None:
            return
        frame = imutils.resize(frame, width=500)
        curr_frame.append(frame)
        read_eve.set()
# initialize the FPS throughput estimator
#fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# # output_movie = cv2.VideoWriter('output.mp4', fourcc, fps, (1280, 720), True)
# fourcc = cv2.VideoWriter_fourcc('M','P','E','G')
# #fourcc = -1
#
# output_movie = cv2.VideoWriter('output.mp4', fourcc, 25.07, (1280, 720))
# count = 0

def main():
    # frame = vs.read()
    # # count = count +1
    # frame = frame[1] if args.get("video", False) else frame
    # if frame is None:
    #     return
    # frame = imutils.resize(frame, width=500)
    # curr_frame.append(frame)
    # (H, W) = frame.shape[:2]


    ## Start video capture ##
    vs = cv2.VideoCapture(args["video"])

    detect_eve = threading.Event()
    read_eve = threading.Event()
    detection = threading.Thread(target = detect_faces, name = "detection", args = (detect_eve,read_eve))
    tracking = threading.Thread(target = trackingfn, name = "tracking" , args = (detect_eve,read_eve))
    frame_read = threading.Thread(target = frame_reader, name = "frame_read", args = (vs,detect_eve,read_eve))
    ## starting the threads ##
    detection.start()
    tracking.start()
    frame_read.start()

    ## joining the threads ##
    detection.join()
    tracking.join()
    frame_read.join()

    vs.release()

    #print("Writing frame {} / {}".format(frame_number, length))
    # output_movie.write(frame)
    # if key == ord("q"):
    #     break


if __name__  == "__main__":
    ## Load classifier ##
    face_cascade = cv2.CascadeClassifier('C:\\Users\\pranj\\anaconda\\pkgs\\libopencv-4.2.0-py37_6\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml')

    ## Argument parser ##
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str,
        help="path to input video file")
    ap.add_argument("-t", "--tracker", type=str, default="kcf",
        help="OpenCV object tracker type")
    args = vars(ap.parse_args())

    ## List of possible trackers ##
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }

    ## Create trackers ##
    trackers = cv2.MultiTracker_create()

    ## Get fps ##
    # fps = vs.get(cv2.CAP_PROP_FPS)
    main()

    # output_movie.release()
    cv2.destroyAllWindows()
