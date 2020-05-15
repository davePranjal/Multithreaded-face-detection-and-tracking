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
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN

def isBoxtooclose(box,boxes):
    res = False
    for element in boxes:
        diff = tuple(np.subtract(element,box))
        res = res or  all(x < y for x, y in zip(diff, (50,50,50,50)))
    return res
### Geting the detection boxes ###

detector = MTCNN()

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
    help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
    help="OpenCV object tracker type")
args = vars(ap.parse_args())

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}

trackers = cv2.MultiTracker_create()
# initialize the bounding box coordinates of the object we are going
# to track
instance = 0
newDetected = False
colors =[]
fps = 30
# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)
# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])
    fps = vs.get(cv2.CAP_PROP_FPS)

# initialize the FPS throughput estimator
#fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# output_movie = cv2.VideoWriter('output.mp4', fourcc, fps, (1280, 720), True)
fourcc = cv2.VideoWriter_fourcc('M','P','E','G')
#fourcc = -1

output_movie = cv2.VideoWriter('output.mp4', fourcc, 25.07, (1280, 720))
count = 0
# loop over frames from the video stream
while True:
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    frame = vs.read()
    count = count +1
    frame = frame[1] if args.get("video", False) else frame
    # check to see if we have reached the end of the stream
    if frame is None:
        break
    # resize the frame (so we can process it faster) and grab the
    # frame dimensions
    frame = imutils.resize(frame, width=500)
    (H, W) = frame.shape[:2]
    # grab the updated bounding box coordinates (if any) for each
    # object that is being tracked
    (success, boxes) = trackers.update(frame)
    # loop over the bounding boxes and draw then on the frame
    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if newDetected:
            roi_color = frame[y:y + h, x:x + w]
            roi_color = cv2.resize(roi_color,None,fx=3,fy=3)
            cv2.imwrite("C:\\Users\\pranj\\Desktop\\Project_mixorg\\Multithreaded-face-detection-and-tracking\\Faces_detected\\"+str(instance) + str(w) + str(h) + '_faces.jpg', roi_color)
            newDetected = False
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    # if key == ord("s"):
    if count == 15 :
        print("Trying to detect new face.... ")
        faces = detector.detect_faces(frame)
        for face in faces:
            # print("reached here ")
                # img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                # roi_gray = gray[y:y+h, x:x+w]
            box = tuple(face["box"])
            # create a new object tracker for the bounding box and add it
            # to our multi-object tracker
            tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
            if not isBoxtooclose(box, boxes):
                trackers.add(tracker, frame, tuple(box))
                instance+=1
                newDetected = True

        count = 0
    #print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)
    if key == ord("q"):
        break
# if we are using a webcam, release the pointer
if not args.get("video", False):
    vs.stop()
# otherwise, release the file pointer
else:
    vs.release()
# close all windows
output_movie.release()
cv2.destroyAllWindows()
