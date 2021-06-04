import cv2
import time
import numpy as np
import modules.HandTrackingModule as htm
import math
import osascript
from subprocess import call

# Video Capture
cap = cv2.VideoCapture(0)

# Get the fps
prev_time = 0

# HTM Object
detector = htm.HandDetector(detection_confidence=0.7)

while True:
  succ, img = cap.read()
  img = detector.find_hands(img)
  lm_list = detector.find_position(img, draw=False)


  if len(lm_list) != 0:
    # Get volume settings
    min_volume = 0
    max_volume = 100

    # Coords for the index finder and thumb
    x1, y1 = lm_list[4][1], lm_list[4][2]
    x2, y2 = lm_list[8][1], lm_list[8][2]

    # Center of line
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    cv2.circle(img, (x1, y1), 15, (252, 165, 3), cv2.FILLED)
    cv2.circle(img, (x2, y2), 15, (252, 165, 3), cv2.FILLED)
    cv2.line(img, (x1, y1), (x2, y2), (252, 165, 3), 3)
    cv2.circle(img, (cx, cy), 15, (252, 165, 3), cv2.FILLED)

    # Length between fingers
    length = math.hypot(x2-x1, y2-y1)

    # The mapped volume
    vol = np.interp(length, [60, 300], [int(min_volume), int(max_volume)])

    print(vol)

    call(["osascript -e 'set volume output volume {}'".format(vol)], shell=True)

    # osascript.osascript("set volume output volume {}".format(vol))

    # Show green dot 
    if length < 60:
      cv2.circle(img, (cx, cy), 15, (144, 245, 66), cv2.FILLED)

  # Set fps
  curr_time = time.time()
  fps = 1 / (curr_time - prev_time)
  prev_time = curr_time

  cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (252, 165, 3), 2)

  cv2.imshow("img", img)
  cv2.waitKey(1)
