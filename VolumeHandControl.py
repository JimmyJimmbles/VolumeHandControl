import cv2
import time
import numpy as np
from modules.WebcamVideoStream import WebcamVideoStream
from modules.HandTrackingModule import HandDetector
import math
from Foundation import NSAppleScript

# Video Capture
wvs = WebcamVideoStream().start()

# Get the fps
prev_time = 0

# HTM Object
detector = HandDetector(detection_confidence=0.7)
setVol = True

while True:
  # Get the frame
  frame = wvs.read()

  # Find hands
  frame = detector.find_hands(frame)
  lm_list = detector.find_position(frame, draw=False)

  # Get current volume
  s = NSAppleScript.alloc().initWithSource_('output volume of (get volume settings)')
  result, error_info = s.executeAndReturnError_(None)
  vol = int(result.stringValue())

  # Display volume bar
  cv2.rectangle(frame, (75, (302 - (2 * vol))), (10, 400), (252, 165, 3), cv2.FILLED)
  cv2.putText(frame, 'Vol: {} %'.format(vol), (10, 440), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (252, 165, 3), 2)

  if len(lm_list) != 0:
    # Get volume settings
    min_volume = 0
    max_volume = 100

    # Coords for the index finder and thumb
    t1, t2 = lm_list[4][1], lm_list[4][2]
    i1, i2 = lm_list[8][1], lm_list[8][2]

    # Center of line
    cx, cy = (t1 + i1) // 2, (t2 + i2) // 2

    cv2.circle(frame, (t1, t2), 15, (252, 165, 3), cv2.FILLED)
    cv2.circle(frame, (i1, i2), 15, (252, 165, 3), cv2.FILLED)
    cv2.line(frame, (t1, t2), (i1, i2), (252, 165, 3), 3)
    cv2.circle(frame, (cx, cy), 15, (252, 165, 3), cv2.FILLED)

    # Length between fingers
    idxLength = math.hypot(i1-t1, i2-t2)

    # The mapped volume
    vol = np.interp(idxLength, [60, 300], [int(min_volume), int(max_volume)])

    print(vol)

    # Check if if fingers are up
    fingers = detector.fingers_up(lm_list)
    print("fingers: {}".format(fingers))


    if setVol:
      s = NSAppleScript.alloc().initWithSource_('set volume output volume {}'.format(vol))
      s.executeAndReturnError_(None)

      if fingers[2] != True:
        setVol = False
    else:
      if fingers[2] != True:
        setVol = True

    # Show green dot 
    if idxLength < 60:
      cv2.circle(frame, (cx, cy), 15, (144, 245, 66), cv2.FILLED)
  # Set fps
  curr_time = time.time()
  fps = 1 / (curr_time - prev_time)
  prev_time = curr_time

  print("vol: {} fps: {}".format(vol, int(fps)))

  cv2.putText(frame, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (252, 165, 3), 2)
  cv2.imshow("frame", frame)
  cv2.waitKey(1) & 0xFF
