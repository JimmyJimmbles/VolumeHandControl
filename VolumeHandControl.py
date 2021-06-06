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

# Get volume settings
min_volume = 0
max_volume = 100
area = 0

# HandDetector Object
detector = HandDetector(detection_confidence=0.7)
setVol = True

while True:
  # Get the frame
  frame = wvs.read()

  # Find hands
  frame = detector.find_hands(frame)
  lm_list, bb = detector.find_position(frame )

  # Get current volume
  s = NSAppleScript.alloc().initWithSource_('output volume of (get volume settings)')
  result, error_info = s.executeAndReturnError_(None)
  vol = int(result.stringValue())

  if len(lm_list) != 0:
    # Filter based on area
    area = (bb[2] - bb[0]) * (bb[3] - bb[1]) // 100

    if 200 < area < 2000:
      # Find distance between index and thumb
      length, img, data = detector.get_distance(4, 8, frame)

      # The mapped volume
      volPer = np.interp(length, [50, 300], [0, 100])
      smooth = 5
      volPer = smooth * round(volPer/smooth)
      vol = volPer

      # Check what fingers are up
      fingers = detector.fingers_up()

      # Set volume when ring finger is down
      if not fingers[3]:
        # Adjust the system volume
        s = NSAppleScript.alloc().initWithSource_('set volume output volume {}'.format(vol))
        s.executeAndReturnError_(None)
        
        # Show green dot 
        cv2.circle(frame, (data[4], data[5]), 15, (144, 245, 66), cv2.FILLED)


  # Display volume bar
  cv2.rectangle(frame, (75, (400 - (2 * vol))), (10, 400), (201, 194, 117), cv2.FILLED)
  cv2.putText(frame, 'Vol: {} %'.format(vol), (10, 440), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (201, 194, 117), 2)
  # Set fps
  curr_time = time.time()
  fps = 1 / (curr_time - prev_time)
  prev_time = curr_time

  print("vol: {} fps: {}".format(vol, int(fps)))

  cv2.putText(frame, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (201, 194, 117), 2)
  cv2.imshow("frame", frame)
  cv2.waitKey(1) & 0xFF
