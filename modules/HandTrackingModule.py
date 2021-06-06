import cv2
import mediapipe as mp
import math

class HandDetector():
  def __init__(self, mode = False, max_hands = 2, detection_confidence = 0.5, tracking_confidence = 0.5):
    self.mode = mode
    self.max_hands = max_hands
    self.detection_confidence = detection_confidence
    self.tracking_confidence = tracking_confidence

    # Set up mediapipe
    self.mp_hands = mp.solutions.hands
    self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.detection_confidence, self.tracking_confidence)
    self.mp_draw = mp.solutions.drawing_utils
    self.tipIds = [4, 8, 12, 16, 20]

  def find_hands(self, frame, draw = True):
    # Get the current image from video 
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    self.results = self.hands.process(frame_rgb)

    # Make sure we have landmarks
    if self.results.multi_hand_landmarks:
      # Loop through all hands
      for hand_mark in self.results.multi_hand_landmarks:
        if draw:
          # Drawing the landmarks on the hand
          self.mp_draw.draw_landmarks(frame, hand_mark, self.mp_hands.HAND_CONNECTIONS)

    return frame

  def find_position(self, frame, hand_no = 0, draw = True):
    x_list = []
    y_list = []
    bb = []

    self.lm_list = []

    # Make sure there are hands
    if self.results.multi_hand_landmarks:
      hands = self.results.multi_hand_landmarks[hand_no]

      for id, lm in enumerate(hands.landmark):
        # Get the shape of the image(hand)
        h, w, c = frame.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        x_list.append(cx)
        y_list.append(cy)
        self.lm_list.append([id, cx, cy])

        if draw:
          cv2.circle(frame, (cx, cy), 8, (235, 228, 150), cv2.FILLED)

      x_min, x_max = min(x_list), max(x_list)
      y_min, y_max = min(y_list), max(y_list)

      bb = x_min, y_min, x_max, y_max

      if draw:
        cv2.rectangle(frame, (bb[0]-20, bb[1]-20), (bb[2]+20, bb[3]+20), (252, 165, 3), 2)

    return self.lm_list, bb

  def fingers_up(self):
    multi_handedness = self.results.multi_handedness
    handedness = multi_handedness[0].classification[0].label
    fingers = []

    # Thumb for both hands
    if handedness == "Right":
      if self.lm_list[self.tipIds[0]][1] < self.lm_list[self.tipIds[0]-1][1]:
        fingers.append(1)
      else:
        fingers.append(0)
    else:
      if self.lm_list[self.tipIds[0]][1] > self.lm_list[self.tipIds[0]-1][1]:
        fingers.append(1)
      else:
        fingers.append(0)

    for id in range(1, 5):
      if self.lm_list[self.tipIds[id]][2] < self.lm_list[self.tipIds[id]-2][2]:
        fingers.append(1)
      else:
        fingers.append(0)
    return fingers

  def get_distance(self, p1, p2, frame, draw = True):
    # Coords for the two fingers
    x1, x2 = self.lm_list[p1][1], self.lm_list[p1][2]
    y1, y2 = self.lm_list[p2][1], self.lm_list[p2][2]

    # Center of line
    cx, cy = (x1 + y1) // 2, (x2 + y2) // 2

    if draw:
      # Display lines between two fingers
      cv2.circle(frame, (x1, x2), 15, (252, 165, 3), cv2.FILLED)
      cv2.circle(frame, (y1, y2), 15, (252, 165, 3), cv2.FILLED)
      cv2.line(frame, (x1, x2), (y1, y2), (252, 165, 3), 3)
      cv2.circle(frame, (cx, cy), 15, (252, 165, 3), cv2.FILLED)

    length = math.hypot(y1-x1, y2-x2)

    return length, frame, [x1 ,y1, x2, y2, cx, cy]
