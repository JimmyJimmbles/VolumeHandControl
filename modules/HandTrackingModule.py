import cv2
import mediapipe as mp

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
    lm_list = []

    if self.results.multi_hand_landmarks:
      hands = self.results.multi_hand_landmarks[hand_no]

      for id, lm in enumerate(hands.landmark):
        # Get the shape of the image(hand)
        h, w, c = frame.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        lm_list.append([id, cx, cy])

        if draw:
          cv2.circle(frame, (cx, cy), 12, (252, 165, 3), cv2.FILLED)

    return lm_list

  def fingers_up(self, lm_list):
    fingers = []

    # Thumb
    if lm_list[self.tipIds[0]][1] < lm_list[self.tipIds[0]-1][1]:
      fingers.append(1)
    else:
      fingers.append(0)

    for id in range(1, 5):
      if lm_list[self.tipIds[id]][2] < lm_list[self.tipIds[id]-2][2]:
        fingers.append(1)
      else:
        fingers.append(0)
    return fingers


