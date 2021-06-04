import cv2
import mediapipe as mp
import time

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

  def find_hands(self, img, draw = True):
    # Get the current image from video 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    self.results = self.hands.process(img_rgb)

    # Make sure we have landmarks
    if self.results.multi_hand_landmarks:
      # Loop through all hands
      for hand_mark in self.results.multi_hand_landmarks:
        if draw:
          # Drawing the landmarks on the hand
          self.mp_draw.draw_landmarks(img, hand_mark, self.mp_hands.HAND_CONNECTIONS)

    return img

  def find_position(self, img, hand_no = 0, draw = True):
    lm_list = []

    if self.results.multi_hand_landmarks:
      hands = self.results.multi_hand_landmarks[hand_no]

      for id, lm in enumerate(hands.landmark):
        # Get the shape of the image(hand)
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        lm_list.append([id, cx, cy])

        if draw:
          cv2.circle(img, (cx, cy), 12, (252, 165, 3), cv2.FILLED)

    return lm_list

def main():
  # Get the fps
  prev_time = 0
  curr_time = 0

  # Start video 
  cap =  cv2.VideoCapture(0)
  detector = HandDetector()

  while True:
    # Get the current image from video 
    succ, img = cap.read()
    
    # Get image from detector
    img = detector.find_hands(img)
    # Get landmark list
    lm_list = detector.find_position(img)
    if len(lm_list) != 0:
      print(lm_list[4])

    # Set fps
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Show fps
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (252, 165, 3), 3)

    cv2.imshow("Image: ", img)
    cv2.waitKey(1)
    

if __name__ == "__main__":
  main()
