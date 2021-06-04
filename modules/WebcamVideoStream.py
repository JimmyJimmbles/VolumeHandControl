from threading import Thread
import cv2

class WebcamVideoStream:
  def __init__(self, src = 0):
    # Set up video capture and read first frames
    self.stream = cv2.VideoCapture(src)
    (self.grabbed, self.frame) = self.stream.read()

    # If thread should be stopped
    self.stopped = False

  def start(self):
    # Start thread to read frames
    Thread(target=self.update, args=()).start()
    return self

  def update(self):
    # Keep loop running until thread is stopped
    while True:
      # Stop the thread
      if self.stopped:
        return

      # Read the next frame
      (self.grabbed, self.frame) = self.stream.read()
  
  def read(self):
    # Return the frame
    return self.frame

  def stop(self):
    # stop the thread
    self.stopped = True
