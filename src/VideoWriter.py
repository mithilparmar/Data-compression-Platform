import numpy as np
import cv2
import matplotlib.pyplot as plt

class VideoWriter:

    # Class to write a video to file

    def _init_(self, filename, fourcc, fps, aspect_rat): 
      
      # Input Arguments:
      # filename: path + name of the file where the output file is to be written
      # fourcc: four-character code to uniquely identify data format
      # fps: frames per second
      # aspect_rat: aspect ratio

      self.out_vid = cv2.VideoWriter(filename, fourcc, fps, aspect_rat)   
      self.aspect_rat = aspect_rat

    def compress(self, frames, isComp):

      # Function to downscale the selected frames

      # Input Arguments:
      # frames: list of frames
      # isComp: list with the same length as frames and True for frames to be downscaled and False for frames not to be downscaled
        
        for frame, i in zip(frames, isComp):
            
            if i: 
              print(self.aspect_rat[0], self.aspect_rat[1])
              print(frame.shape[1], frame.shape[0])
              
              frame = cv2.copyMakeBorder(frame, self.aspect_rat[1] - frame.shape[0], 0, self.aspect_rat[0] - frame.shape[1], 0, cv2.BORDER_CONSTANT, value = (0, 0, 0))
              frame = np.int8(frame)
              plt.imshow(frame)
            print("out")
            print(type(frame))
            print(frame.shape)

            self.out_vid.write(frame)


            
    def close_video(self): 

      # Funtion to release the video
      
      self.out_vid.release()