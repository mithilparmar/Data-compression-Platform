import cv2
import numpy as np

class VideoWriter():

    def __init__(self,filename,fourcc,fps,aspect_rat):
        self.out_vid = cv2.VideoWriter(filename, fourcc, fps, aspect_rat)

    def compress(self,frames,isComp):

        for frame,i in zip(frames,isComp):

            out = frame
            if i:
                out = cv2.resize(frame, (960, 540))
                out = cv2.copyMakeBorder(out, 540, 0, 960, 0, cv2.BORDER_CONSTANT, value = (0, 0, 0))

            self.out_vid.write(out)
            
    def close_video(self):
        self.out_vid.release()

class VideoReader():

    def __init__(self,filename):
        self.video = cv2.VideoCapture(filename)
        
    def decompress(self):
        resized_frames = []
        i = 0
        while True:
            ret, frame = self.video.read()
            if not ret: break
            if np.sum(frame[:540, :960, :]) == 0: resized_frames.append(i)
            i += 1
        