import cv2
import tensorflow_hub as hub
import numpy as np
from PIL import Image

SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

class DeCompressor:
        
    def __init__(self,compressedFileName,decompressedFileName) -> None:
        self.model = hub.load(SAVED_MODEL_PATH)
        self.video = cv2.VideoCapture(compressedFileName)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(self.video.get(cv2.CAP_PROP_FPS))
        width  = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)) 
        height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)) 

        self.vid_writer = cv2.VideoWriter(decompressedFileName, fourcc, fps, (width,height))

    def decompress_image(self,frame):
        new_img = self.model(frame)
        return new_img
    

    def decompress(self): 
        i = 0
        while True:
            ret, frame = self.video.read()

            image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            img = Image.fromarray(image)
            comp_img = self.preprocess_image(img)

            if not ret: 
                break

            if np.sum(frame[:540, :960, :]) == 0: 

                img = self.decompress_image(comp_img)

            self.vid_writer.write()
        self.vid_writer.release()

            
           