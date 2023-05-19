import os
import time
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import cv2
from math import sqrt
from VideoWriter import VideoWriter

SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

class Compressor:

    # Class to compress the videos

    def __init__(self, thresh, model_path) -> None:

        # Input Arguments:
        # thresh: value of RMSE below which the frames will be downscaled
        # model_path: path to the trained and saved model
        
        self.thresh = thresh
        self.model = hub.load(model_path)

    def preprocess_image(self, image):
        
        # Function to load image from path and preprocesses to make it model ready

        # Input Arguments:
        # image_path: Path to the image file
        
        file = tf.io.read_file("temp.jpg")
        hr_image = tf.image.decode_image(file)
        
        if hr_image.shape[-1] == 4: hr_image = hr_image[...,:-1]
        
        hr_image = tf.cast(hr_image, tf.float32)
        
        return tf.expand_dims(hr_image, 0)

    def downscale_image(image):
        
        # Function to scale down images using bicubic downsampling.
        
        # Input Arguments:
        # image: 3D or 4D tensor of preprocessed image
        
        image_size = []
        if len(image.shape) == 3: image_size = [image.shape[1], image.shape[0]]
        else: raise ValueError("Dimension mismatch. Can work only on single image.")

        clipped = tf.clip_by_value(image, 0, 255)
        casted = tf.cast(clipped, tf.uint8)
        image = tf.squeeze(casted)
        img_np = image.numpy()
        img_pil = Image.fromarray(img_np)
        img_pil_resized = img_pil.resize([image_size[0] // 4, image_size[1] // 4], Image.BICUBIC)
        lr_image = np.asarray(img_pil_resized)
        lr_image = tf.expand_dims(lr_image, 0)
        lr_image = tf.cast(lr_image, tf.float32)
        
        return lr_image
    
    def decompress(self, frame): 

      # Function to pass the downscaled frame through model to get the upscaled frame

      # Input Arguments:
      # frame: preprocessed model ready frame
      
      return self.model(frame)

    
    def compare_frames(self, frame1, frame2):

        # Function to calculate 
        
        diff = np.subtract(frame1, frame2)
        norm = diff / 255
        sq = np.square(norm)
        sum = np.sum(sq)
        div = sum / (sq.shape[0]*sq.shape[1])
        sqrt = np.sqrt(div)
        
        return sqrt

    def compress_vid(self,path):
        
        video = cv2.VideoCapture(path)
        success = True

        ret_vid = []
        is_comp = []
        while success:
            
            success, img = video.read()

            if not success: break
              
            cv2.imwrite("temp.jpg", img)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(image)
            comp_img = self.preprocess_image(img)
            sq = tf.squeeze(comp_img)
            comp_img = Compressor.downscale_image(sq)
            reframed_img = self.decompress(comp_img)
            reframed_img_np = reframed_img.numpy()
            reframed_img_np_sq = reframed_img_np.squeeze()
            
            current_frame = None
            err = self.compare_frames(image, reframed_img_np_sq)
            
            if err < self.thresh: 
                comp_img_np = np.array(comp_img)
                current_frame = comp_img_np.squeeze()
                is_comp.append(True)
            else:
                current_frame = np.array(img)
                is_comp.append(False)

            ret_vid.append(current_frame)

       
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(video.get(cv2.CAP_PROP_FPS))
        width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) 
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  
        
        vWriter = VideoWriter("demo.mp4", fourcc = fourcc, fps = fps, aspect_rat = (width, height))
        
        vWriter.compress(ret_vid, is_comp)
        vWriter.close_video()