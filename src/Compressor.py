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

    def __init__(self,thresh) -> None:
        self.thresh = thresh
        self.model = hub.load(SAVED_MODEL_PATH)

    def preprocess_image(self,image):
        """ Loads image from path and preprocesses to make it model ready
            Args:
                image_path: Path to the image file
        """
        hr_image = tf.image.decode_image(tf.io.read_file("temp.jpg"))
        # If PNG, remove the alpha channel. The model only supports
        # images with 3 color channels.
        if hr_image.shape[-1] == 4:
            hr_image = hr_image[...,:-1]
        hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
        #hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
        hr_image = tf.cast(hr_image, tf.float32)
        return tf.expand_dims(hr_image, 0)
    
    def plot_image(image, title=""):
        """
            Plots images from image tensors.
            Args:
            image: 3D image tensor. [height, width, channels].
            title: Title to display in the plot.
        """
        image = np.asarray(image)
        print(type(image))
        image = tf.clip_by_value(image, 0, 255)
        print(type(image))
        print(image.shape)
        print(tf.cast(image, tf.uint8).numpy().shape)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
        plt.imshow(image)
        plt.axis("off")
        plt.title(title)
        plt.show()
        time.sleep(5)
        
    def downscale_image(image):
        """
            Scales down images using bicubic downsampling.
            Args:
                image: 3D or 4D tensor of preprocessed image
        """
        image_size = []
        if len(image.shape) == 3:
            image_size = [image.shape[1], image.shape[0]]
        else:
            raise ValueError("Dimension mismatch. Can work only on single image.")

        image = tf.squeeze(
            tf.cast(
                tf.clip_by_value(image, 0, 255), tf.uint8))

        lr_image = np.asarray(
            Image.fromarray(image.numpy())
            .resize([image_size[0] // 4, image_size[1] // 4],
                    Image.BICUBIC))

        lr_image = tf.expand_dims(lr_image, 0)
        lr_image = tf.cast(lr_image, tf.float32)
        return lr_image
    
    def decompress(self,frame):
        

        new_img = self.model(frame)
        return new_img
    
    def compare_frames(self,frame1,frame2):
        
        #h, w = frame1.shape
        diff= frame1-frame2
        err = np.sum(diff**2)
        #mse = err/(float(h*w))
        return err

    def compress_vid(self,path):
        
        video = cv2.VideoCapture(path)
        success = True

        ret_vid = []
        is_comp = []
        frame_count = 1
        while frame_count < 4:
            success, img = video.read()

            cv2.imwrite("temp.jpg",img)
            image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = Image.fromarray(image)
            
            #Compressor.plot_image(img)
            comp_img = self.preprocess_image(img)
            comp_img = Compressor.downscale_image(tf.squeeze(comp_img))
            #Compressor.plot_image(img)
            #print(type(img))
            reframed_img = self.decompress(comp_img)
            #print(reframed_img.shape)
            #Compressor.plot_image(tf.squeeze(reframed_img))
            current_frame = None
            if self.compare_frames(image,reframed_img.numpy()) < self.thresh:
                current_frame = reframed_img
                is_comp.append(True)
            else:
                current_frame = img
                is_comp.append(False)

            ret_vid.append(current_frame)
            
            frame_count += 1
            print("frame")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(video.get(cv2.CAP_PROP_FPS))
        width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  
        
        vWriter = VideoWriter("demo.mp4",fourcc=fourcc,fps=fps,aspect_rat=(width,height))

        vWriter.compress(ret_vid,is_comp)
        vWriter.close_video()
    
        

if __name__ == "__main__":
    c = Compressor(0.5)
    c.compress_vid("sample_video.mp4")
   