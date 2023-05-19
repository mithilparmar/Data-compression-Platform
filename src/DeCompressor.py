import cv2
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import tensorflow as tf

global glob_frame
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

class DeCompressor:
        
    def __init__(self,compressedFileName,decompressedFileName) -> None:
        self.model = hub.load(SAVED_MODEL_PATH)
        self.video = cv2.VideoCapture(compressedFileName)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(self.video.get(cv2.CAP_PROP_FPS))
        width  = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(width, height)

        self.vid_writer = cv2.VideoWriter(decompressedFileName, fourcc, fps, (width,height))

    def decompress_image(self,frame):
        new_img = self.model(frame)
        return new_img
    
    def preprocess_image(self,image):
        """ Loads image from path and preprocesses to make it model ready
            Args:
                image_path: Path to the image file
        """
        hr_image = tf.image.decode_image(tf.io.read_file("temp_decompress.jpg"))
        # If PNG, remove the alpha channel. The model only supports
        # images with 3 color channels.
        if hr_image.shape[-1] == 4:
            hr_image = hr_image[...,:-1]
        hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
        #hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
        hr_image = tf.cast(hr_image, tf.float32)
        return tf.expand_dims(hr_image, 0)
    
    def decompress(self): 
        i = 0
        while True:
            ret, frame = self.video.read()

            if not ret:
              break
            
            width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if np.sum(frame[:(3*height)//4, :(3*width)//4, :]) == 0:

                frame = frame[(3*height)//4:, (3*width)//4:, :]
                
                cv2.imwrite("temp_decompress.jpg", frame)
                comp_img = self.preprocess_image(frame)
                
                image = self.decompress_image(comp_img)
                frame = np.array(image).squeeze()
                glob_frame = frame

                cv2.imwrite("temp.jpg",frame)
                print(frame.shape)

                frame = cv2.imread("temp.jpg")
                
                #plt.imshow(frame)
                #plt.show()
                #plt.imshow(frame)
                print("decompressing")

            #img = cv2.imread("temp_img")
            # img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            self.vid_writer.write(frame)
        self.vid_writer.release()