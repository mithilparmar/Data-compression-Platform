#importing required libraries
import cv2
import numpy as np
from sklearn.decomposition import PCA
from google.colab import drive
import os

#mounting the google drive
drive.mount("/content/drive/")

# Load video
video = cv2.VideoCapture('/content/drive/My Drive/SEM 4/CMPE 295B Final Year Project/vp9_compressed_videos_Gaming_1080P-0ce6_cbr.webm')

# Get video properties
fps = int(video.get(cv2.CAP_PROP_FPS))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
input_size = os.path.getsize('/content/drive/My Drive/SEM 4/CMPE 295B Final Year Project/vp9_compressed_videos_Gaming_1080P-0ce6_cbr.webm')
print(fps, width, height, num_frames, input_size)

# Create VideoWriter object to write new video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/content/drive/My Drive/SEM 4/CMPE 295B Final Year Project/few_framed_resized_and_padded_video.mp4', fourcc, fps, (1920, 1080))

#resize and pad alternate frames
i = 0
while True:
    ret, frame = video.read()
    if not ret: break
    if i%2 != 0: 
      frame = cv2.resize(frame, (960, 540))
      frame = cv2.copyMakeBorder(frame, 540, 0, 960, 0, cv2.BORDER_CONSTANT, value = (0, 0, 0))
    out.write(frame)
    i+=1

    #release video
out.release()