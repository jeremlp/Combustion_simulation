# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 20:05:59 2024

@author: jerem
"""

import cv2
import os

image_folder = "D:\\JLP\CMI\\_MASTER 2_\\TC5-NUM\AYMERIC\\PROJECT 3\\ANIMATIONS\\CHEM\\"
video_name = 'D:\\JLP\\CMI\\_MASTER 2_\\TC5-NUM\\AYMERIC\\PROJECT 3\\ANIMATIONS\\chem_anim.mp4'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

FPS = 30
video = cv2.VideoWriter(video_name, 0, FPS, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()