import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

directory = 'RawData'
new_dir = 'RGBDataKeypointsMaxOut'

def get_images():
  for image_name in os.listdir(directory):
    if image_name[-4:] == '.png':
      img = cv2.imread(directory + '/' + image_name) #, 0)
      #plt.imshow(img) 
      #plt.show()
      
      orb = cv2.ORB_create(nfeatures=100)

      kp = orb.detect(img, None)

      print('kp = ', kp)
      
      for i in range(0,len(kp)-1):
        overlap = cv2.KeyPoint.overlap(kp[i], kp[i+1])
      img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)

      plt.imshow(img2) 
      plt.savefig(new_dir + '/' + image_name)
      plt.show()
  return kp_pix_coords
  
  
kp_pix_coords = get_images()

