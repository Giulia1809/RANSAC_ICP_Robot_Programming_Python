import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
#env: tf-gpu


def get_images_and_keypoints():
  for image_name in sorted(os.listdir(directory)):
    print('img name = ',image_name)
    if image_name[-4:] == '.png':
      img = cv2.imread(directory + '/' + image_name) #, 0)
      #print(image_name)
      #plt.imshow(img) 
      #plt.show()
      
      orb = cv2.ORB_create(nfeatures=100)

      kp = orb.detect(img, None)
      #kp = orb.getMaxFeatures() #(img, None)
      kp, des = orb.compute(img, kp)
      kp_pix_coords = []
      #descriptors = []
      #print('des = ', des)
      for p in kp:
        #print('kp response = ', p.response) 
        #print('kp coords = ', p.pt)     
        kp_pix_coords.append(p.pt)
        #print('kp pix coords = ', kp_pix_coords)
        #print(len(kp_pix_coords))
        
      #for d in des:
        #print(d)
        #descriptors.append(d)
      
      
      img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
      plt.imshow(img2) 
      #plt.savefig(new_dir + '/' + image_name)
      #plt.show()
      
    #all_kp_pix_coords.append(kp_pix_coords)
    #print('all kp pix coords = ', all_kp_pix_coords)
      
      
  return kp_pix_coords, des #returns keypoint's pixel coordinate x,y vector and descriptors for each image (as is, saves only last image) -call recursively
 
def get_kp_coords(K, kp_pix_coords):
  K_inv = np.linalg.inv(K)
  kp_coords = []
  for pix_coord in kp_pix_coords:
    #print('pix_coord = ', pix_coord)
    xy_coord = K_inv * np.array([[pix_coord[0], pix_coord[1], 1]]).T
    kp_coords.append(xy_coord)
  return K_inv, kp_coords #returns camera inv and keypoints coords in xy [m] for each image (as is, saves only last image) -call recursively
  
def get_keypoint_depth(kp_pix_coords):
  keypoint_depths = []
  for image_name in os.listdir(depth_directory):
    #print('depth_img = ',image_name)
    img = cv2.imread(depth_directory + '/' + image_name, -1) #-1: open as greyscale in [mm]
    kp_depths = []
    for coords in kp_pix_coords:
      #print('coord = ', int(coords[0]))
      kp_d =  img[int(coords[1]), int(coords[0])] #depth of single kp in [mm]
      #print('kp_d = ', kp_d)
      kp_depths.append(kp_d/1000)
  return kp_depths #returns keypoint's depth z vector in [m] for each image (as is, saves only last image) -call recursively
  
def get_point_in_camera_frame(kp_coords, kp_depths):
  P_vector = []
  #P_k = []
  for i in range(len(kp_coords)):
    print('i ', i)
    P_k = kp_coords[i] * kp_depths[i] 
    P_vector.append(P_k)
  return P_vector #returns keypoint's x-y-z coord.s in camera frame in [m] for each image -call recursively
  
#def 
  
directory = 'RawData'
new_dir = 'RGBDataKeypointsMaxOut'
depth_directory = 'DepthData'
  
kp_pix_coords, descriptors = get_images_and_keypoints()
#print('kp pix coords = ', kp_pix_coords[0][0])
#print('descriptors = ', descriptors)

K = np.matrix([[600, 0, 320], [0, 600, 240], [0, 0, 1]])
K_inv, kp_coords = get_kp_coords(K, kp_pix_coords)
#print(kp_coords)
#print('length kp_coords = ', len(kp_coords))

kp_depths = get_keypoint_depth(kp_pix_coords)
#print('kp_depths = ', kp_depths)
#print('length kp depths = ', len(kp_depths))

P_vector = get_point_in_camera_frame(kp_coords, kp_depths)
#print('P vector = ', P_vector)
#print('lengt P vector = ', len(P_vector))





