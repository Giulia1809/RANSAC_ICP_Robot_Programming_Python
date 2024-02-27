import numpy as np 

def skew(v): 
  v_skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) #shape=(3,3)
  return v_skew
  
def Rx(rot_x): 
  c = np.cos(rot_x)
  s = np.sin(rot_x)
  R_x = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
  return R_x
  
def Ry(rot_y): 
  c = np.cos(rot_y)
  s = np.sin(rot_y)
  R_y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
  return R_y
  
def Rz(rot_z): 
  c = np.cos(rot_z)
  s = np.sin(rot_z)
  R_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
  return R_z
  
def angles2R(a):
  R = Rx(a[0]) @ Ry(a[1]) @ Rz(a[2])
  return R
  
def v2t(state_vector):
  T = np.eye(4)
  T[0:3,0:3] = angles2R(state_vector[3:6])
  T[0:3,3] = state_vector[0:3]
  return T
  

