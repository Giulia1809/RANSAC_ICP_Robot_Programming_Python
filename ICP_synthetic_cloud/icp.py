import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
#import geometry_helpers
#env: tf-gpu
from geometry_helpers import v2t, skew


def brute_force_matcher(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)  # match descriptors
    matches = sorted(matches, key=lambda x: x.distance)  # sort matches
    good = []
    kp1_pix_coords = []
    kp2_pix_coords = []
    matches = matches[0:100]
    #print('matches = ', matches)
    for i in range(0, len(matches)):  # enumerate(matches):
        kp1_pix_coords.append(kp1[i].pt)
        kp2_pix_coords.append(kp2[i].pt)
    #print('kp1 pix coords = ', len(kp1_pix_coords))
    #print('kp2 pix coords = ', len(kp2_pix_coords))
    image_with_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(image_with_matches)
    plt.savefig(matched_dir + '/' + L[k])
    # plt.show()
    # returns keypoints coords in xy [pix] for each image -call recursively
    return kp1_pix_coords, kp2_pix_coords


def brute_force_matcher_sift(img1, img2):  # probable bug in distance method!
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    #matchesMask = [[0, 0] for i in range(len(matches))]
    good = []
    kp1_pix_coords = []
    kp2_pix_coords = []
    for i, (m, n) in enumerate(matches):
        #print('matches = ', len(matches))
        if m.distance < 0.7 * n.distance:
            #matchesMask[i] = [1,0]
            good.append([m])
            kp1_pix_coords.append(kp1[i].pt)
            #print('kp1 pix coords = ', len(kp1_pix_coords))
            #print('kp2 pix coords = ', len(kp2_pix_coords))
            kp2_pix_coords.append(kp2[i].pt)
            #print('kp2 pix coords = ', len(kp2_pix_coords))
            #[kp1_pix_coords.append(kp1[m.trainIdx].pt) for m in good]
            #[kp2_pix_coords.append(kp2[m.trainIdx].pt) for m in good]
    #print('good = ', len(good))
    #print('kp1 pix coords = ', len(kp1_pix_coords))
    #print('kp2 pix coords = ', len(kp2_pix_coords))

    image_with_matches = cv2.drawMatchesKnn(
        img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(image_with_matches)
    plt.savefig(matched_dir + '/' + L[k])
    plt.show()
    # returns keypoints coords in xy [pix] for each image -call recursively
    return kp1_pix_coords, kp2_pix_coords


def get_kp_coords(K, kp_pix_coords):
    K_inv = np.linalg.inv(K)
    kp_coords = []
    for pix_coord in kp_pix_coords:
        #print('pix_coord = ', pix_coord)
        xy_coord = K_inv * np.array([[pix_coord[0], pix_coord[1], 1]]).T
        kp_coords.append(xy_coord)
    # returns camera inv and keypoints coords in xy [m] for each image -call recursively
    return K_inv, kp_coords


def get_keypoint_depth(kp_pix_coords, img):
    keypoint_depths = []
    kp_depths = []
    for coords in kp_pix_coords:
        #print('coord = ', int(coords[0]))
        # depth of single kp in [mm]
        kp_d = img[int(coords[1]), int(coords[0])]
        #print('kp_d = ', kp_d)
        kp_depths.append(kp_d/1000)
    # returns keypoint's depth z vector in [m] for each image -call recursively
    return kp_depths


def get_point_in_camera_frame(kp_coords, kp_depths):
    P_vector = []
    #P_k = []
    for i in range(len(kp_coords)):
        P_k = kp_coords[i] * kp_depths[i]
        P_vector.append(P_k)
        # print('p vec z = ', P_vector[i][2]) #check if z is positive
    # returns keypoint's x-y-z coord.s in camera frame in [m] for each image -call recursively
    return P_vector


def sample_from_distribution_couples(P_fixed, P_moving):
    inliers = []
    for correspondance in zip(P_fixed[0:4], P_moving[0:4]):  # 0,1,2,3 idx
        #print('correspondance = ', correspondance)
        inliers.append(correspondance)
    return inliers


def compute_pose_ransac(P_fixed, P_moving, min_inliers=50):
    best_X = v2t(np.zeros(6))  # array
    best_err = np.inf
    for ransac_iter in range(0, 100):  # do 100 iters
        # Sample 4 indices in [0, P_fixed.shape[0]]
        inliers_indices = np.random.choice(P_fixed.shape[0], 4, replace=False)
        inliers_fixed = P_fixed[inliers_indices, :]
        inliers_moving = P_moving[inliers_indices, :]
        X_guess = do_icp(inliers_fixed, inliers_moving)
        total_error = 0.0
        inliers = 0
        # for p_f, p_m in zip(P_fixed, P_moving):
        for i in range(P_fixed.shape[0]):
            p_f = P_fixed[i, :]
            p_m = P_moving[i, :]
            error = np.linalg.norm(- p_f + X_guess[0:3, 0:3].T @
                                   p_m - X_guess[0:3, 3][np.newaxis].T)
            print('norm error = ', error)
            if error < 0.01:
                inliers += 1
            total_error += error
        print('num inliers = ', inliers)
        print('ransac tot error = ', total_error)
        if inliers >= min_inliers:
            if total_error < best_error:
                best_error = total_error
                best_X = X_guess
    return best_X, total_error


def error_and_jacobian(X, P_fixed, P_moving):  # X=X_initial
    J = np.zeros((3, 6))
    J[0:3, 0:3] = np.eye(3)
    J[0:3, 3:6] = -skew(P_moving)
    J = -X[:3, :3].T @ J
    error = X[:3, :3].T @ (P_moving[np.newaxis].T - X[:3, 3]
                           [np.newaxis].T) - P_fixed[np.newaxis].T
    return error, J


def do_icp(P_fixed, P_moving):
    X_init = v2t([0, 0, 0, 0, 0, 0])
    chi_lst = []
    for it in range(0, 100):
        H = np.zeros([6, 6])
        b = np.zeros([6, 1])
        chi = 0
        for i in range(P_fixed.shape[0]):
            p_f = P_fixed[i, :]
            p_m = P_moving[i, :]
            e, J = error_and_jacobian(X_init, p_f, p_m)
            chi += np.linalg.norm(e)
            H += np.transpose(J) @ J
            b += np.transpose(J) @ e

        dx = - np.linalg.inv(H) @ b
        #print('dx = ', dx)
        X_init = v2t(dx.flatten()) @ X_init
        chi_lst.append(chi)
    return X_init


if __name__ == "__main__":

    #directory = 'RawData'
    directory = 'RawDataSmall'
    #new_dir = 'RGBDataKeypointsMaxOut'
    #depth_directory = 'DepthData'
    depth_directory = 'DepthDataSmall'
    #matched_dir = 'MatchedRGBF2F'
    matched_dir = 'MatchedRGBF2FSmall'

    K = np.matrix([[600, 0, 320], [0, 600, 240], [0, 0, 1]])
    L = sorted(os.listdir(directory))
    D = sorted(os.listdir(depth_directory))

    for k, _ in enumerate(L[:-1]):
        #print(L[k+1] + " is the successive of (RGB) " + L[k])
        #print(D[k+1] + " is the successive of (depth) " + D[k])

        #print('k-th RGB image name : ', L[k])
        img1 = cv2.imread(directory + '/' + (L[k]), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(directory + '/' + (L[k+1]), cv2.IMREAD_GRAYSCALE)

        kp1_pix_coords, kp2_pix_coords = brute_force_matcher(img1, img2)

        _, kp1_coords = get_kp_coords(K, kp1_pix_coords)
        _, kp2_coords = get_kp_coords(K, kp2_pix_coords)

        img_depth1 = cv2.imread(depth_directory + '/' + (D[k]), -1)
        img_depth2 = cv2.imread(depth_directory + '/' + (D[k+1]), -1)
        # plt.imshow(img_depth1)
        # plt.show()
        #print('k-th depth image name : ', D[k])
        kp1_depths = get_keypoint_depth(kp1_pix_coords, img_depth1)
        kp2_depths = get_keypoint_depth(kp2_pix_coords, img_depth2)
        P_fixed = get_point_in_camera_frame(
            kp1_coords, kp1_depths)  # P1_vector = P_fixed
        P_moving = get_point_in_camera_frame(
            kp2_coords, kp2_depths)  # P2_vector = P_moving
        P_fixed = np.asarray(P_fixed)
        P_moving = np.asarray(P_moving)
        #print('P1 fixed vec  = ', P_fixed.shape)
        # print(P_fixed[0])
        #print('P2 vector = ', P_moving[0].shape)
        #print('length P1 vector = ', len(P1_vector))
        #inliers = sample_from_distribution_couples(P_fixed, P_moving)
        #print('correspondance = ', inliers)
        #fixed = []
        #moving = []
        # for i in range(0, 4):
        #  inl = inliers[i][0]
        #  fixed.append(inl)
        #fixed = np.asarray(fixed)
        #print('inl = ', fixed)
        #X_init = v2t([0,0,0,0,0,0])
        #J = np.zeros((3, 6))
        #error, J = error_and_jacobian(X_init, P_fixed, P_moving)
        best_X, total_error = compute_pose_ransac(
            P_fixed, P_moving, min_inliers=50)
        print('best_X = ', best_X)
        print('total error = ', total_error)
