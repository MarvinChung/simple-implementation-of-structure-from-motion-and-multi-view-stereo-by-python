import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2
import glob, os
from mpl_toolkits import mplot3d

from Normalized8pointsAlgo import RANSAC
from CameraConfig import *
from Triangulation import *
from BundleAdjustment import *
from scipy.optimize import least_squares

import time

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    (r,c,_) = img1.shape
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


def TwoImage(img1_name, img2_name, C0, R0):
    img1 = cv2.imread(img1_name)
    img2 = cv2.imread(img2_name)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    #Oriented FAST and Rotated BRIEF
    orb = cv2.ORB_create(edgeThreshold=3)

    # find the keypoints with ORB
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    #img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)

    #plt.imshow(img3)
    #plt.show()

    trainPoints = cv2.KeyPoint_convert([kp1[matches[i].trainIdx] for i in range(len(matches))])
    queryPoints = cv2.KeyPoint_convert([kp2[matches[i].queryIdx] for i in range(len(matches))])
    pts1 = np.int32(trainPoints)
    pts2 = np.int32(queryPoints)
    
    my_F, train_inliers, query_inliers, ind = RANSAC(pts1, pts2, max_iter_times = 1000000) 
    
#     # Find epilines corresponding to points in right image (second image) and
#     # drawing its lines on left image
#     lines1 = cv2.computeCorrespondEpilines(query_inliers.reshape(-1,1,2), 2, my_F)
#     lines1 = lines1.reshape(-1,3)
#     img5,img6 = drawlines(img1.copy(),img2.copy(),lines1,train_inliers,query_inliers)
#     # Find epilines corresponding to points in left image (first image) and
#     # drawing its lines on right image
#     lines2 = cv2.computeCorrespondEpilines(train_inliers.reshape(-1,1,2), 1, my_F)
#     lines2 = lines2.reshape(-1,3)
#     img3,img4 = drawlines(img2.copy(),img1.copy(),lines2,query_inliers,train_inliers)
# #     print("my Fundamental matrix")
#     plt.imshow(img5)
#     plt.show()
#     plt.imshow(img3)
#     plt.show()
    
#     imageA = img1.copy()
#     imageB = img2.copy()
#     for i in train_inliers:
#         imageA = cv2.circle(imageA , (int(i[0]), int(i[1])), 2, (255, 0, 0), 20)
#     plt.imshow(imageA)
#     plt.show()
#     for i in query_inliers:
#         imageB = cv2.circle(imageB , (int(i[0]), int(i[1])), 2, (255, 0, 0), 20)
#     print("my inliers")
#     plt.imshow(imageB)
#     plt.show()
    
#     print('my F results')

    E = getEssentialMatrix(my_F)
    U, D, Vh = getEssentialConfig(E)

    #second camera
    C1, R1 = CameraPosition1Config(U, D, Vh)
    C2, R2 = CameraPosition2Config(U, D, Vh)
    C3, R3 = CameraPosition3Config(U, D, Vh)
    C4, R4 = CameraPosition4Config(U, D, Vh)
    Cs = [C1,C2,C3,C4]
    Rs = [R1,R2,R3,R4]

    best_ct = 0
    best_secondCamera_C = None
    best_secondCamera_R = None
    points = None

    for C,R in zip(Cs, Rs):
        P1, P2 = getCameraMatrix(U, Vh, K, R0, C0, R, C) 
        temp_points = getWorldPoints(train_inliers, query_inliers, P1, P2)
        n_in_front_of_C1, n_in_front_of_C2 = CheckCheirality(temp_points, C1, R1, C2, R2)
        if(n_in_front_of_C1 + n_in_front_of_C2 > best_ct):
            best_ct = n_in_front_of_C1 + n_in_front_of_C2
            best_secondCamera_C = C
            best_secondCamera_R = R
            points = temp_points
    #print("max points in front of both cameras:", best_ct)
    print("len(points):",len(points))
    #XZ
#     plt.scatter(points[:,0], points[:,2])
#     plt.scatter(C0[0], C0[2], c='red')
#     plt.scatter(C1[0], C1[2], c='black')
#     plt.show()
#     ax = plt.axes(projection='3d')
#     ax.scatter(points[:,0], points[:,1], points[:,2], c=points[:,2], cmap='viridis', linewidth=0.5);
#     ax.scatter(C0[0], C0[1], C0[2], c='red', linewidth=15);
#     ax.scatter(C1[0], C1[1], C1[2], c='black', linewidth=15);
#     plt.show()
    return best_secondCamera_C, best_secondCamera_R, K.astype(np.float64), train_inliers.astype(np.float64), query_inliers.astype(np.float64), points.astype(np.float64)
    
if __name__== "__main__":
    files = []
    for file in glob.glob("*.bmp"):
        print(file)
        files.append(file)
    camera_params = []
    camera_indices = []
    points_3d = []
    points_2d = []
    
    #first camera
    C0 = np.array([0,0,0])
    R0 = np.array([[1,0,0],[0,1,0],[0,0,1]])
    
    C = {}
    R = {}
    C[0] = C0
    R[0] = R0
    
    #table = []
    
    for img1_ct,img1 in enumerate(files):
        for img2_ct,img2 in enumerate(files):
            if img1 != img2 and img1_ct < img2_ct:
                print(img1_ct, img2_ct, img1, img2)
                dist_coef = np.zeros((4,1))
                outC, outR, K, train_inliers, query_inliers, points3D = TwoImage(img1,img2, C[img1_ct], R[img1_ct])
                if img1_ct == 0:
                    C[img2_ct] = outC
                    R[img2_ct] = outR
                print(len(train_inliers))
                print(len(points3D))
                try:
                    (_, rvec, tvec, _) = cv2.solvePnPRansac(points3D[:,0:3], train_inliers, K, dist_coef, cv2.SOLVEPNP_EPNP)
                except:
                    if(img1_ct==0):
                        raise
                    else:
                        continue
                camera_params.append([rvec[0].item(), rvec[1].item(), rvec[2].item(), tvec[0].item(), tvec[1].item(), tvec[2].item(), f1, 0, 0])
                for i in range(len(points3D)):
                    points_3d.append([points3D[i,0], points3D[i,1], points3D[i,2]])
                    camera_indices.append(img1_ct)
                    
                for i in range(len(train_inliers)):
                    points_2d.append(train_inliers[i]) 
                
                
                try:
                    (_, rvec, tvec, _) = cv2.solvePnPRansac(points3D[:,0:3], query_inliers, K, dist_coef, cv2.SOLVEPNP_EPNP)
                except:
                    continue
                camera_params.append([rvec[0].item(), rvec[1].item(), rvec[2].item(), tvec[0].item(), tvec[1].item(), tvec[2].item(), f1, 0, 0])
                for i in range(len(points3D)):
                    points_3d.append([points3D[i,0], points3D[i,1], points3D[i,2]])
                    camera_indices.append(img2_ct)
                for i in range(len(query_inliers)):
                    points_2d.append(query_inliers[i])
                

    camera_params = np.array(camera_params)
    camera_indices = np.array(camera_indices)
    print(len(points_3d))
    point_indices = np.array(range(len(points_3d)))
    points_3d = np.array(points_3d)
    points_2d = np.array(points_2d)
    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]

    n = 9 * n_cameras + 3 * n_points
    m = 2 * points_2d.shape[0]

    print("n_cameras: {}".format(n_cameras))
    print("n_points: {}".format(n_points))
    print("Total number of parameters: {}".format(n))
    print("Total number of residuals: {}".format(m))
    print(camera_params[0])
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))

    f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)
    #plt.plot(f0)
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
    t0 = time.time()
    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                    args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
    t1 = time.time()
    print("Optimization took {0:.0f} seconds".format(t1 - t0))
    #plt.plot(res.fun)
    #fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d)
    
    camera_params = res.x[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = res.x[n_cameras * 9:].reshape((n_points, 3))
    #XZ
    plt.scatter(points_3d[:,0], points_3d[:,2])
    #plt.scatter(C0[0], C0[2], c='red')
    #plt.scatter(C1[0], C1[2], c='black')
    plt.show()
    ax = plt.axes(projection='3d')
    ax.scatter(points_3d[:,0], points_3d[:,1], points_3d[:,2], c=points_3d[:,2], cmap='viridis', linewidth=0.5);
    #ax.scatter(C0[0], C0[1], C0[2], c='red', linewidth=15);
    #ax.scatter(C1[0], C1[1], C1[2], c='black', linewidth=15);
    plt.show()
