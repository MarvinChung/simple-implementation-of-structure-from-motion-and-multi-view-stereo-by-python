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
import matplotlib.cm as cm

from BundleAdjustment import *
from scipy.optimize import least_squares
import time
import pdb
import math
from argparse import ArgumentParser

def example_plot(ax):
    ax.plot([1, 2])
    ax.set_xlabel('x-label', fontsize=next(fontsizes))
    ax.set_ylabel('y-label', fontsize=next(fontsizes))
    ax.set_title('Title', fontsize=next(fontsizes))

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


def TwoImage(img1, img2, des1, kp1, des2, kp2, pre_C, pre_R, K = None, calibs = None, P1 = None ,P2 = None):
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    trainPoints = cv2.KeyPoint_convert([kp1[matches[i].trainIdx] for i in range(len(matches))])
    queryPoints = cv2.KeyPoint_convert([kp2[matches[i].queryIdx] for i in range(len(matches))])
    pts1 = np.int32(trainPoints)
    pts2 = np.int32(queryPoints)
    
    #use opencv 
    F, mask = cv2.findFundamentalMat(trainPoints,queryPoints,cv2.FM_LMEDS)
    train_inliers = trainPoints[mask.ravel()==1]
    query_inliers = queryPoints[mask.ravel()==1]
    #use myself F
    #my_F, train_inliers, query_inliers, ind = RANSAC(pts1, pts2, max_iter_times = 1000) 
    
    
    
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(query_inliers.reshape(-1,1,2), 2, F)
    lines1 = lines1.reshape(-1,3)
    left_img_with_lines, _ = drawlines(img1.copy(),img2.copy(),lines1,train_inliers,query_inliers)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(train_inliers.reshape(-1,1,2), 1, F)
    lines2 = lines2.reshape(-1,3)
    right_img_with_lines, _ = drawlines(img2.copy(),img1.copy(),lines2,query_inliers,train_inliers)
    
    if(calibs == None):
        best_secondCamera_C, best_secondCamera_R, points, chosen_train_inliers, chosen_query_inliers = getCameraPos(train_inliers, query_inliers, F, pre_R, pre_C, K = K)
        return [left_img_with_lines, right_img_with_lines], best_secondCamera_C, best_secondCamera_R, chosen_train_inliers.astype(np.float64), chosen_query_inliers.astype(np.float64), points.astype(np.float64)
    else:
        return [left_img_with_lines, right_img_with_lines], train_inliers.astype(np.float64), query_inliers.astype(np.float64), getWorldPoints(train_inliers, query_inliers, P1, P2).astype(np.float64)
    
def getCameraPos(train_inliers, query_inliers, F, pre_R, pre_C, K = None):    

    E = getEssentialMatrix(K, F)
    U, D, Vh = getEssentialConfig(E)

    #second camera
    #C, R are relative to pre camera at [0,0,0]
    C1, R1 = CameraPosition1Config(U, Vh)
    C2, R2 = CameraPosition2Config(U, Vh)
    C3, R3 = CameraPosition3Config(U, Vh)
    C4, R4 = CameraPosition4Config(U, Vh)
    
    Cs = [C1,C2,C3,C4]
    Rs = [R1,R2,R3,R4]
    
    #relative to previous camera at exactly position
    print("[DEBUG]pre_R\n",pre_R)
    print("[DEBUG]pre_C\n",pre_C)
    for i, (C_diff, R_diff) in enumerate(zip(Cs, Rs)):
        Rs[i] = R_diff @ pre_R
        Cs[i] = C_diff @ pre_R + pre_C
        
    best_ct = 0
    best_secondCamera_C = None
    best_secondCamera_R = None
    best_idx = None
    points = None

    for i in range(1,4):
        P1, P2 = getCameraMatrix(K, Rs[i-1], Cs[i-1], Rs[i], Cs[i]) 
        temp_points = getWorldPoints(train_inliers, query_inliers, P1, P2)
        pts_in_front_of_C1, pts_in_front_of_C2, idx = CheckCheirality(temp_points, Cs[i-1], Rs[i-1], Cs[i], Rs[i])
        if(len(pts_in_front_of_C1) + len(pts_in_front_of_C2) > best_ct):
            best_ct = len(pts_in_front_of_C1) + len(pts_in_front_of_C2)
            best_secondCamera_C = Cs[i]
            best_secondCamera_R = Rs[i]
            best_idx = idx
            points = temp_points[idx]
    print("[DEBUG]R\n",best_secondCamera_R)
    print("[DEBUG]C\n",best_secondCamera_C)

    
    
    if(best_ct == 0):
        print("[DEBUG]best_ct==0")
        pdb.set_trace()
##########draw############           
    scale = 1
    print("[DEBUG]pre_C",pre_C)
    print("[DEBUG]pre_R",pre_R)
    print("[DEBUG]C",best_secondCamera_C)
    print("[DEBUG]R",best_secondCamera_R)
    ax = plt.axes(projection='3d')
    ax.scatter(scale * pts_in_front_of_C1[:,0], scale * pts_in_front_of_C1[:,1], scale * pts_in_front_of_C1[:,2], c="blue", cmap='viridis', linewidth=0.5);
    ax.scatter(scale * pts_in_front_of_C2[:,0], scale * pts_in_front_of_C2[:,1], scale * pts_in_front_of_C2[:,2], c="pink", cmap='viridis', linewidth=0.5);
    colors = cm.rainbow(np.linspace(0, 1, 2))
    
    ax.scatter(scale * pre_C[0], scale * pre_C[1], scale * pre_C[2], c=colors[0], linewidth=8, label='camera1');
    ax.scatter(scale * best_secondCamera_C[0], scale * best_secondCamera_C[1], scale * best_secondCamera_C[2], c=colors[1], linewidth=8, label='camera2');
    ax.legend()
    plt.show() 

#########################         
        
        
    return best_secondCamera_C, best_secondCamera_R, points.astype(np.float64), train_inliers[best_idx], query_inliers[best_idx] 


def main(args):
    files = []
    print("get img from" + args.img_dir+"/*."+args.img_type)
    for file in glob.glob(args.img_dir+"/*."+args.img_type):      
        files.append(file)
    
    calibs = None
    if(args.calib_dir!=None):
        calibs = []
        for calib in glob.glob(args.calib_dir+"/*.txt"):
            calibs.append(calib)
        calibs.sort()
        
    files.sort()

    #files = files[:2]
    
    print(files)
    
    imgs = []
    
    for file in files:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
        
        
    camera_params = []
    camera_indices = []
    points_3d = []
    points_2d = []
 
    C = {}
    R = {}
    pre_img = None
    pre_kp = None
    pre_des = None
    dist_coef = np.zeros((4,1))
    
    lines_imgs = []
    
    P = None
    if calibs != None:
        P = {}
        for ct,file in enumerate(calibs):
            with open(file) as f:
                P[ct] = np.zeros((3,4), dtype=np.float64)
                for i,line in enumerate(f.readlines()):
                    if i==0:
                        continue
                    else:
                        #print(np.array([float(w) for w in line.split()]))
                        P[ct][i-1,:] = np.array([float(w) for w in line.split()])
                              
    for img_ct, img in tqdm(enumerate(imgs), total = len(imgs)):
        #print(img_ct)
        if(img_ct == 0):
            orb = cv2.ORB_create(edgeThreshold=3)
            pre_kp, pre_des = orb.detectAndCompute(img,None)
            #first camera
            C[0] = np.array([0,0,0], dtype=np.float64)
            R[0] = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=np.float64)
        else:
            #Oriented FAST and Rotated BRIEF
            orb = cv2.ORB_create(edgeThreshold=3)

            # find the keypoints with ORB
            kp, des = orb.detectAndCompute(img,None)
            if calibs == None:
                lines_img, C[img_ct], R[img_ct], train_inliers, query_inliers, points3D = TwoImage(imgs[img_ct-1], img, pre_des, pre_kp, des, kp, C[img_ct-1] ,R[img_ct-1], K = K, calibs = None)
                #the first image hasn't been count
                if(img_ct == 1):
                    #add the first image rvec and tvec
                    #(_, rvec, tvec, _) = cv2.solvePnPRansac(points3D[:,0:3], train_inliers, K, dist_coef, cv2.SOLVEPNP_EPNP)            
                    tvec = -R[0].transpose()@C[0]
                    rvec, jacobian = cv2.Rodrigues(R[0])
                    #print(tvec)
                    #print(rvec)
                    camera_params.append([rvec[0][0], rvec[1][0], rvec[2][0], tvec[0], tvec[1], tvec[2], f1, 0, 0])
                    for i in range(len(points3D)):
                        points_3d.append([points3D[i,0], points3D[i,1], points3D[i,2]])
                        camera_indices.append(0)
                    for i in range(len(train_inliers)):
                         points_2d.append(train_inliers[i]) 

                tvec = -R[img_ct].transpose()@C[img_ct]
                rvec, jacobian = cv2.Rodrigues(R[img_ct])
                #(_, rvec, tvec, _) = cv2.solvePnPRansac(points3D[:,0:3], query_inliers, K, dist_coef, cv2.SOLVEPNP_EPNP)
                camera_params.append([rvec[0][0], rvec[1][0], rvec[2][0], tvec[0], tvec[1], tvec[2], f1, 0, 0])
                for i in range(len(points3D)):
                    points_3d.append([points3D[i,0], points3D[i,1], points3D[i,2]])
                    camera_indices.append(img_ct)              
                for i in range(len(query_inliers)):
                    points_2d.append(query_inliers[i])    
                
                
            else:
                try:
                    lines_img, train_inliers, query_inliers, points3D = TwoImage(imgs[img_ct-1], img, pre_des, pre_kp, des, kp, None, None, None, calibs, P[img_ct-1], P[img_ct])
                except Exception as e: 
                    print(e)
                    pdb.set_trace()
                for i in range(len(points3D)):
                    points_3d.append([points3D[i,0], points3D[i,1], points3D[i,2]])
                
            lines_imgs.append(lines_img)
            pre_kp = kp
            pre_des = des

    if calibs == None:
        camera_params = np.array(camera_params)
        camera_indices = np.array(camera_indices)
        print("len points:", len(points_3d))
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
        A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
        t0 = time.time()
        res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
        t1 = time.time()
        print("Optimization took {0:.0f} seconds".format(t1 - t0))

        camera_params = res.x[:n_cameras * 9].reshape((n_cameras, 9))
        points_3d = res.x[n_cameras * 9:].reshape((n_points, 3))
    else:
        points_3d = np.array(points_3d).reshape((-1,3))
    
    
    fig=plt.figure(figsize=(64, 64))
    columns = 6
    rows = math.ceil(len(lines_imgs) * 2 / 6 )
    for i, (left, right) in enumerate(lines_imgs):
        img = left
        fig.add_subplot(rows, columns, 2*i+1)
        plt.imshow(img,aspect='auto')
        
        img = right
        fig.add_subplot(rows, columns, 2*i+2)
        plt.imshow(img,aspect='auto')
    plt.show()
    
    
#     for i, (left, right) in enumerate(lines_imgs):
#         ax = plt.subplot(len(lines_imgs*2),2,2*i+1)
#         plt.imshow(left,aspect='auto')
#         ax = plt.subplot(len(lines_imgs*2),2,2*i+2)
#         plt.imshow(right,aspect='auto')
#     plt.show()
    
    
    print("points_3d points:",len(points_3d))
    Percentile = np.percentile(points_3d,[0,25,50,75,100],axis=0)
    IQR = Percentile[3] - Percentile[1]
    UpLimit = Percentile[3] + IQR*1.5
    DownLimit = Percentile[1] - IQR*1.5
    clean_points_3d = []
    for i in points_3d:
        if(i[0] >= DownLimit[0] and i[1] >= DownLimit[1] and i[2] >= DownLimit[2] and i[0] <= UpLimit[0] and i[1] <= UpLimit[1] and i[2] <= UpLimit[2]):
            clean_points_3d.append(i)
    
    clean_points_3d = np.array(clean_points_3d)
    print("clean_points_3d points:",len(clean_points_3d))
    
    scale = 1
    ax = plt.axes(projection='3d')
    ax.scatter(scale * clean_points_3d[:,0], scale * clean_points_3d[:,1], scale * clean_points_3d[:,2], c=clean_points_3d[:,2], cmap='viridis', linewidth=0.5);
    if calibs == None:
        colors = cm.rainbow(np.linspace(0, 1, len(C)))
        for i,c in zip(C,colors):
            ax.scatter(scale * C[i][0], scale * C[i][1], scale * C[i][2], c=c, linewidth=8, label='camera'+str(i+1));
        ax.legend()
    plt.show()

    ax = plt.axes(projection='3d')
    if calibs == None:
        colors = cm.rainbow(np.linspace(0, 1, len(C)))
        for i,c in zip(C,colors):
            ax.scatter(scale * C[i][0], scale * C[i][1], scale * C[i][2], c=c, linewidth=8, label='camera'+str(i+1));
        ax.legend()
    plt.show()
    #w,h = lines_imgs[0].shape

    
if __name__== "__main__":
    parser = ArgumentParser()
    parser.add_argument("-img_p", help="image directory", dest="img_dir", default=None)
    parser.add_argument("-calib_p", help="calib directory", dest="calib_dir", default=None)
    parser.add_argument("-t", help="image file type", dest="img_type", default="ppm")
    args = parser.parse_args()
    main(args)