import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2
import glob, os
from mpl_toolkits import mplot3d

from Normalized8pointsAlgo import RANSAC
from CameraConfig import *
#from Triangulation import *
import matplotlib.cm as cm

from BundleAdjustment import *
from QuickSets import GlobalSets
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
        try:
            #r[1] can be extremely small and will cause OverflowError
            if(r[1] > 0.01):
                x0,y0 = map(int, [0, -r[2]/r[1] ])
                x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
                img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
                img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
                img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
        except OverflowError:
            print(pts1)
            print(pts2)
            print("x0",x0)
            print("y0",y0)
            print("x1",x1)
            print("y1",y1)
            print("len(pts1):",len(pts1))
            print("len(pts2):",len(pts2))
            print(r[0])
            print(r[1])
            print(r[2])
            print(-r[2]/r[1])
            print(-(r[2]+r[0]*c)/r[1])
            
            raise OverflowError
                 
    return img1,img2


# def TwoImage(img1, img2, des1, kp1, des2, kp2, pre_C, pre_R, K = None):
#     # create BFMatcher object
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

#     # Match descriptors.
#     matches = bf.match(des1,des2)

#     # Sort them in the order of their distance.
#     matches = sorted(matches, key = lambda x:x.distance)

#     trainPoints = cv2.KeyPoint_convert([kp1[matches[i].trainIdx] for i in range(len(matches))])
#     queryPoints = cv2.KeyPoint_convert([kp2[matches[i].queryIdx] for i in range(len(matches))])
#     #pts1 = np.int32(trainPoints)
#     #pts2 = np.int32(queryPoints)
    
#     #use opencv 
#     F, mask = cv2.findFundamentalMat(trainPoints,queryPoints, cv2.FM_LMEDS)#cv2.FM_RANSAC)
#     train_inliers = trainPoints[mask.ravel()==1]
#     query_inliers = queryPoints[mask.ravel()==1]
#     #use myself F
#     #my_F, train_inliers, query_inliers, ind = RANSAC(pts1, pts2, max_iter_times = 1000) 
        
#     # Find epilines corresponding to points in right image (second image) and
#     # drawing its lines on left image
#     lines1 = cv2.computeCorrespondEpilines(query_inliers.reshape(-1,1,2), 2, F)
#     lines1 = lines1.reshape(-1,3)
#     left_img_with_lines, _ = drawlines(img1.copy(),img2.copy(),lines1,train_inliers,query_inliers)
    
#     # Find epilines corresponding to points in left image (first image) and
#     # drawing its lines on right image
#     lines2 = cv2.computeCorrespondEpilines(train_inliers.reshape(-1,1,2), 1, F)
#     lines2 = lines2.reshape(-1,3)
#     right_img_with_lines, _ = drawlines(img2.copy(),img1.copy(),lines2,query_inliers,train_inliers)
    
#     best_secondCamera_C, best_secondCamera_R, points, chosen_train_inliers, chosen_query_inliers = getCameraPos(train_inliers, query_inliers, F, pre_R, pre_C, K = K)
#     return [left_img_with_lines, right_img_with_lines], best_secondCamera_C, best_secondCamera_R, chosen_train_inliers.astype(np.float64), chosen_query_inliers.astype(np.float64), points.astype(np.float64), matches

def read_imgs(args):
    files = []
    print("read images from " + args.img_dir+"/*."+args.img_type)
    for file in glob.glob(args.img_dir+"/*."+args.img_type):      
        files.append(file)        
    files.sort()
    print(files)
    
    imgs = []
    for file in files:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
    return imgs

def read_pars(args):
    """
    name_par.txt: camera parameters. There is one line for each image. The format for each line is: "imgname.png k11 k12 k13 k21 k22 k23 k31 k32 k33 r11 r12 r13 r21 r22 r23 r31 r32 r33 t1 t2 t3". The projection matrix for that image is K*[R t]. The image origin is top-left, with x increasing horizontally, y vertically.
    
    Returns:
        tuple(dict(n,intrinsic np array), dict(n,rotation np array), dict(n, translation np array))
    """
    
    
    print("get parameters from" + args.par_path)
    f = open(args.par_path, "r")
    
    par_K = {}
    par_r = {}
    par_t = {}
    
    for i,line in enumerate(f.readlines()):
        if(i==0):
            img_ct = line.split()[0]
        else:
            tp = line.split()[1:]
            tp = [float(i) for i in tp]
            par_K[i-1] = np.array([[tp[0], tp[1], tp[2]],[tp[3],tp[4],tp[5]],[tp[6],tp[7],tp[8]]]).reshape(3,3)
            par_r[i-1] = np.array([[tp[9], tp[10], tp[11]],[tp[12],tp[13],tp[14]],[tp[15],tp[16],tp[17]]]).reshape(3,3)
            par_t[i-1] = np.array([tp[18], tp[19], tp[20]]).reshape(3,1)
    return par_K, par_r, par_t
def getCombination(imgs):
    """
    input: list
        imgs                    
    Returns: list
        C(n,2) of input imgs   
    """
    imgs_combination = []
    for idx_A, img_A in enumerate(imgs):
        for idx_B, img_B in enumerate(imgs):
            if(idx_A != idx_B):
                imgs_combination.append([idx_A,img_A,idx_B,img_B])
    return imgs_combination

def DebugShow(img1, img2, left_inliers, right_inliers, matches, kp1, kp2, left_img_with_lines, right_img_with_lines):
    
    #draw features on both images   
    fig=plt.figure(figsize=(64, 64))
    fig.suptitle('features on both images', fontsize=16)
    imageA = img1.copy()
    imageB = img2.copy()
    for i in left_inliers:
        imageA = cv2.circle(imageA , (int(i[0]), int(i[1])), 1, (255, 0, 0), 10)
    fig.add_subplot(1, 2, 1)
    plt.imshow(imageA)
    for i in right_inliers:
        imageB = cv2.circle(imageB , (int(i[0]), int(i[1])), 1, (255, 0, 0), 10)
    fig.add_subplot(1, 2, 2)
    plt.imshow(imageB)
    plt.show()
   
    #draw matches of both images
    img3 = cv2.drawMatches(imageA, kp1, imageB, kp2, matches, None, flags=2)
    plt.imshow(img3)
    plt.title("draw matches of both images")
    plt.show()

    #draw Epilines and features on both images
    fig=plt.figure(figsize=(64, 64))
    fig.suptitle('draw Epilines and features on both images')
    fig.add_subplot(1, 2, 1)
    left = left_img_with_lines.copy()
    for i in left_inliers:
        left = cv2.circle(left, (int(i[0]), int(i[1])), 1, (255, 0, 0), 10)
    plt.imshow(left,aspect='auto')
    fig.add_subplot(1, 2, 2)
    right = right_img_with_lines.copy()
    for i in right_inliers:
        right = cv2.circle(right, (int(i[0]), int(i[1])), 1, (255, 0, 0), 10)
    plt.imshow(right,aspect='auto')
    plt.show()



def getFeatures(img1, img2, debug):
    """
    Returns:
        tuple(np.array(n,2), np.array(n,2), n)
    """
    #Oriented FAST and Rotated BRIEF
    orb = cv2.ORB_create(edgeThreshold=3, nfeatures=1000)
    # find the keypoints with ORB
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    
    #in order to prevent error from flann.knnMatch
    des1 = np.float32(des1)
    des2 = np.float32(des2)
    
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # store all the good matches as per Lowe's ratio test.
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good_matches.append(m)             
    
    MIN_MATCH_COUNT = 8
       
    if len(good_matches) >= MIN_MATCH_COUNT:
        print("good_matches numbers:", len(good_matches))
        #Important: do not use np.int32, this will cause cv2.triangulation bad use and abort trap!!
        #src_pts = np.int32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,2)
        #dst_pts = np.int32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,2)
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,2)
    else:
        print("Not enough matches are found - %d/%d(at least 8 for findFundamentalMat to apply 8 points algorithm)" % (len(good_matches), MIN_MATCH_COUNT))
        return None, None, 0
        #raise ValueError('Not enough matches are found in getFeatures. Need at least 8 point for findFundamentalMat to apply 8 points algorithm')
        
        
    #use opencv findFundamentalMat input shape: 2xN/Nx2
    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_LMEDS)#cv2.FM_RANSAC)
    if F is not None:
        query_inliers = src_pts[mask.ravel()==1]       
        train_inliers = dst_pts[mask.ravel()==1]
    else:
        print("F is failed to found")
        return None, None, 0
    
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(train_inliers.reshape(-1,1,2), 2, F)
    lines1 = lines1.reshape(-1,3)
    left_img_with_lines, _ = drawlines(img1.copy(),img2.copy(),lines1,query_inliers,train_inliers)
    
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(query_inliers.reshape(-1,1,2), 1, F)
    lines2 = lines2.reshape(-1,3)
    right_img_with_lines, _ = drawlines(img2.copy(),img1.copy(),lines2,train_inliers,query_inliers)
    
    if(debug == "1"):
        DebugShow(img1, img2, query_inliers, train_inliers, good_matches, kp1, kp2, left_img_with_lines, right_img_with_lines)
    return query_inliers, train_inliers, len(train_inliers)#, left_img_with_lines, right_img_with_lines

def getProjectionMatrix(K, r, t):
    extrinsic = np.concatenate((r, t), axis=1)
    return K @ extrinsic

def traingulatePoints(P1,P2,query_inliers,train_inliers):
    return cv2.triangulatePoints(P1,P2,query_inliers.transpose(),train_inliers.transpose()).transpose()

def main2(args):
    scale = args.scale
    imgs = read_imgs(args)
    #3 dictionary
    par_K, par_r, par_t = read_pars(args)
    imgs_combination = getCombination(imgs)
    global_sets = GlobalSets()
    for (idx_A,img_A,idx_B,img_B) in tqdm(imgs_combination, total = len(imgs_combination)):
        query_inliers, train_inliers, inliers_n = getFeatures(img_A, img_B, args.debug)
        print("inliers_n:",inliers_n)
        if(inliers_n != 0):
            P1 = getProjectionMatrix(par_K[idx_A], par_r[idx_A], par_t[idx_A])
            P2 = getProjectionMatrix(par_K[idx_B], par_r[idx_B], par_t[idx_B])
            temp_points_3d = []

            #abort trap in cv2.triangulatePoints = =
            unnormalized_points = traingulatePoints(P1,P2,query_inliers,train_inliers)
            for query_inlier, train_inlier, un_point in zip(query_inliers, train_inliers, unnormalized_points):
                if(un_point[-1] == 0):
                    #don't use this point.
                    #There will be a lot of [0,0,0] but not reasonably inside global_sets
                    point = 0*un_point[:-1]
                else:
                    print("add")
                    
                    point = un_point[:-1]/un_point[-1]
                    if(idx_A >= 48):
                        print("oh no")
                        pdb.set_trace()
                    a_set = set([(idx_A, query_inlier[0], query_inlier[1]), (idx_B, train_inlier[0], train_inlier[1])])
                    global_sets.add(a_set, point)
                    temp_points_3d.append(point)
                    #print(global_sets.sets_list)
                    #print(len(global_sets.sets_list))
                    #input("")
                print(point)
            print(len(temp_points_3d))
            if(args.debug==1):
                ax = plt.axes(projection='3d')
                ax.set_title('without bundle adjustment temp points')
                ax.scatter(scale * np.array(temp_points_3d)[:,0], scale * np.array(temp_points_3d)[:,1], scale * np.array(temp_points_3d)[:,2], c=np.array(temp_points_3d)[:,2], cmap='viridis', linewidth=0.5);
                plt.show()
            
    print("len(global_sets_list)",len(global_sets.sets_list))
    
    #Bundle adjustment
    n_observations = global_sets.get_n_observations()
    n_cameras = len(imgs)
    n_points = len(global_sets.sets_list)
    
    camera_indices = np.empty(n_observations, dtype=int)
    point_indices = np.empty(n_observations, dtype=int)
    points_2d = np.empty((n_observations, 2))
    points_3d = np.empty((n_points, 3))
    
    legal_sets = []
    legal_points = []
    
    for ct,i in enumerate(global_sets.sets_list):
        if(global_sets.mask_list[ct] == 1):
            legal_sets.append(i)
            legal_points.append(global_sets.world_points[ct])
    
    ct = 0
    for point_index, (observation, world_point) in enumerate(zip(legal_sets,legal_points)):
        points_3d[point_index] = world_point
        
        print(world_point)
        #These point2ds are correspond to a 3d point
        for point2d_tuple in observation:
            #image_index is same as camera index
            #point2d_tuple: (image_index, x, y)
            print(point2d_tuple[0])
            if(point2d_tuple[0] >= n_cameras):
                pdb.set_trace()
            camera_indices[ct] = point2d_tuple[0]
            point_indices[ct] = point_index
            points_2d[ct] = [float(point2d_tuple[1]), float(point2d_tuple[2])]
            ct += 1
    
    
    ax = plt.axes(projection='3d')
    ax.set_title('without bundle adjustment')
    ax.scatter(scale * points_3d[:,0], scale * points_3d[:,1], scale * points_3d[:,2], c=points_3d[:,2], cmap='viridis', linewidth=0.5);
    plt.show()
    
    
#     print("points_3d points:",len(points_3d))
#     #remove extreme value
#     points_3d = np.array(points_3d) 
#     Percentile = np.percentile(points_3d,[0,25,50,75,100],axis=0)
#     IQR = Percentile[3] - Percentile[1]
#     UpLimit = Percentile[3] + IQR*1.5
#     DownLimit = Percentile[1] - IQR*1.5
    
#     clean_points_3d = []
#     for i in points_3d:
#         print(i)
#         print(DownLimit)
#         if(i[0] >= DownLimit[0] and i[1] >= DownLimit[1] and i[2] >= DownLimit[2] and i[0] <= UpLimit[0] and i[1] <= UpLimit[1] and i[2] <= UpLimit[2]):
#             clean_points_3d.append(i)
    
#     clean_points_3d = np.array(clean_points_3d)
#     print("clean_points_3d points:",len(clean_points_3d))
    
#     ax = plt.axes(projection='3d')
#     ax.set_title('with bundle adjustment(with cameara)')
#     ax.scatter(scale * clean_points_3d[:,0], scale * clean_points_3d[:,1], scale * clean_points_3d[:,2], c=clean_points_3d[:,2], cmap='viridis', linewidth=0.5);
#     plt.show()
            
    camera_params = np.empty((n_cameras, 9))
    for i in range(n_cameras):
        rvec, jacobian = cv2.Rodrigues(par_r[i])
        #rotation vector, translation vector, then a focal distance and two distortion parameters
        camera_params[i] = np.array([rvec[0][0],rvec[1][0],rvec[2][0],par_t[i][0][0],par_t[i][1][0],par_t[i][2][0],(par_K[i][0][0]+par_K[i][1][1])/2,0.0,0.0])
        print(rvec)
    
    n = 9 * n_cameras + 3 * n_points
    m = 2 * points_2d.shape[0]
    
    print("n_cameras: {}".format(n_cameras))
    print("n_points: {}".format(n_points))
    print("Total number of parameters: {}".format(n))
    print("Total number of residuals: {}".format(m))
    print(camera_params[0])
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    #f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
    t0 = time.time()
    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                    args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
    t1 = time.time()
    print("Optimization took {0:.0f} seconds".format(t1 - t0))

    #No need to refine camera_params
    #camera_params = res.x[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = res.x[n_cameras * 9:].reshape((n_points, 3))
    
    ax = plt.axes(projection='3d')
    ax.set_title('with bundle adjustment')
    ax.scatter(scale * points_3d[:,0], scale * points_3d[:,1], scale * points_3d[:,2], c=points_3d[:,2], cmap='viridis', linewidth=0.5);
    plt.show()
    
#     print("points_3d points:",len(points_3d))
#     #remove extreme value
#     Percentile = np.percentile(points_3d,[0,25,50,75,100],axis=0)
#     IQR = Percentile[3] - Percentile[1]
#     UpLimit = Percentile[3] + IQR*1.5
#     DownLimit = Percentile[1] - IQR*1.5
    
#     clean_points_3d = []
#     for i in points_3d:
#         if(i[0] >= DownLimit[0] and i[1] >= DownLimit[1] and i[2] >= DownLimit[2] and i[0] <= UpLimit[0] and i[1] <= UpLimit[1] and i[2] <= UpLimit[2]):
#             clean_points_3d.append(i)
    
#     clean_points_3d = np.array(clean_points_3d)
#     print("clean_points_3d points:",len(clean_points_3d))
    
#     ax = plt.axes(projection='3d')
#     ax.set_title('with bundle adjustment(with cameara)')
#     ax.scatter(scale * clean_points_3d[:,0], scale * clean_points_3d[:,1], scale * clean_points_3d[:,2], c=clean_points_3d[:,2], cmap='viridis', linewidth=0.5);

# def main(args):
    
#     scale = args.scale
    
#     imgs = read_imgs(args)
    
#     camera_params = []
#     camera_indices = []
#     points_3d = []
#     points_2d = []
#     point_indices = []
 
#     C = {}
#     R = {}
#     pre_img = None
#     pre_kp = None
#     pre_des = None
    
#     lines_imgs = []
                              
#     for img_ct, img in tqdm(enumerate(imgs), total = len(imgs)):
#         if(img_ct == 0):
#             orb = cv2.ORB_create(edgeThreshold=3)
#             pre_kp, pre_des = orb.detectAndCompute(img,None)
#             #first camera
#             C[0] = np.array([0,0,0], dtype=np.float32)
#             R[0] = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
#         else:
#             #Oriented FAST and Rotated BRIEF
#             orb = cv2.ORB_create(edgeThreshold=3)

#             # find the keypoints with ORB
#             kp, des = orb.detectAndCompute(img,None)
#             if calibs == None:
#                 lines_img, C[img_ct], R[img_ct], train_inliers, query_inliers, points3D, matches = TwoImage(imgs[img_ct-1], img, pre_des, pre_kp, des, kp, C[img_ct-1] ,R[img_ct-1], K = K)
                
#                 #the first image hasn't been count
#                 if(img_ct == 1):
#                     #add the first image rvec and tvec
#                     #(_, rvec, tvec, _) = cv2.solvePnPRansac(points3D[:,0:3], train_inliers, K, dist_coef, cv2.SOLVEPNP_EPNP)            
#                     #C = -R.transpose() * t
#                     #t = -R * C
#                     tvec = -R[0]@C[0]
#                     rvec, jacobian = cv2.Rodrigues(R[0])
#                     #print(tvec)
#                     #print(rvec)
#                     camera_params.append([rvec[0][0], rvec[1][0], rvec[2][0], tvec[0], tvec[1], tvec[2], f1, 0, 0])
#                     for i in range(len(points3D)):
#                         points_3d.append([points3D[i,0], points3D[i,1], points3D[i,2]])
#                     for i in range(len(train_inliers)):
#                         points_2d.append(train_inliers[i])
#                         camera_indices.append(0)

#                 tvec = -R[img_ct]@C[img_ct]
#                 rvec, jacobian = cv2.Rodrigues(R[img_ct])
#                 #(_, rvec, tvec, _) = cv2.solvePnPRansac(points3D[:,0:3], query_inliers, K, dist_coef, cv2.SOLVEPNP_EPNP)
#                 camera_params.append([rvec[0][0], rvec[1][0], rvec[2][0], tvec[0], tvec[1], tvec[2], f1, 0, 0])
#                 for i in range(len(points3D)):
#                     points_3d.append([points3D[i,0], points3D[i,1], points3D[i,2]])
                    
#                 for i in range(len(query_inliers)):
#                     points_2d.append(query_inliers[i])    
#                     camera_indices.append(img_ct)
                
                                
#                 ##########draw############ 
#                 if(args.debug != 0):
#                     fig=plt.figure(figsize=(64, 64))
#                     imageA = imgs[img_ct-1].copy()
#                     imageB = imgs[img_ct].copy()
#                     for i in train_inliers:
#                         imageA = cv2.circle(imageA , (int(i[0]), int(i[1])), 2, (255, 0, 0), 20)
#                     fig.add_subplot(1, 2, 1)
#                     plt.imshow(imageA)
#                     for i in query_inliers:
#                         imageB = cv2.circle(imageB , (int(i[0]), int(i[1])), 2, (255, 0, 0), 20)
#                     fig.add_subplot(1, 2, 2)
#                     plt.imshow(imageB)
#                     plt.show()
#                     img3 = cv2.drawMatches(imageA, pre_kp, imageB, kp, matches, None, flags=2)
#                     plt.imshow(img3)
#                     plt.show()

#                     fig=plt.figure(figsize=(64, 64))
#                     fig.add_subplot(1, 2, 1)
#                     left = lines_img[0].copy()
#                     for i in train_inliers:
#                         left = cv2.circle(left, (int(i[0]), int(i[1])), 2, (255, 0, 0), 20)
#                     plt.imshow(left,aspect='auto')

#                     fig.add_subplot(1, 2, 2)
#                     right = lines_img[1].copy()
#                     for i in query_inliers:
#                         right = cv2.circle(right, (int(i[0]), int(i[1])), 2, (255, 0, 0), 20)
#                     plt.imshow(right,aspect='auto')
#                     plt.show()

#                     ax = plt.axes(projection='3d')
#                     ax.set_title("debug add")
#                     ax.scatter(scale * np.array(points_3d)[:,0], scale * np.array(points_3d)[:,1], scale * np.array(points_3d)[:,2], c="blue", cmap='viridis', linewidth=0.5);
#                     colors = cm.rainbow(np.linspace(0, 1, len(C)))
#                     for ct, c in enumerate(colors):
#                         ax.scatter(scale * C[ct][0], scale * C[ct][1], scale * C[ct][2], c=c, linewidth=8, label='camera'+str(ct+1));
#                     ax.legend()
#                     plt.show() 
#                 ######################## 
                
#             else:
#                 try:
#                     lines_img, train_inliers, query_inliers, points3D = TwoImage(imgs[img_ct-1], img, pre_des, pre_kp, des, kp, None, None, None, calibs, P[img_ct-1], P[img_ct])
#                 except Exception as e: 
#                     print(e)
#                     pdb.set_trace()
#                 for i in range(len(points3D)):
#                     points_3d.append([points3D[i,0], points3D[i,1], points3D[i,2]])
                
#             lines_imgs.append(lines_img)
#             pre_kp = kp
#             pre_des = des

#     #points_3d = np.array(points_3d).reshape((-1,3))
    
#     fig=plt.figure(figsize=(64, 64))
#     columns = math.ceil(math.pow(len(2 * lines_imgs), 1/2))
#     rows = math.ceil(len(lines_imgs) * 2 / columns )
#     for i, (left, right) in enumerate(lines_imgs):
#         img = left
#         fig.add_subplot(rows, columns, 2*i+1)
#         plt.imshow(img,aspect='auto')
        
#         img = right
#         fig.add_subplot(rows, columns, 2*i+2)
#         plt.imshow(img,aspect='auto')
#     plt.show()
    

#     ax = plt.axes(projection='3d')
#     ax.set_title('without bundle adjustment')
#     ax.scatter(scale * np.array(points_3d)[:,0], scale * np.array(points_3d)[:,1], scale * np.array(points_3d)[:,2], c=np.array(points_3d)[:,2], cmap='viridis', linewidth=0.5);
#     if calibs == None:
#         colors = cm.rainbow(np.linspace(0, 1, len(C)))
#         for ct, c in enumerate(colors):
#             ax.scatter(scale * C[ct][0], scale * C[ct][1], scale * C[ct][2], c=c, linewidth=8, label='camera'+str(ct+1));
#         ax.legend()
#     plt.show()
    
#     camera_params = np.array(camera_params)
#     camera_indices = np.array(camera_indices)
#     print("len points:", len(points_3d))
#     point_indices = np.array(range(len(points_3d)))
#     #point_indices = np.array(point_indices)
#     points_3d = np.array(points_3d)
#     points_2d = np.array(points_2d)
#     n_cameras = camera_params.shape[0]
#     n_points = points_3d.shape[0]

#     n = 9 * n_cameras + 3 * n_points
#     m = 2 * points_2d.shape[0]

#     print("n_cameras: {}".format(n_cameras))
#     print("n_points: {}".format(n_points))
#     print("Total number of parameters: {}".format(n))
#     print("Total number of residuals: {}".format(m))
#     print(camera_params[0])
#     x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
#     #f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)
#     A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
#     t0 = time.time()
#     res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
#                     args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
#     t1 = time.time()
#     print("Optimization took {0:.0f} seconds".format(t1 - t0))

#     camera_params = res.x[:n_cameras * 9].reshape((n_cameras, 9))
#     points_3d = res.x[n_cameras * 9:].reshape((n_points, 3))
    
    
    
# #     for i, (left, right) in enumerate(lines_imgs):
# #         ax = plt.subplot(len(lines_imgs*2),2,2*i+1)
# #         plt.imshow(left,aspect='auto')
# #         ax = plt.subplot(len(lines_imgs*2),2,2*i+2)
# #         plt.imshow(right,aspect='auto')
# #     plt.show()
    
    
#     print("points_3d points:",len(points_3d))
#     #remove extreme value
#     Percentile = np.percentile(points_3d,[0,25,50,75,100],axis=0)
#     IQR = Percentile[3] - Percentile[1]
#     UpLimit = Percentile[3] + IQR*1.5
#     DownLimit = Percentile[1] - IQR*1.5
    
#     clean_points_3d = []
#     for i in points_3d:
#         if(i[0] >= DownLimit[0] and i[1] >= DownLimit[1] and i[2] >= DownLimit[2] and i[0] <= UpLimit[0] and i[1] <= UpLimit[1] and i[2] <= UpLimit[2]):
#             clean_points_3d.append(i)
    
#     clean_points_3d = np.array(clean_points_3d)
#     print("clean_points_3d points:",len(clean_points_3d))
    
#     ax = plt.axes(projection='3d')
#     ax.set_title('with bundle adjustment(with cameara)')
#     ax.scatter(scale * clean_points_3d[:,0], scale * clean_points_3d[:,1], scale * clean_points_3d[:,2], c=clean_points_3d[:,2], cmap='viridis', linewidth=0.5);
    
#     bundle_C = []
    
#     colors = cm.rainbow(np.linspace(0, 1, len(camera_params)))
#     #camera_params.append([rvec[0][0], rvec[1][0], rvec[2][0], tvec[0], tvec[1], tvec[2], f1, 0, 0])
#     for ct, (i,c) in enumerate(zip(camera_params,colors)):
#         #C = -R.transpose() * t
#         cam_pos = -cv2.Rodrigues(np.array([i[0],i[1],i[2]]))[0] @ np.array([[i[4]], [i[5]], [i[6]]])
#         bundle_C.append(cam_pos)
#         ax.scatter(scale * cam_pos[0], scale * cam_pos[1], scale * cam_pos[2], c=c, linewidth=8, label='camera'+str(ct+1));
#     ax.legend()
#     plt.show()

#     ax = plt.axes(projection='3d')
#     ax.set_title('with bundle adjustment(without cameara)')
#     ax.scatter(scale * clean_points_3d[:,0], scale * clean_points_3d[:,1], scale * clean_points_3d[:,2], c=clean_points_3d[:,2], cmap='viridis', linewidth=0.5);
#     plt.show()
    
#     print("non bundle:C\n",C)
#     print("bundle:C\n",bundle_C)


if __name__== "__main__":
    parser = ArgumentParser()
    parser.add_argument("-img_p", help="image directory", dest="img_dir", default=None)
    parser.add_argument("-par_p", help="parameter path", dest="par_path", default=None)
    parser.add_argument("-t", help="image file type", dest="img_type", default="ppm")
    parser.add_argument("-scale", help="scale", dest="scale", default=1)
    parser.add_argument("-debug", help="debug mode on", dest="debug", default=0)
    args = parser.parse_args()
    try:
        main2(args)
    except RuntimeError:
        print("")