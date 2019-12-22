import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from tqdm import tqdm
import cv2
import glob, os
import time
import pdb
import math
from scipy.optimize import least_squares
import matplotlib.cm as cm

def example_plot(ax):
    ax.plot([1, 2])
    ax.set_xlabel('x-label', fontsize=next(fontsizes))
    ax.set_ylabel('y-label', fontsize=next(fontsizes))
    ax.set_title('Title', fontsize=next(fontsizes))

def drawlines(img1,img2,lines,pts1,pts2):
    img1 = img1.copy()
    img2 = img2.copy()
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    (r,c,_) = img1.shape
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        try:
            #r[1] can be extremely small and will cause OverflowError
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



def getSequence(imgs):
    """
    input: list
        imgs                    
    Returns: list
        n-1 of imgs  
    """
    imgs_combination = []

    for i, img in enumerate(imgs):
        if(i!=0):
            imgs_combination.append([i-1,pre_img,i,img])
        pre_img = img
    return imgs_combination

def DebugShow(img1, img2, left_inliers, right_inliers, matches, kp1, kp2, lines1, lines2):
    print("DebugShow")
    #draw features on both images   
    fig=plt.figure(figsize=(64, 64))
    fig.suptitle('features on both images', fontsize=16)
    imageA = img1.copy()
    imageB = img2.copy()
    for i in left_inliers:
        imageA = cv2.circle(imageA , (int(i[0]), int(i[1])), 1, (0, 255, 0), 1)
    fig.add_subplot(1, 2, 1)
    plt.imshow(imageA)
    for i in right_inliers:
        imageB = cv2.circle(imageB , (int(i[0]), int(i[1])), 1, (0, 255, 0), 1)
    fig.add_subplot(1, 2, 2)
    plt.imshow(imageB)
    plt.show()
   
    #draw matches of both images
    img3 = cv2.drawMatches(imageA, kp1, imageB, kp2, matches, None, flags=2)
    plt.imshow(img3)
    plt.title("draw matches of both images")
    plt.show()

    #draw Epilines and features on both images
    left_img_with_lines, right_img = drawlines(img1,img2,lines1,left_inliers,right_inliers)    
    fig=plt.figure(figsize=(64, 64))
    fig.suptitle('draw Epilines on both images(first)')
    fig.add_subplot(1, 2, 1)
#     for i in left_inliers:
#         left = cv2.circle(left, (int(i[0]), int(i[1])), 1, (255, 0, 0), 10)
    plt.imshow(left_img_with_lines,aspect='auto')
    fig.add_subplot(1, 2, 2)
    plt.imshow(right_img,aspect='auto')
    plt.show()
    
    right_img_with_lines, left_img = drawlines(img2.copy(),img1.copy(),lines2,right_inliers,left_inliers)
    fig=plt.figure(figsize=(64, 64))
    fig.suptitle('draw Epilines on both images(second)')
    fig.add_subplot(1, 2, 1)
#     for i in right_inliers:
#         right = cv2.circle(right, (int(i[0]), int(i[1])), 1, (255, 0, 0), 10)
    plt.imshow(right_img_with_lines,aspect='auto')
    fig.add_subplot(1, 2, 2)
    plt.imshow(left_img,aspect='auto')
    plt.show()

def getORBFeatures(img1, img2, debug = False, return_F = False):
    """
    Returns:
        tuple(np.array(n,2), np.array(n,2), n)
    """
    
    #Oriented FAST and Rotated BRIEF
    orb = cv2.ORB_create(nfeatures=100000)#, scoreType=cv2.ORB_FAST_SCORE, fastThreshold=20)
    # find the keypoints with ORB
    kp1, des1 = orb.detectAndCompute(img1,None) #query image
    kp2, des2 = orb.detectAndCompute(img2,None) #train image
    
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
    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)#cv2.FM_LMEDS)#
    if F is not None:
        query_inliers = src_pts[mask.ravel()==1]       
        train_inliers = dst_pts[mask.ravel()==1]

    else:
        print("F is failed to found")
        if return_F is True:
            return None
        return None, None, 0
    
    lines1 = cv2.computeCorrespondEpilines(train_inliers.reshape(-1,1,2), 2, F)
    lines1 = lines1.reshape(-1,3)
        
    lines2 = cv2.computeCorrespondEpilines(query_inliers.reshape(-1,1,2), 1, F)
    lines2 = lines2.reshape(-1,3)

    
    if(debug):
        DebugShow(img1, img2, query_inliers, train_inliers, good_matches, kp1, kp2, lines1, lines2)

    if return_F is True:
        return F
    return query_inliers, train_inliers, len(train_inliers)#, left_img_with_lines, right_img_with_lines

def getProjectionMatrix(K, r, t):
    extrinsic = np.concatenate((r, t), axis=1)
    return K @ extrinsic

def traingulatePoints(P1,P2,query_inliers,train_inliers):
    return cv2.triangulatePoints(P1,P2,query_inliers.transpose(),train_inliers.transpose()).transpose()

def projectPoint(point, par_r, par_t, par_K):
    rvec, jacobian = cv2.Rodrigues(par_r)
    out, jacobian = cv2.projectPoints(point, rvec, par_t, par_K, None)
    return out.ravel()