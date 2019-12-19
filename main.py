import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2
import glob, os
from mpl_toolkits import mplot3d

#from Normalized8pointsAlgo import RANSAC
from CameraConfig import *
#from Triangulation import *
import matplotlib.cm as cm

from BundleAdjustment import *
#from QuickSets import GlobalSets
from GlobalSet import GlobalSet
from scipy.optimize import least_squares
import time
import pdb
import math
from argparse import ArgumentParser
from HarrisFeatures import *

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
    
    

                        
def StructureFromMotion(imgs, global_set, args, MIN_REPROJECTION_ERROR=0.5):  
    
    if args.isSeq == 1:
        imgs_combination = getSequence(imgs)
    else:
        imgs_combination = getCombination(imgs)

    #3 dictionary
    par_K, par_r, par_t = read_pars(args)
    
    for ct, (idx_A,img_A,idx_B,img_B) in enumerate(tqdm(imgs_combination, total = len(imgs_combination))):
        query_inliers, train_inliers, inliers_n = getORBFeatures(img_A, img_B, args.debug)
        print("inliers_n:",inliers_n)
        if(inliers_n != 0):
            P1 = getProjectionMatrix(par_K[idx_A], par_r[idx_A], par_t[idx_A])
            P2 = getProjectionMatrix(par_K[idx_B], par_r[idx_B], par_t[idx_B])
            #temp_global_set = GlobalSet()

            #abort trap in cv2.triangulatePoints if query_inliers or train_inliers are type int
            points = traingulatePoints(P1,P2,query_inliers,train_inliers) 
#{
#             ##normalized however get strange results
#             normalized query inliers and train_inliers in normalized image coordinates(-1 ~ 1)
#             normalized_query_inliers = cv2.undistortPoints(query_inliers, par_K[idx_A], None).reshape(-1,2)
#             normalized_train_inliers = cv2.undistortPoints(train_inliers, par_K[idx_B], None).reshape(-1,2)            
#             unnormalized_points = traingulatePoints(P1,P2,normalized_query_inliers,normalized_train_inliers)
#   
#             for query_inlier, train_inlier, un_point in zip(normalized_query_inliers, normalized_train_inliers, unnormalized_points):
#}
             
            for query_inlier, train_inlier, un_point in zip(query_inliers, train_inliers, points):
                if(un_point[-1] == 0):
                    #don't use this point.
                    #There will be a lot of [0,0,0] but not reasonably inside global_set
                    point = 0*un_point[:-1]
                else:                    
                    point = un_point[:-1]/un_point[-1]
                    print(np.linalg.norm(projectPoint(point, par_r[idx_A], par_t[idx_A], par_K[idx_A]) - query_inlier))
                    print(np.linalg.norm(projectPoint(point, par_r[idx_B], par_t[idx_B], par_K[idx_B]) - train_inlier))
                    if (np.linalg.norm(projectPoint(point, par_r[idx_A], par_t[idx_A], par_K[idx_A]) - query_inlier) > MIN_REPROJECTION_ERROR or np.linalg.norm(projectPoint(point, par_r[idx_B], par_t[idx_B], par_K[idx_B]) - train_inlier) > MIN_REPROJECTION_ERROR):
                        continue
                    
                    a_list = [(idx_A, query_inlier[0], query_inlier[1]), (idx_B, train_inlier[0], train_inlier[1])]
                    global_set.add2pts(a_list, point)
                    
            global_set.show_list()
            #uncomment will make it iterative bundle adjusmtnet and update global_set world points 
            #DrawPointClouds(global_set, ct+2, scale, par_K, par_r, par_t, debug = args.debug, show = False)
    
    DrawPointClouds(global_set, len(imgs), args.scale, par_K, par_r, par_t, debug = args.debug, show = True)
                

import heapq

class MyMatchHeap(object):
    #add negative for max heap
    def __init__(self, initial=None, key=lambda x:(-x.ncc_score, x.src_point[0], x.src_point[1], x.dst_point[0], x.dst_point[1])):
        self.key = key
        if initial:
            self._data = [(key(item), item) for item in initial]
            heapq.heapify(self._data)
        else:
            self._data = []

    def push(self, item):
        heapq.heappush(self._data, (self.key(item), item))

    def pop(self):
        #[1] for return item
        return heapq.heappop(self._data)[1]

    def size(self):
        return len(self._data)
class MyMatch(object):
    def __init__(self, src_point, dst_point, ncc_score):
        self.ncc_score = ncc_score
        self.src_point = src_point
        self.dst_point = dst_point

def ctNcc(desc1, desc2):
    n_pixels = len(desc1)
    d1 = (desc1-np.mean(desc1))/np.std(desc1) 
    d2 = (desc2-np.mean(desc2))/np.std(desc2) 
    return sum(d1*d2)/(n_pixels - 1)        

def DensePointsWithMVS(imgs, args):
    par_K, par_r, par_t = read_pars(args)
    debug = args.debug
    wid = 5
    harris_features = []
    harris_tables = []
    cutoff = 2
    # expand_base = wid+1


    for img in imgs[:cutoff]:
        #expand space of the image for wid
        # expand_img = np.zeros((img.shape[0] + 2 * expand_base, img.shape[1]+ 2 * expand_base, img.shape[2]), dtype='uint8')
        # for i in range(expand_base, img.shape[0]):
        #     for j in range(expand_base, img.shape[1]):
        #         expand_img[i,j,:] = img[i-expand_base, j-expand_base, :]
        # print(img.shape)
        # img = expand_img
        # print(img.shape)
        harris_pts = getHarrisPoints(img, debug = debug)
        harris_features.append(harris_pts)

        #for P1 and P2 and opencv : The image origin is top-left, with x increasing horizontally, y vertically
        #Therefore make image as size as img.shape[1], img.shape[0]
        harris_table = np.full((img.shape[1],img.shape[0]), False)
        for pt in harris_pts:
            harris_table[pt[0], pt[1]] = True
        harris_tables.append(harris_table)
    
    threshold = 0.5
    scale = args.scale

    refined_src_features = []   
    refined_dst_features = []
    
    for idx_A, img in enumerate(imgs[:cutoff]):
        filtered_coords1 = harris_features[idx_A]
        for idx_B, cmp_img in enumerate(imgs[:cutoff]):
            print(idx_A, idx_B)
            if(idx_A != idx_B):

                filtered_coords2 = harris_features[idx_B]

                #for P1 and P2 and opencv : The image origin is top-left, with x increasing horizontally, y vertically
                #Therefore make image as size as img.shape[1], img.shape[0]
                src_is_seen = np.full((img.shape[1],img.shape[0]), False)
                dst_is_seen = np.full((img.shape[1],img.shape[0]), False)
                 
                #use in method1 and method2               
                d1 = getDescFeatures(img, filtered_coords1, wid) 
                d2 = getDescFeatures(cmp_img, filtered_coords2, wid) 
                
                #method1
                #the output matches have a threshold for NCC score which ensure photo consistency
                
                matches = MatchTwoSided(d1, d2, threshold = threshold)
                
                src_pts, dst_pts, ncc_scores = getMatches(img, cmp_img, filtered_coords1, filtered_coords2, matches, show_below=debug)

                F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)
                
                heap = MyMatchHeap()
                if F is not None:
                    src_inliers = src_pts[mask.ravel()==1]       
                    dst_inliers = dst_pts[mask.ravel()==1]
                    ncc_inliers = ncc_scores[mask.ravel()==1]
                    for s,d,c in zip(src_inliers, dst_inliers, ncc_inliers):
                        src_is_seen[s[0],s[1]] = True
                        dst_is_seen[d[0],d[1]] = True
                        # print(s,d,c)
                        # s array([447,  22], dtype=int32)
                        # d array([443,  16], dtype=int32)
                        # c 1
                        heap.push(MyMatch(s,d,c))
                        
                    if debug:
                        lines = cv2.computeCorrespondEpilines(src_inliers.reshape(-1,1,2), 1, F).reshape(-1,3)
                        cmp_img_with_lines, right = drawlines(cmp_img, img, lines, dst_inliers, src_inliers)                  
                        fig=plt.figure(figsize=(64, 64))
                        fig.suptitle('draw Epilines on both images(first)')
                        fig.add_subplot(1,2,1)
                        plt.imshow(cmp_img_with_lines, aspect='auto')
                        fig.add_subplot(1,2,2)
                        plt.imshow(right, aspect='auto')
                        plt.show()
                else:
                    print("F is none")
                    raise RuntimeError
                
                #end method1    

                """
                #method2
                F = getORBFeatures(img, cmp_img, debug = debug, return_F = True)

                heap = MyMatchHeap()
                lines = cv2.computeCorrespondEpilines(filtered_coords1, 1, F).reshape(-1,3)

                if debug:
                    cmp_img_with_lines, right = drawlines(cmp_img, img, lines, filtered_coords2, filtered_coords1)                  
                    fig=plt.figure(figsize=(64, 64))
                    fig.suptitle('draw Epilines on both images(first)')
                    fig.add_subplot(1,2,1)
                    plt.imshow(cmp_img_with_lines, aspect='auto')
                    fig.add_subplot(1,2,2)
                    plt.imshow(right, aspect='auto')
                    plt.show()


                for line, left_pt in zip(lines, filtered_coords1):
                    #temp_heap find the best ncc value of the src point and another dst point on the corresponding epipolar line
                    temp_heap = MyMatchHeap()
                    for i in range(-2,3,1):
                        a = line[0]
                        b = line[1]
                        c = line[2]
                        for x in range(cmp_img.shape[1]):
                            #i: up down 2 pixel
                            y = int(round((-a*x-c)/b + i))

                            if y < cmp_img.shape[0] and y >= 0 and dst_is_seen[x,y] == False :
                                if  harris_tables[idx_B][x,y] == True:
                                    left_des = getDescFeatures(img, [left_pt], wid=wid)[0]
                                    right_des = getDescFeatures(cmp_img, [np.array([x,y])], wid=wid)[0]
                                    if type(left_des) != type(None) and type(right_des) != type(None) and left_des.shape == right_des.shape:
                                        ncc_value = ctNcc(left_des, right_des)
                                        #print("temp heap value:",ncc_value)
                                        temp_heap.push(MyMatch(left_pt, np.array([x,y]), ncc_value))
                    if(temp_heap.size()!=0):
                        match_obj = temp_heap.pop()
                        # src_point and dst_point is match, therefore don't seen them again
                        if(src_is_seen[match_obj.src_point[0],match_obj.src_point[1]] == True):
                            print("wtf src")
                            input("")
                        if(dst_is_seen[match_obj.dst_point[0],match_obj.dst_point[1]] == True):
                            print("wtf dst")
                            input("")
                        src_is_seen[match_obj.src_point[0],match_obj.src_point[1]] = True
                        dst_is_seen[match_obj.dst_point[0],match_obj.dst_point[1]] = True
                        heap.push(match_obj)
                #end method 3
                """    
                while(heap.size() != 0):
                    print(np.count_nonzero(src_is_seen))
                    print(np.count_nonzero(dst_is_seen))
                    print(src_is_seen.shape)
                    print(dst_is_seen.shape)
                    print("heap size:", heap.size())
                    match_obj = heap.pop()
                    src_point = match_obj.src_point
                    dst_point = match_obj.dst_point
                    
                    refined_src_features.append(src_point.astype('float32'))
                    refined_dst_features.append(dst_point.astype('float32'))
                    
                    src_min_x = max(0,src_point[0] - wid)
                    src_max_x = min(img.shape[1]-1,src_point[0] + wid)
                    src_min_y = max(0,src_point[1] - wid)
                    src_max_y = min(img.shape[0]-1,src_point[1] + wid)
                    
                    dst_min_x = max(0,dst_point[0] - wid)
                    dst_max_x = min(cmp_img.shape[1]-1,dst_point[0] + wid)
                    dst_min_y = max(0,dst_point[1] - wid)
                    dst_max_y = min(cmp_img.shape[0]-1,dst_point[1] + wid)
                    """
                    #Add new potential matches in their immediate spatial neighborhood into heap
                    for i in range(src_min_x, src_max_x):
                        for j in range(src_min_y, src_max_y):
                            if(src_is_seen[i,j] == False):
                                #temp_heap find the best ncc value of the src point and dst point of this patch 
                                temp_heap = MyMatchHeap()
                                for k in range(dst_min_x, dst_max_x):
                                    for l in range(dst_min_y, dst_max_y):
                                        if(dst_is_seen[k,l] == False):
                                            left_des = getDescFeatures(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), [np.array([i,j])], wid=wid)[0]
                                            right_des = getDescFeatures(cv2.cvtColor(cmp_img,cv2.COLOR_BGR2GRAY), [np.array([k,l])], wid=wid)[0]
                                            if type(left_des) != type(None) and type(right_des) != type(None) and left_des.shape == right_des.shape:
                                                ncc_value = ctNcc(left_des, right_des)
                                                #print("ncc_value",ncc_value)
                                                if (ncc_value == math.nan):
                                                    pdb.set_trace()
                                                if(1 - ncc_value > 0.5):
                                                    temp_heap.push(MyMatch(np.array([i,j]),np.array([k,l]), ncc_value))
                                if temp_heap.size() != 0:
                                    match_obj = temp_heap.pop()

                                    if(src_is_seen[match_obj.src_point[0],match_obj.src_point[1]] == True):
                                        print("wtf src")
                                        input("")
                                    if(dst_is_seen[match_obj.dst_point[0],match_obj.dst_point[1]] == True):
                                        print("wtf dst")
                                        input("")

                                    # src_point and dst_point is match, therefore don't seen them again
                                    src_is_seen[match_obj.src_point[0],match_obj.src_point[1]] = True
                                    dst_is_seen[match_obj.dst_point[0],match_obj.dst_point[1]] = True
                                    heap.push(match_obj)
                    #Only construct a point in a patch
                    for i in range(src_min_x, src_max_x):
                        for j in range(src_min_y, src_max_y):
                            for k in range(dst_min_x, dst_max_x):
                                    for l in range(dst_min_y, dst_max_y):
                                        src_is_seen[i,j] = True
                                        dst_is_seen[k,l] = True
                    """                            
                P1 = getProjectionMatrix(par_K[idx_A], par_r[idx_A], par_t[idx_A])
                P2 = getProjectionMatrix(par_K[idx_B], par_r[idx_B], par_t[idx_B])
                
                #abort trap in cv2.triangulatePoints if query_inliers or train_inliers are type int
                
                points = traingulatePoints(P1, P2, np.array(refined_src_features), np.array(refined_dst_features))
                #points = traingulatePoints(P1, P2, np.concatenate((np.array(refined_src_features),np.array([1 for i in range(len(refined_src_features))]).reshape(-1,1)),axis=1), np.concatenate((np.array(refined_dst_features),np.array([1 for i in range(len(refined_dst_features))]).reshape(-1,1)),axis=1)) 
                ax = plt.axes(projection='3d')
                ax.set_title('all images MVS')
                ax.scatter(scale * np.array(points)[:,0], scale * np.array(points)[:,1], scale * np.array(points)[:,2], c=np.array(points)[:,2], cmap='viridis', linewidth=0.1);
                plt.show()
                                    
                                    
                

def DrawPointClouds(global_set, n_cameras, scale, par_K, par_r, par_t, debug = False, show = False, use_BundleAdjustment = True):
    """
    n_camera is same as len(imgs) 
    """
    
    #Bundle adjustment
    n_observations, n_world_points, legal_sets = global_set.getInfo()
    print("len:",n_world_points)
    n_points= n_world_points
    
    camera_indices = np.empty(n_observations, dtype=int)
    point_indices = np.empty(n_observations, dtype=int)
    points_2d = np.empty((n_observations, 2))
    points_3d = np.empty((n_points, 3))    
    
    draw_points = []
    draw_color = []
    
    ct = 0
    for point_index, legal_set in enumerate(legal_sets):
        points_3d[point_index] = np.asarray(legal_set.world_point)
        
        draw_points.append(list(legal_set.world_point))
        draw_color.append('blue')
        #These point2ds are correspond to a 3d point
        for point2d_tuple in legal_set.point2d_list:
            #image_index is same as camera index
            #point2d_tuple: (image_index, x, y)
            #print(point2d_tuple[0])
            if(point2d_tuple[0] >= n_cameras):
                print("out of bound")
                pdb.set_trace()
            camera_indices[ct] = point2d_tuple[0]
            point_indices[ct] = point_index
            points_2d[ct] = [float(point2d_tuple[1]), float(point2d_tuple[2])]
            ct += 1
            
    assert(ct == n_observations)
    
    #add camera position
    if show == True:
        # for i in range(len(par_t)):
        #     draw_points.append(-(par_r[i].transpose()@par_t[i]).ravel())
        #     draw_color.append('pink')

        ax = plt.axes(projection='3d')
        ax.set_title('without bundle adjustment')
        ax.scatter(scale * np.array(draw_points)[:,0], scale * np.array(draw_points)[:,1], scale * np.array(draw_points)[:,2], c=draw_color, cmap='viridis', linewidth=0.1);

        plt.show()
    
    if use_BundleAdjustment == True:
        camera_params = np.empty((n_cameras, 11))
        for i in range(n_cameras):
            rvec, jacobian = cv2.Rodrigues(par_r[i])
            #rotation vector, translation vector, then a focal distance and two distortion parameters
            #Note that the images have been corrected to remove radial distortion.
            camera_params[i] = np.array([rvec[0][0],rvec[1][0],rvec[2][0],par_t[i][0][0],par_t[i][1][0],par_t[i][2][0],(par_K[i][0][0]+par_K[i][1][1])/2,0.0,0.0,par_K[i][0][2],par_K[i][1][2]])

        n = 11 * n_cameras + 3 * n_points
        m = 2 * points_2d.shape[0]



    #     for point_index, legal_set in enumerate(legal_sets[:5]):
    #         points_3d[point_index] = np.asarray(legal_set.world_point)

    #         print("3d point",points_3d[point_index])

    #         #These point2ds are correspond to a 3d point
    #         for point2d_tuple in legal_set.point2d_list:
    #             P = getProjectionMatrix(par_K[point2d_tuple[0]], par_r[point2d_tuple[0]], par_t[point2d_tuple[0]])
    #             print(point2d_tuple)
    #             temp = P@np.concatenate((points_3d[point_index],np.array([1])), axis=0).reshape(4,-1)
    #             print("project to",temp[:2]/temp[-1])

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
        #camera_params = res.x[:n_cameras * 11].reshape((n_cameras, 11))
        points_3d = res.x[n_cameras * 11:].reshape((n_points, 3))
        draw_points = list(points_3d)
        draw_color = ['blue' for i in range(len(draw_points))]
        if show == True:
            #add camera position
            # for i in range(n_cameras):
            #     refine_par_t = res.x[i*11+3:i*11+6]
            #     refine_par_r, jacobian = cv2.Rodrigues(res.x[i*11:i*11+3])
            #     draw_points.append(-(refine_par_r.transpose() @ refine_par_t.reshape(3,-1)).ravel())
            #     draw_color.append('pink')

            ax = plt.axes(projection='3d')
            ax.set_title('with bundle adjustment')
            ax.scatter(scale * np.array(draw_points)[:,0], scale * np.array(draw_points)[:,1], scale * np.array(draw_points)[:,2], c=draw_color, cmap='viridis', linewidth=0.1);
            plt.show()
            #print("bundle points")
            #print(points_3d)

#         for i in range(n_cameras):
#             print("original camera_pars")
#             print(camera_params[i])
#             print("after")
#             print(res.x[i*11:i*11+11])

        #update_global_set
        for point_index, legal_set in enumerate(legal_sets):
            legal_set.world_point = points_3d[point_index]
            
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


def main(args, threshold = 0.01, MIN_REPROJECTION_ERROR = 0.3):
    scale = args.scale
    imgs = read_imgs(args)

        
    global_set = GlobalSet(threshold = threshold)    
        
    #StructureFromMotion(imgs, global_set, args, MIN_REPROJECTION_ERROR)
    DensePointsWithMVS(imgs, args) 

if __name__== "__main__":
    parser = ArgumentParser()
    parser.add_argument("-img_p", help="image directory", dest="img_dir", default=None)
    parser.add_argument("-par_p", help="parameter path", dest="par_path", default=None)
    parser.add_argument("-t", help="image file type", dest="img_type", default="ppm")
    parser.add_argument("-scale", help="scale", dest="scale", default=1, type=float)
    parser.add_argument("--debug", help="debug mode on", dest="debug", action='store_true')
    parser.add_argument("-Sequence", help="", dest="isSeq", default=1, type=int)
    args = parser.parse_args()
    try:
        main(args)
    except RuntimeError:
        print("")