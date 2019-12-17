#based on https://kknews.cc/zh-tw/news/l6kvoyz.html

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pdb
import multiprocessing
from multiprocessing import Process

def wrapper(func, id ,args, return_dict):
    print("in wrapper")
    return_dict[id] = func(*args)


def Match(desc1, desc2, threshold=0.5): 
    """For each corner point descriptor in the first image, select its match to second image using normalized cross correlation.""" 
    #pair-wise distance
    #d = -ones((len(desc1),len(desc2)))
    n_pixels = len(desc1[0])
    dist = np.zeros((len(desc1),len(desc2)))
    for i in tqdm(range(len(desc1)), total = len(desc1)):
        for j in range(len(desc2)): 
            d1 = (desc1[i]-np.mean(desc1[i]))/np.std(desc1[i]) 
            d2 = (desc2[j]-np.mean(desc2[j]))/np.std(desc2[j]) 
            try:
                ncc_value = sum(d1*d2)/(n_pixels - 1)
            except:
                print("[debug]Match")
                print(d1)
                print(d2)
                pdb.set_trace()
            if ncc_value > threshold: 
                dist[i,j] = ncc_value 
    ndx = np.argsort(-dist) 
    matchscores = ndx[:,0] 
    return matchscores

            
def MatchTwoSided(desc1,desc2,threshold=0.5): 
    """two sided symmetric version of match."""
    print("MatchTwoSided")
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    p1 = Process(target = wrapper, args=(Match, 0, (desc1, desc2, threshold), return_dict))
    p2 = Process(target = wrapper, args=(Match, 1, (desc2, desc1, threshold), return_dict))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    matches_12 = return_dict[0]
    matches_21 = return_dict[1]

    print('finish thread')
    print(len(matches_21))
    #matches_12 = Match(desc1, desc2, threshold) 
    #matches_21 = Match(desc2, desc1, threshold) 
    ndx_12 = np.where(matches_12 >= 0)[0] 
    
    # remove matches that are not symmetric 
    for n in ndx_12: 
        if matches_21[matches_12[n]] != n: 
            matches_12[n] = -1 
            
    return matches_12

def appendImages(im1,im2): 
    """Return a new image that appends that two images side-by-side.""" 
    #select the image with the fewest rows and fill in enough empty rows 
    print("appendImages")
    rows1 = im1.shape[0] 
    rows2 = im2.shape[0] 
    if rows1 < rows2: 
        im1 = np.concatenate((im1,zeros((rows2-rows1,im1.shape[1]))),axis=0) 
    elif rows1 < rows2: 
        im2 = np.concatenate((im2,zeros((rows1-rows2,im2.shape[1]))),axis=0) 
    return np.concatenate((im1,im2),axis=1)


def getMatches(im1, im2, locs1, locs2, matchscores, show_below=True): 
    """
    show a figure with lines joinging the accepted matches Input:
    im1,im2(images as arrays),
    locs1,locs2,(feature locations), 
    metachscores(as output from 'match'), 
    show_below(if images should be shown matches).
    """ 
    print("plot_matches")
    im3 = appendImages(im1, im2) 
    
    im3 = np.vstack((im3, im3)) 
    if show_below: 
        plt.imshow(im3) 
    cols1 = im1.shape[1] 
    src_pts = []
    dst_pts = []
    match_scores = []
    
    for i,m in enumerate(matchscores): 
        if m>0:
            if show_below: 
                plt.plot([locs1[i][1],locs2[m][1]+cols1],[locs1[i][0],locs2[m][0]],'c')
            src_pts.append([locs1[i][1],locs1[i][0]])
            dst_pts.append([locs2[m][1],locs2[m][0]])
            match_scores.append(m)
            
    if show_below: 
        plt.show()
        
    return np.int32(src_pts), np.int32(dst_pts), np.array(match_scores)

def getDescFeatures(image, filter_coords, wid=5): 
    """
    For each point return pixel values around the point using a neihborhood of 2*width+1.
    Important:
        coords[0] - wid should not be less than 0 either coords[1] - wid
        The image should be expand to at least (i+2*width+1, j+2*width+1 ,3) shape
    """ 
    print("getDescPatches")
    desc = []
    for coords in filter_coords: 
        patch = image[coords[0]-wid:coords[0]+wid+1, coords[1]-wid:coords[1]+wid+1] 
        desc.append(patch.flatten()) # use append to add new elements return desc
    return desc
                
def getHarrisPoints(input_img, debug = False):
    print("getHarrisPoints")
    img = input_img.copy()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    img[dst>0.01*dst.max()]=[0,0,255]
    if debug == True:
        plt.imshow(img)
        plt.show()
    
    temp = np.zeros(gray.shape)
    temp[dst>0.01*dst.max()] = 1
    xs,ys = np.where(temp==1)
    points = []
    for x,y in zip(xs,ys):
        points.append([x,y])
    return np.int32(points)
    #return np.float32(points)