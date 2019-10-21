import scipy.linalg
from scipy.optimize import lsq_linear
import numpy as np
import cv2

def LinearTriangulation(v1, v2, P1, P2):
    row1 = v1[0]*P1[2,:] - P1[1,:]
    row2 = v1[1]*P1[2,:] - P1[0,:]
    row3 = v2[0]*P2[2,:] - P2[1,:]
    row4 = v2[1]*P2[2,:] - P2[0,:]
    #row5 = np.array([1,1,1,1])
    A = np.array([row1,row2,row3,row4])
    b = np.array([0,0,0,0])
    _, _, Vh = scipy.linalg.svd(A)
    #(Hermitian) transpose back
    x = Vh.transpose()[:, -1];
    print(x)
    return x
    #res = lsq_linear(A, b, bounds=([-1000,-1000,-1000,1], [1000,1000,1000,1.0001]),lsmr_tol='auto')
    #return res.x

def getWorldPoints(train_inliers, query_inliers, P1, P2):
    points = []
    for (train_inlier, query_inlier) in zip(train_inliers, query_inliers):
        #points.append(LinearTriangulation(train_inlier, query_inlier, P1, P2))
        point = cv2.triangulatePoints(P1,P2,np.array([[train_inlier[0]], [train_inlier[1]]]),np.array([[query_inlier[0]], [query_inlier[1]]])).reshape(-1)
        if(point[-1] == 0):
            point = 0*point[:-1]
        else:
            point = point[:-1]/point[-1]
        points.append(point)
    return np.array(points)