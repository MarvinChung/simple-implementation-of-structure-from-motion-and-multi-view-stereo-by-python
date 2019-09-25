from scipy.linalg import lstsq
from scipy.optimize import lsq_linear
import numpy as np

def LinearTriangulation(v1, v2, P1, P2):
    row1 = v1[0]*P1[2,:] - P1[1,:]
    row2 = v1[1]*P1[2,:] - P1[0,:]
    row3 = v2[0]*P2[2,:] - P2[1,:]
    row4 = v2[1]*P2[2,:] - P2[0,:]
    #row5 = np.array([1,1,1,1])
    A = np.array([row1,row2,row3,row4])
    b = np.array([0,0,0,0])
    res = lsq_linear(A, b, bounds=([-1000,-1000,-1000,1], [1000,1000,1000,1.0001]),lsmr_tol='auto')
    return res.x

def getWorldPoints(train_inliers, query_inliers, P1, P2):
    points = []
    for (train_inlier, query_inlier) in zip(train_inliers, query_inliers):
        points.append(LinearTriangulation(train_inlier, query_inlier, P1, P2))
    return np.array(points)