import scipy.linalg
from scipy.optimize import lsq_linear
import numpy as np
import cv2

def getWorldPoints(query_inliers, train_inliers, P1, P2):
    points = []
    for (train_inlier, query_inlier) in zip(query_inliers, train_inliers):
        #points.append(LinearTriangulation(train_inlier, query_inlier, P1, P2))
        point = cv2.triangulatePoints(P1,P2,np.array([[query_inlier[0]], [query_inlier[1]]]),np.array([[train_inlier[0]], [train_inlier[1]]])).reshape(-1)
        if(point[-1] == 0):
            point = 0*point[:-1]
        else:
            point = point[:-1]/point[-1]
        points.append(point)
    return np.array(points)