import numpy as np
import scipy.linalg

#hyperperamaters
f1 = 568.996140852
f2 = 568.988362396
s = 0
mx = my = 1
px = 643.21055941
py = 477.982801038
K = np.array([[f1*mx, s, px],[0, f2*my, py], [0, 0, 1]])
W = np.array([[0, -1, 0],[1, 0 ,0], [0, 0, 1]])
eps = 1e-12

def getEssentialMatrix(F):
    E = K.transpose() @ F @ K
    return E
def getEssentialConfig(E):
    U, D, Vh = scipy.linalg.svd(E)
    return U, D, Vh

def CameraPosition1Config(U, D, Vh):
    R1 = U @ W @ Vh
    C1 = U[:,2]
    det = np.linalg.det(R1)
    if det != 1:
        assert(abs(det) <= 1 + eps)
        C1 = -C1
        R1 = -R1 
    return C1, R1

def CameraPosition2Config(U, D, Vh):
    R2 = U @ W @ Vh
    C2 = -U[:,2]
    det = np.linalg.det(R2)
    if det != 1:
        assert(abs(det) <= 1 + eps)
        C2 = -C2
        R2 = -R2    
    return C2, R2

def CameraPosition3Config(U, D, Vh):
    R3 = U @ W.transpose() @ Vh
    C3 = U[:,2]
    det = np.linalg.det(R3)
    if det != 1:
        assert(abs(det) <= 1 + eps)
        C3 = -C3
        R3 = -R3
    return C3, R3    
        
def CameraPosition4Config(U, D, Vh):
    R4 = U @ W.transpose() @ Vh
    C4 = -U[:,2]
    det = np.linalg.det(R4)
    if det != 1:
        assert(abs(det) <= 1 + eps)
        C4 = -C4
        R4 = -R4
    return C4, R4

def CameraPoseMatrix(K, R, C):
    return K @ R @ np.array([[1,0,0, -C[0]],[0,1,0, -C[1]],[0,0,1, -C[2]]])

def getCameraMatrix(U, Vh, K, R0, C0, R1, C1):
    #first camera
    P1 = CameraPoseMatrix(K, R0, C0)#K @ R0 @ np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])

    #second camera
    E = U @ np.array([[1,0,0],[0,1,0],[0,0,0]]) @ Vh
    P2 = CameraPoseMatrix(K, R1, C1) #np.concatenate((R1, -C1.reshape(3,1)), axis=1)
    return P1, P2

def CheckCheirality(points, C1, R1, C2, R2):
    n_in_front_of_C1 = 0
    n_in_front_of_C2 = 0
    for point in points:
        point = point[0:3]
        if(R1[2,:].dot(point-C1)>0):
            n_in_front_of_C1 +=1
        if(R2[2,:].dot(point-C2)>0):
            n_in_front_of_C2 +=1
    return n_in_front_of_C1, n_in_front_of_C2