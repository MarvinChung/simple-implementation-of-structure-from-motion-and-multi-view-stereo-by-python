import numpy as np
import scipy.linalg

#hyperperamaters
# f1 = 568.996140852
# f2 = 568.988362396
# s = 0
# mx = my = 1
# px = 643.21055941
# py = 477.982801038

f1 = f2 = 100
s = 0
mx = my = 1
px = 0
py = 0

K = np.array([[f1*mx, s, px],[0, f2*my, py], [0, 0, 1]])

W = np.array([[0, -1, 0],[1, 0 ,0], [0, 0, 1]])
eps = 1e-12

def getEssentialMatrix(K, F):
    E = K.transpose() @ F @ K
    
    return E
def getEssentialConfig(E):
    U, D, Vh = scipy.linalg.svd(E)
    e = (D[0] + D[1]) / 2
    E = U @ np.array([[e,0,0],[0,e,0],[0,0,0]]) @ Vh
    U, D, Vh = scipy.linalg.svd(E)
    return U, D, Vh

def CameraPosition1Config(U, Vh):
    R1 = U @ W @ Vh
    C1 = U[:,2]
    det = np.linalg.det(R1)
    #print(det)
    #input("4")
    if det < 0:
        assert(abs(det) <= 1 + eps)
        C1 = -C1
        R1 = -R1 
    return C1, R1

def CameraPosition2Config(U, Vh):
    R2 = U @ W @ Vh
    C2 = -U[:,2]
    det = np.linalg.det(R2)
    #print(det)
    #input("2")
    if det < 0:
        assert(abs(det) <= 1 + eps)
        C2 = -C2
        R2 = -R2    
    return C2, R2

def CameraPosition3Config(U, Vh):
    R3 = U @ W.transpose() @ Vh
    C3 = U[:,2]
    det = np.linalg.det(R3)
    #print(det)
    #input("3")
    if det < 0:
        assert(abs(det) <= 1 + eps)
        C3 = -C3
        R3 = -R3
    return C3, R3    
        
def CameraPosition4Config(U, Vh):
    R4 = U @ W.transpose() @ Vh
    C4 = -U[:,2]
    det = np.linalg.det(R4)
    #print(det)
    #input("4")
    if det < 0:
        assert(abs(det) <= 1 + eps)
        C4 = -C4
        R4 = -R4
    return C4, R4

def CameraPoseMatrix(K, R, C):
    return K @ R @ np.array([[1,0,0, -C[0]],[0,1,0, -C[1]],[0,0,1, -C[2]]])

def getCameraMatrix(K, R0, C0, R1, C1):
    #first camera
    P1 = CameraPoseMatrix(K, R0, C0)#K @ R0 @ np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])

    #second camera
    P2 = CameraPoseMatrix(K, R1, C1) #np.concatenate((R1, -C1.reshape(3,1)), axis=1)
    #print("[DEBUG]P1",P1)
    #print("[DEBUG]P2",P2)
    
    return P1, P2

def CheckCheirality(points, C1, R1, C2, R2):
    pts_in_front_of_C1 = []
    pts_in_front_of_C2 = []
    temp_idx = []
    idx = [] 
    for (ct,point) in enumerate(points):
        is_chosen = False
        if(R1[2,:].dot(point-C1)>0):
            pts_in_front_of_C1.append(point)
            is_chosen = True
            
        if(R2[2,:].dot(point-C2)>0):
            pts_in_front_of_C2.append(point)
            is_chosen = True
            
        if(is_chosen):
            temp_idx.append(ct)
            
    temp_idx = np.array(temp_idx)
    Percentile = np.percentile(points[temp_idx],[0,25,50,75,100],axis=0)
    IQR = Percentile[3] - Percentile[1]
    UpLimit = Percentile[3] + IQR*1.5
    DownLimit = Percentile[1] - IQR*1.5
    for i in temp_idx:
        if(points[i][0] >= DownLimit[0] and points[i][1] >= DownLimit[1] and points[i][2] >= DownLimit[2] and points[i][0] <= UpLimit[0] and points[i][1] <= UpLimit[1] and points[i][2] <= UpLimit[2]):
            idx.append(i)
    return np.array(pts_in_front_of_C1).reshape(-1,3), np.array(pts_in_front_of_C2).reshape(-1,3), np.array(idx)