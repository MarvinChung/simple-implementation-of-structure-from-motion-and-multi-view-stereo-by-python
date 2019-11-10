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
    #print(temp_idx)
    Percentile = np.percentile(points[temp_idx],[0,25,50,75,100],axis=0)
    IQR = Percentile[3] - Percentile[1]
    UpLimit = Percentile[3] + IQR*1.5
    DownLimit = Percentile[1] - IQR*1.5
    for i in temp_idx:
        if(points[i][0] >= DownLimit[0] and points[i][1] >= DownLimit[1] and points[i][2] >= DownLimit[2] and points[i][0] <= UpLimit[0] and points[i][1] <= UpLimit[1] and points[i][2] <= UpLimit[2]):
            idx.append(i)
    return np.array(pts_in_front_of_C1).reshape(-1,3), np.array(pts_in_front_of_C2).reshape(-1,3), np.array(idx)

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
        Rs[i] = pre_R @ R_diff 
        Cs[i] = pre_R @ C_diff + pre_C
        
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
    #dist_coef = np.zeros((4,1))
    #(_, rvec, tvec, _) = cv2.solvePnPRansac(points, query_inliers[best_idx], K, dist_coef, cv2.SOLVEPNP_EPNP)
    
    if(best_ct == 0):
        print("[DEBUG]best_ct==0")
        pdb.set_trace()        
        
  ##########draw############           
#     scale = 1
#     print("[DEBUG]pre_C",pre_C)
#     print("[DEBUG]pre_R",pre_R)
#     print("[DEBUG]C", best_secondCamera_C)
#     print("[DEBUG]R", best_secondCamera_R)
#     ax = plt.axes(projection='3d')
#     ax.scatter(scale * pts_in_front_of_C1[:,0], scale * pts_in_front_of_C1[:,1], scale * pts_in_front_of_C1[:,2], c="blue", cmap='viridis', linewidth=0.5);
#     ax.scatter(scale * pts_in_front_of_C2[:,0], scale * pts_in_front_of_C2[:,1], scale * pts_in_front_of_C2[:,2], c="pink", cmap='viridis', linewidth=0.5);
#     colors = cm.rainbow(np.linspace(0, 1, 2))
    
#     ax.scatter(scale * pre_C[0], scale * pre_C[1], scale * pre_C[2], c=colors[0], linewidth=8, label='camera1');
#     ax.scatter(scale * best_secondCamera_C[0], scale * best_secondCamera_C[1], scale * best_secondCamera_C[2], c=colors[1], linewidth=8, label='camera2');
#     ax.legend()
#     plt.show() 

#########################       
    return best_secondCamera_C, best_secondCamera_R, points.astype(np.float32), train_inliers[best_idx], query_inliers[best_idx] 
