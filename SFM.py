from utils import *
from BundleAdjustment import *
#from Normalized8pointsAlgo import RANSAC
#from CameraConfig import *
#from Triangulation import *


##### get the intrinsic and extrinsic matrix ########
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


def StructureFromMotion(imgs, global_set, args, MIN_REPROJECTION_ERROR=0.5):  
    
    if args.nonSeq:
        raise NotImplementedError
        #imgs_combination = getCombination(imgs)
    else:
        imgs_combination = getSequence(imgs)

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
        global_set.updateWorldPoints(legal_sets)    
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
