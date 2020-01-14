from HarrisFeatures import *
from Matchable_area import *
from utils import *
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import heapq
import multiprocessing
from multiprocessing import Process, Queue
import copy
import sys
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


class MyPatchHeap(object):
    #add negative for max heap
    def __init__(self, initial=None, key=lambda x:(-x.visible_ct, x.tuple_list[0][0], x.tuple_list[0][1], x.tuple_list[0][2])):
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
class MyPatch(object):
    def __init__(self, visible_ct, tuple_list):
        self.visible_ct = visible_ct
        #tuple_list : [(img_id, src_pt, dst_pt),....]
        self.tuple_list = tuple_list
        



def ctNcc(desc1, desc2):
    n_pixels = len(desc1)
    d1 = (desc1-np.mean(desc1))/np.std(desc1) 
    d2 = (desc2-np.mean(desc2))/np.std(desc2) 
    return sum(d1*d2)/(n_pixels - 1) 



def Test2MethodsOfDensePointsWithTwoViewStereo(imgs, args):
    #this is only a testing, filtering outliers are not implemented yet
    par_K, par_r, par_t = read_pars(args)
    debug = args.debug
    wid = 5
    harris_features = []
    harris_tables = []
    cutoff = 2
    # expand_base = wid+1
    if args.isSeq == 1:
        imgs_combination = getSequence(imgs)

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
            #harris point is row first then col second
            harris_table[pt[0], pt[1]] = True
        harris_tables.append(harris_table)
    
    threshold = 0.5
    scale = args.scale

    refined_src_features = []   
    refined_dst_features = []
    
    for ct, (idx_A,img,idx_B,cmp_img) in enumerate(tqdm(imgs_combination[:cutoff], total = len(imgs_combination[:cutoff]))):
        filtered_coords1 = harris_features[idx_A]    
        filtered_coords2 = harris_features[idx_B]

        #for P1 and P2 and opencv : The image origin is top-left, with x increasing horizontally, y vertically
        #Therefore make image as size as img.shape[1], img.shape[0]
        src_is_seen = np.full((img.shape[1],img.shape[0]), False)
        dst_is_seen = np.full((img.shape[1],img.shape[0]), False)

        src_mat = non_Matchable_area(img, debug = args.debug, grad_threshold = 0.01)
        dst_mat = non_Matchable_area(cmp_img, debug = args.debug, grad_threshold = 0.01)
        
        #The image origin is top-left, with x increasing horizontally, y vertically -> transpose src_mat 
        src_is_seen = src_mat.transpose()
        dst_is_seen = dst_mat.transpose()
        

         
        #use in method1 and method2               
        d1 = getDescFeatures(img, filtered_coords1, wid) 
        d2 = getDescFeatures(cmp_img, filtered_coords2, wid) 
        
    #method1
    #Match the harris first, then use the matches to form the Fundamental matrix
        """
        #the output matches have a threshold for NCC score which ensure photo consistency
        
        matches = MatchTwoSided(d1, d2, threshold = threshold)
        #filtered_coords are row first then col second but src_pts and dst_pts are col first row second
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
    #Use orb features to form the fundamental matrix, then find harris features that satisfy the epipolar constraints
        F = getORBFeatures(img, cmp_img, debug = debug, return_F = True)

        heap = MyMatchHeap()
        lines = cv2.computeCorrespondEpilines(filtered_coords1, 1, F).reshape(-1,3)

        if True:#debug:
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

                    if y < cmp_img.shape[0] and y >= 0 and src_is_seen[left_pt[0],left_pt[1]] == False and dst_is_seen[x,y] == False :
                        if  harris_tables[idx_B][x,y] == True:
                            left_des = getDescFeatures(img, [left_pt[[1,0]]], wid=wid)[0]
                            right_des = getDescFeatures(cmp_img, [np.array([y,x])], wid=wid)[0]
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
                if True:
                    print(match_obj.dst_point)
                    print(match_obj.src_point)
                    cmp_img_with_lines, right = drawlines(cmp_img, img, [line], [match_obj.dst_point], [match_obj.src_point])                  
                    fig=plt.figure(figsize=(64, 64))
                    fig.suptitle('draw Epilines on both images(first)')
                    fig.add_subplot(1,2,1)
                    plt.imshow(cmp_img_with_lines, aspect='auto')
                    fig.add_subplot(1,2,2)
                    plt.imshow(right, aspect='auto')
                    plt.show()
    #end method 2
         
        # while(heap.size() != 0):
        #     print(np.count_nonzero(src_is_seen))
        #     print(np.count_nonzero(dst_is_seen))
        #     print(src_is_seen.shape)
        #     print(dst_is_seen.shape)
        #     print("heap size:", heap.size())
        #     match_obj = heap.pop()
        #     src_point = match_obj.src_point
        #     dst_point = match_obj.dst_point
            
        #     refined_src_features.append(src_point.astype('float32'))
        #     refined_dst_features.append(dst_point.astype('float32'))
            
        #     src_min_x = max(0,src_point[0] - wid)
        #     src_max_x = min(img.shape[1]-1,src_point[0] + wid)
        #     src_min_y = max(0,src_point[1] - wid)
        #     src_max_y = min(img.shape[0]-1,src_point[1] + wid)
            
        #     dst_min_x = max(0,dst_point[0] - wid)
        #     dst_max_x = min(cmp_img.shape[1]-1,dst_point[0] + wid)
        #     dst_min_y = max(0,dst_point[1] - wid)
        #     dst_max_y = min(cmp_img.shape[0]-1,dst_point[1] + wid)
            
        #     #Add new potential matches in their immediate spatial neighborhood into heap
        #     #temp_heap find the best ncc value of the src point and dst point of this patch 
        #     temp_heap = MyMatchHeap()
        #     for i in range(src_min_x, src_max_x):
        #         for j in range(src_min_y, src_max_y):
        #             if(src_is_seen[i,j] == False):
        #                 for k in range(dst_min_x, dst_max_x):
        #                     for l in range(dst_min_y, dst_max_y):
        #                         if(dst_is_seen[k,l] == False):
        #                             #function getDescFeatures expect point coordinate as (row, col)
        #                             left_des = getDescFeatures(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), [np.array([j,i])], wid=wid)[0]
        #                             right_des = getDescFeatures(cv2.cvtColor(cmp_img,cv2.COLOR_BGR2GRAY), [np.array([l,k])], wid=wid)[0]
        #                             if type(left_des) != type(None) and type(right_des) != type(None) and left_des.shape == right_des.shape:
        #                                 ncc_value = ctNcc(left_des, right_des)
        #                                 #print("ncc_value",ncc_value)
        #                                 if (ncc_value == math.nan):
        #                                     pdb.set_trace()
        #                                 if(1 - ncc_value > 0.8):
        #                                     temp_heap.push(MyMatch(np.array([i,j]),np.array([k,l]), ncc_value))
        #     if temp_heap.size() != 0:
        #         match_obj = temp_heap.pop()

        #         if(src_is_seen[match_obj.src_point[0],match_obj.src_point[1]] == True):
        #             print("wtf src")
        #             input("")
        #         if(dst_is_seen[match_obj.dst_point[0],match_obj.dst_point[1]] == True):
        #             print("wtf dst")
        #             input("")

        #         # src_point and dst_point is match, therefore don't seen them again
        #         src_is_seen[match_obj.src_point[0],match_obj.src_point[1]] = True
        #         dst_is_seen[match_obj.dst_point[0],match_obj.dst_point[1]] = True
        #         heap.push(match_obj)
        #     #Only construct a point in a patch
        #     for i in range(src_min_x, src_max_x):
        #         for j in range(src_min_y, src_max_y):
        #             for k in range(dst_min_x, dst_max_x):
        #                     for l in range(dst_min_y, dst_max_y):
        #                         src_is_seen[i,j] = True
        #                         dst_is_seen[k,l] = True
                                       
        # P1 = getProjectionMatrix(par_K[idx_A], par_r[idx_A], par_t[idx_A])
        # P2 = getProjectionMatrix(par_K[idx_B], par_r[idx_B], par_t[idx_B])
        


        # plt.subplot(221)
        # img1 = img.copy()
        # for pt in refined_src_features:
        #     img1 = cv2.circle(img1,tuple(pt),5,(0, 255, 0),-1)
        # plt.imshow(img1)
        # #plt.scatter(np.array(refined_src_features)[:,1], np.array(refined_src_features)[:,0])
        # plt.subplot(222)
        # img2 = cmp_img.copy()
        # for pt in refined_dst_features:
        #     img2 = cv2.circle(img2,tuple(pt),5,(0, 255, 0),-1)
        # plt.imshow(img2)
        # #plt.scatter(np.array(refined_dst_features)[:,1], np.array(refined_dst_features)[:,0])
        # plt.show()


        # #abort trap in cv2.triangulatePoints if query_inliers or train_inliers are type int
        
        # points = traingulatePoints(P1, P2, np.array(refined_src_features), np.array(refined_dst_features))
        # #points = traingulatePoints(P1, P2, np.concatenate((np.array(refined_src_features),np.array([1 for i in range(len(refined_src_features))]).reshape(-1,1)),axis=1), np.concatenate((np.array(refined_dst_features),np.array([1 for i in range(len(refined_dst_features))]).reshape(-1,1)),axis=1)) 
        # ax = plt.axes(projection='3d')
        # ax.set_title('all images MVS')
        # ax.scatter(scale * np.array(points)[:,0], scale * np.array(points)[:,1], scale * np.array(points)[:,2], c=np.array(points)[:,2], cmap='viridis', linewidth=0.1);
        # plt.show()

class CellTable(object):
    def __init__(self, imgs, cell_size=5.0):
        self.table = []
        self.cell_size = cell_size
        for img in imgs:
            row = img.shape[0]
            col = img.shape[1]        
            self.table.append(np.ones((math.ceil((col-1)/cell_size), math.ceil((row-1)/cell_size)), dtype=bool))

    def is_vacant(self, img_id, col, row):
        return self.table[img_id][math.floor(col/self.cell_size)][math.floor(row/self.cell_size)]

    def fill(self, img_id, col, row):
        self.table[img_id][math.floor(col/self.cell_size)][math.floor(row/self.cell_size)] = False

def expand_patch(p_id, img, cmp_img, src_index, dst_index, src_matchable_area, dst_matchable_area, cells, src_point, dst_point, cell_size, wid):
    #Add new potential matches in their immediate spatial neighborhood into heap
    #temp_heap find the best ncc value of the src point and dst point of two images

    src_min_x = max(0,int(src_point[0]) - 2*cell_size)
    src_max_x = min(img.shape[1]-1,int(src_point[0]) + 2*cell_size)
    src_min_y = max(0,int(src_point[1]) - 2*cell_size)
    src_max_y = min(img.shape[0]-1,int(src_point[1]) + 2*cell_size)

    dst_min_x = max(0,int(dst_point[0]) - 2*cell_size)
    dst_max_x = min(cmp_img.shape[1]-1,int(dst_point[0]) + 2*cell_size)
    dst_min_y = max(0,int(dst_point[1]) - 2*cell_size)
    dst_max_y = min(cmp_img.shape[0]-1,int(dst_point[1]) + 2*cell_size)

    #print("src_min_x",src_min_x)
    #print("src_max_x",src_max_x)
    #print("src_min_y",src_min_y)
    #print("src_max_y",src_max_y)

    #print("dst_min_x",dst_min_x)
    #print("dst_max_x",dst_max_x)
    #print("dst_min_y",dst_min_y)
    #print("dst_max_y",dst_max_y)
    trange = tqdm(range(src_min_x, src_max_x))

    #using temp_heap will cause huge memory use on child process, which causes a deadlock
    #temp_heap = MyMatchHeap()
    max_ncc = 0
    match_obj = None
    for i in trange:
        for j in range(src_min_y, src_max_y):
            #print(p_id,i,j,src_max_x,src_max_y)
            if src_matchable_area[i,j]  and cells.is_vacant(src_index, i ,j): #and not is_seen[src_index][i,j]:              
                for k in range(dst_min_x, dst_max_x):
                    for l in range(dst_min_y, dst_max_y):
                        #print(p_id,i,j,k,l,src_max_x,src_max_y,dst_max_x,dst_max_y)
                        if dst_matchable_area[k,l] and cells.is_vacant(dst_index, k, l): #and not is_seen[dst_index][k,l]                  
                            #function getDescFeatures expect point coordinate as (row, col)
                            left_des = getDescFeatures(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), [np.array([j,i])], wid=wid)[0]
                            right_des = getDescFeatures(cv2.cvtColor(cmp_img,cv2.COLOR_BGR2GRAY), [np.array([l,k])], wid=wid)[0]
                            if type(left_des) != type(None) and type(right_des) != type(None) and left_des.shape == right_des.shape:
                                ncc_value = ctNcc(left_des, right_des)
                                if (ncc_value == math.nan):
                                    print("wtf")
                                    pdb.set_trace()
                                if(ncc_value > 0.7 and ncc_value > max_ncc):
                                    match_obj = MyMatch(np.float32([i,j]),np.float32([k,l]), ncc_value)
                                    #temp_heap.push(MyMatch(np.float32([i,j]),np.float32([k,l]), ncc_value))
    # if temp_heap.size() != 0:
    #     flag = 1
    #     match_obj = temp_heap.pop()
    #     matches_list.append((dst_index,match_obj.src_point,match_obj.dst_point))
    #     temp_list.append((dst_index, match_obj.dst_point[0], match_obj.dst_point[1]))
    #     #is_seen[src_index][int(match_obj.src_point[0]),int(match_obj.src_point[1])] = True
    #     #is_seen[dst_index][int(match_obj.dst_point[0]),int(match_obj.dst_point[1])] = True
    #     visible_ct += 1 
    if type(match_obj) != None:
        return [(dst_index,match_obj.src_point,match_obj.dst_point), (dst_index, match_obj.dst_point[0], match_obj.dst_point[1]), 1, match_obj.ncc_score] 
    else:
        return None

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def wrapper(func, id ,args, q):
    print("in wrapper")
    q.put(func(*args))

def DensePointsWithMVS2(imgs, global_set, args):
    t0 = time.time()
    par_K, par_r, par_t = read_pars(args)
    n_observations, n_world_points, legal_sets = global_set.getInfo()
    n_cameras = len(imgs)
    wid = args.desc_wid
    scale = args.scale
    debug = args.debug
    cell_size = args.cell_size

    points_3d = []
    initial_3d_table = {}

    # is_seen = []
    # for img in imgs:
    #     is_seen.append(np.zeros(img.shape[:2], dtype = bool).transpose())

    # using sfm outpus as initial sparse set
    heap = MyPatchHeap()
    cells = CellTable(imgs, cell_size = args.cell_size)

    if len(imgs) > 2:
        visible_lower_bound = 3
    else:
        visible_lower_bound = 2

    for legal_set in legal_sets:
        visible_ct = 0
        base_des = None
        temp_list = []
        world_point = np.asarray(legal_set.world_point)
        color = np.array([0.0,0.0,0.0])
        #These point2ds are correspond to a 3d point
        for ct, point2d_tuple in enumerate(legal_set.point2d_list):
            

            #image_index is same as camera index
            if(point2d_tuple[0] >= n_cameras):
                print("out of bound")
                pdb.set_trace()
            camera_index = point2d_tuple[0]
            points_2d = [float(point2d_tuple[1]), float(point2d_tuple[2])]
            initial_3d_table[point2d_tuple] = world_point
            color += imgs[camera_index][int(point2d_tuple[2])][int(point2d_tuple[1])]


            if visible_ct == 0:
                #getDesc point input using [row, col] coordinate, however points_2d is based on [col, row] coordinate
                des = getDescFeatures(cv2.cvtColor(imgs[camera_index],cv2.COLOR_BGR2GRAY), [np.array(points_2d)[[1,0]]], wid=wid)[0]
                base_des = des
                temp_list.append(point2d_tuple)
                if type(base_des) != type(None):
                    visible_ct += 1
            else:
                #getDesc point input using [row, col] coordinate, however points_2d is based on [col, row] coordinate
                des = getDescFeatures(cv2.cvtColor(imgs[camera_index],cv2.COLOR_BGR2GRAY), [np.array(points_2d)[[1,0]]], wid=wid)[0]
                if type(des) != type(None) and base_des.shape == des.shape:
                    ncc_value = ctNcc(des, base_des)
                    if (ncc_value == math.nan):
                        return None
                        #pdb.set_trace()
                    elif(ncc_value > 0.7):
                        visible_ct += 1
                        temp_list.append(point2d_tuple)
        if visible_ct >= visible_lower_bound:
            patch = MyPatch(visible_ct, temp_list)
            for img_id, col, row in temp_list:
                cells.fill(img_id, col, row)
            heap.push(patch)       
            points_3d.append([world_point[0], world_point[1], world_point[2]]+list( (color/len(legal_set.point2d_list)).astype(int)))

def DensePointsWithMVS(imgs, global_set, args):
    t0 = time.time()
    par_K, par_r, par_t = read_pars(args)
    n_observations, n_world_points, legal_sets = global_set.getInfo()
    n_cameras = len(imgs)
    wid = args.desc_wid
    scale = args.scale
    debug = args.debug
    cell_size = args.cell_size

    points_3d = []
    initial_3d_table = {}

    # is_seen = []
    # for img in imgs:
    #     is_seen.append(np.zeros(img.shape[:2], dtype = bool).transpose())

    # using sfm outpus as initial sparse set
    heap = MyPatchHeap()
    cells = CellTable(imgs, cell_size = args.cell_size)

    if len(imgs) > 2:
        visible_lower_bound = 3
    else:
        visible_lower_bound = 2

    for legal_set in legal_sets:
        visible_ct = 0
        base_des = None
        temp_list = []
        world_point = np.asarray(legal_set.world_point)
        color = np.array([0.0,0.0,0.0])
        #These point2ds are correspond to a 3d point
        for ct, point2d_tuple in enumerate(legal_set.point2d_list):
            

            #image_index is same as camera index
            if(point2d_tuple[0] >= n_cameras):
                print("out of bound")
                pdb.set_trace()
            camera_index = point2d_tuple[0]
            points_2d = [float(point2d_tuple[1]), float(point2d_tuple[2])]
            initial_3d_table[point2d_tuple] = world_point
            color += imgs[camera_index][int(point2d_tuple[2])][int(point2d_tuple[1])]


            if visible_ct == 0:
                #getDesc point input using [row, col] coordinate, however points_2d is based on [col, row] coordinate
                des = getDescFeatures(cv2.cvtColor(imgs[camera_index],cv2.COLOR_BGR2GRAY), [np.array(points_2d)[[1,0]]], wid=wid)[0]
                base_des = des
                temp_list.append(point2d_tuple)
                if type(base_des) != type(None):
                    visible_ct += 1
            else:
                #getDesc point input using [row, col] coordinate, however points_2d is based on [col, row] coordinate
                des = getDescFeatures(cv2.cvtColor(imgs[camera_index],cv2.COLOR_BGR2GRAY), [np.array(points_2d)[[1,0]]], wid=wid)[0]
                if type(des) != type(None) and base_des.shape == des.shape:
                    ncc_value = ctNcc(des, base_des)
                    if (ncc_value == math.nan):
                        return None
                        #pdb.set_trace()
                    elif(ncc_value > 0.7):
                        visible_ct += 1
                        temp_list.append(point2d_tuple)
        if visible_ct >= visible_lower_bound:
            patch = MyPatch(visible_ct, temp_list)
            for img_id, col, row in temp_list:
                cells.fill(img_id, col, row)
            heap.push(patch)       
            points_3d.append([world_point[0], world_point[1], world_point[2]]+list( (color/len(legal_set.point2d_list)).astype(int)))
    #cell size is 1 pixel
    



    export2ply(np.array(points_3d)[:,:3], np.array(points_3d)[:,3:])

    iteration_ct = 0
    while(heap.size() != 0):
        iteration_ct +=1
        if(iteration_ct % 10 == 0):
            export2ply(np.array(points_3d)[:,:3], np.array(points_3d)[:,3:], path="multiprocessing_output")
        print("heap size:", heap.size()," len(points_3d)):",len(points_3d))
        print("iteration_ct:", iteration_ct)
        patch_obj = heap.pop()
        origin_visible_ct = patch_obj.visible_ct
        tuple_list = patch_obj.tuple_list
        visible_ct = 0
        matches_list = []
        temp_list = []
        P1 = None
        src_index = -1

        jobs = []
        queue = Queue()
        numberOfThreads = multiprocessing.cpu_count()
        print("numberOfThreads:",numberOfThreads)
        for run_ct, t in enumerate(tuple_list):
            camera_index = t[0]
            print(t)
            points_2d = [float(t[1]), float(t[2])]

            if run_ct == 0:
                # src image
                P1 = getProjectionMatrix(par_K[camera_index], par_r[camera_index], par_t[camera_index])
                src_index = camera_index
                img = imgs[camera_index]
                src_3d_point = initial_3d_table[t]
                src_point = points_2d
                

                src_matchable_area = Matchable_area(img, debug=args.debug).transpose()
                temp_list.append((camera_index, src_point[0], src_point[1]))
            else:

                cmp_img = imgs[camera_index]
                dst_index = camera_index
                dst_point = points_2d            
                dst_matchable_area = Matchable_area(cmp_img, debug=args.debug).transpose()    
                print(run_ct)
                jobs.append(Process(target = wrapper, args=(expand_patch,run_ct-1,(run_ct-1, img.copy(), cmp_img.copy(), src_index, dst_index, src_matchable_area.copy(), dst_matchable_area.copy(), copy.deepcopy(cells), src_point, dst_point, cell_size, wid), queue)))
        return_values = []
        for chunk in chunks(jobs,numberOfThreads):
            for j in chunk:
                print("start")
                j.start()
            while True:
                running = any(p.is_alive() for p in chunk)
                while not queue.empty():
                    return_values.append(queue.get(False))
                if not running:
                    break    
            for j in chunk:
                j.join()
        max_ncc = 0
        max_obj = None    
        for i in return_values:
            if(type(i) != None):
                if i[-1] > max_ncc:
                    max_obj = i
                visible_ct += i[2]
        if type(max_obj)!= None:
            matches_list.append(max_obj[0])
            temp_list.append(max_obj[1])
            print("visible_ct:",visible_ct)    

        if visible_ct >= visible_lower_bound:
            patch = MyPatch(visible_ct, temp_list)
            heap.push(patch)
            sum_3d_point = 0
            sum_2d_point = np.zeros((1,2),dtype=float)
            color = np.array([0,0,0])
            #calculate its 3d point
            for match in matches_list:
                #match => (dst_index,match_obj.src_point,match_obj.dst_point)
                dst_camera_index = match[0]             
                P2 = getProjectionMatrix(par_K[dst_camera_index], par_r[dst_camera_index], par_t[dst_camera_index])
                cells.fill(src_index, match[1][0], match[1][1])
                cells.fill(dst_camera_index, match[2][0], match[2][1])
                un_point = traingulatePoints(P1,P2,np.array([match[1]]),np.array([match[2]]))[0]
                if(un_point[-1] == 0):
                    point = 0*un_point[:-1]
                else:                    
                    point = un_point[:-1]/un_point[-1]
                sum_3d_point += point
                color += imgs[src_index][int(match[1][1]),int(match[1][0])]
                color += imgs[dst_camera_index][int(match[2][1]),int(match[2][0])]
                if debug:
                    print("src camera:",src_index)
                    print("dst camera:",match[0])
                    print("src point:",match[1])
                    print("dst point:",match[2])
                    print("point:",point)
                    print("unpoint:",un_point)
                    plt.subplot(221)
                    a = imgs[src_index].copy()
                    a = cv2.circle(a,tuple(match[1]),5,(0,255,0),-1)
                    plt.imshow(a)
                    plt.subplot(222)
                    b = imgs[dst_camera_index].copy()
                    b = cv2.circle(b,tuple(match[2]),5,(0,255,0),-1)
                    plt.imshow(b)
                    plt.show()
                    sum_2d_point += match[2]
            if distance(sum_3d_point/len(matches_list), src_3d_point) < 0.1:
                print("dist to origin:",distance(sum_3d_point/len(matches_list), src_3d_point))
                points_3d.append(list(sum_3d_point/len(matches_list))+list(( color/(2*len(matches_list) ) ).astype(int)))
            print(src_point)
            print(sum_2d_point/len(matches_list))
            print(src_3d_point)
            print(sum_3d_point/len(matches_list))
            if debug:
                ax = plt.axes(projection='3d')
                ax.set_title('all images MVS')
                ax.scatter(scale * np.array(points_3d)[:,0], scale * np.array(points_3d)[:,1], scale * np.array(points_3d)[:,2], c=np.array(points_3d)[:,3], cmap='viridis', linewidth=0.1);
                plt.show()

    export2ply(np.array(points_3d)[:,:3], np.array(points_3d)[:,3:])

    t1 = time.time()
    print("Optimization took {0:.0f} seconds".format(t1 - t0))
    ax = plt.axes(projection='3d')
    ax.set_title('all images MVS')
    ax.scatter(scale * np.array(points_3d)[:,0], scale * np.array(points_3d)[:,1], scale * np.array(points_3d)[:,2], c=np.array(points_3d)[:,3], cmap='viridis', linewidth=0.1);
    plt.show()
