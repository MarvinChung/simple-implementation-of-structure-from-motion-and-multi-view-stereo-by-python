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
import queue  
from collections import defaultdict
class MyPatchHeapSort(object):
    #add negative for max heap
    def __init__(self, initial=None, key=lambda x:(x.dist, x.c[0], x.c[1], x.c[2], x.R)):
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

class MyPatch(object):
    def __init__(self, centroid, normal, reference_img_index, visible_set, color, dist, patch_size = 5):
        self.dist = dist
        self.c = centroid
        self.n = normal
        self.R = reference_img_index
        self.V = visible_set
        if type(self.V) == type(None):
            self.V = []
        self.color = color
        self.patch_size = patch_size
        self.avg_ncc_score = 0


    def visible_ct(self):
        return len(self.V)

    def photo_consistenecy_test(self, imgs, par_K, par_r, par_t, MIN_NCC = 0.7):
        base_point = projectPoint(self.c,  par_r[self.R],  par_t[self.R],  par_K[self.R])   
        base_des = getDescFeatures(imgs[self.R], np.array([base_point[[1,0]]]), wid=5)[0]

        for idx, img in enumerate(imgs):
            if(idx != self.R):
                point = projectPoint(self.c,  par_r[self.R],  par_t[self.R],  par_K[self.R])
                des = getDescFeatures(imgs[idx], np.array([point[[1,0]]]), wid=5)[0]
                if type(base_des) != type(None) and type(des) != type(None) and base_des.shape == des.shape:
                    ncc_score = ctNcc(base_des, des)
                    if(ncc_score > MIN_NCC):
                        self.avg_ncc_score += ncc_score
                        self.V.append([idx, point[0], point[1]])
        if (self.visible_ct() > 0):
            self.avg_ncc_score /= self.visible_ct()
        return self.V


class CellTable(object):
    def __init__(self, imgs, cell_size=4.0):
        self.table = []
        self.Q_table = defaultdict(list)
        self.cell_size = cell_size
        for img in imgs:
            row = img.shape[0]
            col = img.shape[1]        
            self.table.append(np.ones((math.ceil((col-1)/cell_size), math.ceil((row-1)/cell_size)), dtype=bool))

    def is_vacant(self, img_id, cell_i, cell_j):
        if cell_i >= self.table[img_id].shape[0] or cell_i < 0:
            return False
        elif cell_j >= self.table[img_id].shape[1] or cell_j < 0:
            return False
        else:
            return self.table[img_id][cell_i][cell_j]

    def fill_with_point(self, img_id, col, row, patch):
        if math.floor(col/self.cell_size) >= self.table[img_id].shape[0] or col < 0:
            print("CellTable col out of index")
            pdb.set_trace()
        elif math.floor(row/self.cell_size) >= self.table[img_id].shape[1] or row < 0:
            print("CellTable row out of index")
            pdb.set_trace()
        self.table[img_id][math.floor(col/self.cell_size)][math.floor(row/self.cell_size)] = False
        for idx, l_col, l_row in patch.V:
            self.Q_table[(img_id,math.floor(l_col/self.cell_size),math.floor(l_row/self.cell_size))].append(patch)

    def show_table_non_zeros(self):
        for i,t in enumerate(self.table):
            print("img ",i," size:",t.shape," has seen:",t.shape[0]*t.shape[1]-np.count_nonzero(t))

    def which_cell(self, col, row):
        return math.floor(col/self.cell_size), math.floor(row/self.cell_size)

    def cell_center(self, cell_i ,cell_j):
        return np.array([self.cell_size * (cell_i + 0.5), self.cell_size * (cell_j + 0.5)])

    def get_color(self, img, col, row):
        return img[int(row)][int(col)]

    def show_cell(self, img, cell_i, cell_j):
        print("cell_i:",cell_i)
        print("cell_j:",cell_j)
        img = img.copy()
        arr = self.cell_center(cell_i,cell_j)
        img = cv2.circle(img,tuple([int(arr[0]),int(arr[1])]),5,(0,255,0),-1)
        plt.imshow(img)
        plt.show()


    def filter_out_outlier(self):
        for idx, t in enumerate(self.table):
            for i in range(t.shape[0]):
                for j in range(t.shape[1]):
                    if(t[i][j] == False):
                        #filter 3 times for better accuracy
                        for iteration in range(1):
                            threshold = 0
                            for p in self.Q_table[(idx,i,j)]:
                                threshold += 1 - p.avg_ncc_score
                            threshold /= len(self.Q_table[(idx,i,j)])    
                            outlier_patch = []
                            for p1 in self.Q_table[(idx,i,j)]:
                                for p2 in self.Q_table[(idx,i,j)]:
                                    if(p1 != p2):
                                        if(not is_patch_neighbor(p1, p2) and p2.visible_ct() * p2.avg_ncc_score < threshold):
                                            outlier_patch.append(p2)
                            for p in outlier_patch:
                                #eliminate p in all Q
                                for img_id, point_i, point_j in p.V:
                                    temp = []
                                    for query_p in self.Q_table[(img_id,math.floor(point_i/self.cell_size),math.floor(point_j/self.cell_size))]:
                                        if( p != query_p):
                                            temp.append(query_p)
                                        else:
                                            print("remove a outlier")
                                    self.Q_table[(img_id,math.floor(point_i/self.cell_size),math.floor(point_j/self.cell_size))] = temp
    def reconstruct_from_Q(self):
        hash_table = {}
        points_3d = []
        colors = []
        for idx, t in enumerate(self.table):
            for i in range(t.shape[0]):
                for j in range(t.shape[1]):
                    for p in self.Q_table[(idx,i,j)]:
                        if(p not in hash_table):
                            hash_table[p] = True
                            points_3d.append(p.c)
                            colors.append(p.color)


        return points_3d, colors


def DensePointsWithMVS2(imgs, global_set, args):
    t0 = time.time()
    par_K, par_r, par_t = read_pars(args)
    n_observations, n_world_points, legal_sets = global_set.getInfo()
    n_cameras = len(imgs)
    wid = args.desc_wid
    scale = args.scale
    debug = args.debug
    cell_size = args.cell_size

    camera_pos = []

    for i in range(len(imgs)):
        camera_pos.append(-(par_r[i].transpose() @ par_t[i].reshape(3,-1)).reshape(-1))

    points_3d = []

    # is_seen = []
    # for img in imgs:
    #     is_seen.append(np.zeros(img.shape[:2], dtype = bool).transpose())

    # using sfm outpus as initial sparse set
    cells = CellTable(imgs, cell_size = args.cell_size)

    if len(imgs) > 2:
        visible_lower_bound = 3
    else:
        visible_lower_bound = 2

    initial_patches = []

    #initial sparse patches
    for legal_set in legal_sets:
        base_point = None
        #world_point = np.asarray(legal_set.world_point)
        CandidatePatchesHeap = MyPatchHeapSort()
                    
        
        
        O_center = None
        centroid = None
        reference_img_index = None
        normal = None
        visible_set = []
        color = np.array([0.0,0.0,0.0])

        #These point2ds are correspond to a 3d point
        for ct, point2d_tuple in enumerate(legal_set.point2d_list):
            #image_index is same as camera index
            if(point2d_tuple[0] >= n_cameras):
                print("out of bound")
                pdb.set_trace()
            camera_index = point2d_tuple[0]
            points_2d = [float(point2d_tuple[1]), float(point2d_tuple[2])]


            #base img
            if ct == 0:
                reference_img_index = camera_index
                base_point = points_2d
                O_center = camera_pos[camera_index]
            else:
                P1 = getProjectionMatrix(par_K[reference_img_index], par_r[reference_img_index], par_t[reference_img_index])
                P2 = getProjectionMatrix(par_K[camera_index], par_r[camera_index], par_t[camera_index])
                un_point = traingulatePoints(P1, P2, np.array([base_point]),np.array([points_2d]))[0]
                if(un_point[-1] == 0):
                    c = 0*un_point[:-1]
                else:                    
                    c = un_point[:-1]/un_point[-1]
                dist = distance(c, O_center) 
                n = (O_center - c)/dist
                #(centroid, normal, reference_img_index, visible_set, color, dist)
                color = cells.get_color(imgs[camera_index], point2d_tuple[1], point2d_tuple[2])
                candidate_patch = MyPatch(c, n, reference_img_index, None, color, dist)
                CandidatePatchesHeap.push(candidate_patch)

        #sort in an increasing order of distance from O_center
        while(CandidatePatchesHeap.size() != 0):
            non_finished_patch = CandidatePatchesHeap.pop()
            hit_points = non_finished_patch.photo_consistenecy_test(imgs, par_K, par_r, par_t, MIN_NCC = 0.4)
            if(non_finished_patch.visible_ct() >= visible_lower_bound):
                initial_patches.append(non_finished_patch)
                for hit_point in hit_points:
                    cells.fill_with_point(hit_point[0], hit_point[1], hit_point[2], non_finished_patch)
                break   

    print("len of initial patches",len(initial_patches))

    points_3d = []
    colors = []
    for p in initial_patches:
        points_3d.append(p.c)
        colors.append(p.color)
    if False:
        ax = plt.axes(projection='3d')
        ax.set_title('initial_patches')
        ax.scatter(scale * np.array(points_3d)[:,0], scale * np.array(points_3d)[:,1], scale * np.array(points_3d)[:,2], c=np.array(colors)/255.0, cmap='viridis', linewidth=0.1);
        plt.show()
    export2ply(np.array(points_3d), np.array(colors), path="initial_patches")
    
    patch_expansion(args, imgs, initial_patches, cells, camera_pos, visible_lower_bound)

    #filter 
    print("filter outliers")
    #very very slow
    #cells.filter_out_outlier()

    #reconstruct
    print("reconstruct point cloud")
    points_3d, colors = cells.reconstruct_from_Q()

    t1 = time.time()

    print("Optimization took {0:.0f} seconds".format(t1 - t0))
    print("points len:",len(points_3d))
    ax = plt.axes(projection='3d')
    ax.set_title('reconstruct_results')
    ax.scatter(scale * np.array(points_3d)[:,0], scale * np.array(points_3d)[:,1], scale * np.array(points_3d)[:,2], c=np.array(colors)/255.0, cmap='viridis', linewidth=0.1);
    plt.show()
    export2ply(np.array(points_3d), np.array(colors), path="all_patches")


def is_patch_neighbor(patch, non_finished_patch, threshold=0.2):
    return abs(np.dot(patch.c - non_finished_patch.c, patch.n) + np.dot(patch.c - non_finished_patch.c, non_finished_patch.n)) < threshold


def ray_plane_intersection(ray_origin, ray_direction, plane_center, plane_normal):
    #pdb.set_trace()
    dot_out = np.dot(ray_direction, plane_normal)
    t = np.dot(plane_center-ray_origin, plane_normal)/dot_out
    return ray_origin + t * ray_direction

def patch_expansion(args, imgs, initial_patches, cells, camera_pos, visible_lower_bound):
    par_K, par_r, par_t = read_pars(args)
    wid = args.desc_wid
    scale = args.scale
    debug = args.debug

    patch_queue = queue.Queue() 
    for p in initial_patches:
        patch_queue.put(p)

    #expand_patches = []

    iteration = 0
    while(not patch_queue.empty() and iteration < 100000):
        iteration += 1
        print("iteration:",iteration)
        print("patch_queue size",patch_queue.qsize())
        #if iteration % 20 == 0:
        #    cells.show_table_non_zeros()
        patch = patch_queue.get()
        for hit_point in patch.V:
            img_idx = hit_point[0]
            cell_i, cell_j = cells.which_cell(hit_point[1], hit_point[2])
            for i in [-1,1]:
                for j in [-1,1]:
                    if(cells.is_vacant(img_idx, cell_i + i, cell_j + j)):
                        cell_center = cells.cell_center(cell_i + i, cell_j + i)
                        c_x = par_K[img_idx][0,2]
                        c_y = par_K[img_idx][1,2]
                        f_x = par_K[img_idx][0,0]
                        f_y = par_K[img_idx][1,1]
                        
                        #p_norm = K^-1 * p
                       
                        #P_c = R(P_w - C)
                        #C = -R^transpose @ t
                        #P_w = R^-1 @ P_c - C 
                        #R^-1 = R^transpose()
                        #p_norm = np.linalg.inv(par_K[img_idx]) @ np.array([cell_center[0], cell_center[1], 1])
                        #p_unnorm = np.array([cell_center[0], cell_center[1], 1])
                        #C = (-par_r[img_idx].transpose() @ par_t[img_idx]).ravel()
                        #P_w = par_r[img_idx].transpose() @ p_norm - C
                        
                        C = (-par_r[img_idx].transpose() @ par_t[img_idx]).ravel()
                        P_w = par_r[img_idx].transpose() @ np.array([cell_center[0] - c_x, cell_center[1] - c_y, (f_x + f_y)/2]).reshape(-1) - C
                        norm_vector = vector_norm(P_w)
                        intersect_point = ray_plane_intersection(camera_pos[img_idx], norm_vector, patch.c, patch.n)
                        #(centroid, normal, reference_img_index, visible_set, color, dist)
                        O_center = camera_pos[img_idx]
                        color = cells.get_color(imgs[img_idx], cell_center[0], cell_center[1])
                        dist = distance(intersect_point, O_center)
                        n = (O_center - intersect_point)/dist
                        non_finished_patch = MyPatch(intersect_point, n, img_idx, None, color, None)
                        hit_points = non_finished_patch.photo_consistenecy_test(imgs, par_K, par_r, par_t, MIN_NCC = 0.7)


                        #filter out non neighbors
                        #print(non_finished_patch.visible_ct())
                        #print(is_patch_neighbor(patch, non_finished_patch, threshold=0.1))
                        if(non_finished_patch.visible_ct() >= visible_lower_bound and is_patch_neighbor(patch, non_finished_patch, threshold=0.1) and distance(patch.c, non_finished_patch.c) < 0.05/scale):
                            #expand_patches.append(non_finished_patch)
                            print("add")

                            if debug:
                                print("hit_point",hit_point[1],"and",hit_point[2])
                                cells.show_cell(imgs[img_idx],cell_i + i,cell_j + j)
                                points_3d, colors = cells.reconstruct_from_Q()

                                print("patch c:",patch.c)
                                print("non_finished_patch c:",non_finished_patch.c)
                                print("dist:",distance(patch.c, non_finished_patch.c))
                                t1 = time.time()
                                ax = plt.axes(projection='3d')
                                ax.set_title('reconstruct_results tp')
                                # for i in range(len(par_t)):
                                #     points_3d.append(-(par_r[i].transpose()@par_t[i]).ravel())
                                #     colors.append([150,150,155])

                                points_3d.append(non_finished_patch.c)
                                points_3d.append(patch.c)
                                #points_3d.append(C)

                                colors.append([255,0,0])
                                colors.append([0,255,0])
                                #colors.append([0,255,255])

                                for i in range(1,10,1):
                                    points_3d.append(patch.c+i*0.01*patch.n)
                                    colors.append([0,0,255])
                                ax.scatter(scale * np.array(points_3d)[:,0], scale * np.array(points_3d)[:,1], scale * np.array(points_3d)[:,2], c=np.array(colors)/255.0, cmap='viridis', linewidth=0.1);
                                plt.show()

                            for hit_point in hit_points:
                                cells.fill_with_point(hit_point[0], hit_point[1], hit_point[2], non_finished_patch)
                                patch_queue.put(non_finished_patch)
                            break   
                        
                    

    #return expand_patches