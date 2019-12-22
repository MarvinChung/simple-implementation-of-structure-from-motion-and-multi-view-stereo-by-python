import numpy
import math
import pdb

class MySet(object):
    def __init__(self, world_point, point2d_list):
        self.world_point = world_point
        self.point2d_list = point2d_list
    
#     def Union(self, lst1, lst2): 
#         final_list = list(set(lst1) | set(lst2)) 
#         return final_list 
    
    def union(self, other_world_point, other_point2d_list):
        #self.world_point = self.world_point * len(self.point2d_list) + other_world_point * len(other_point2d_list)/(len(self.point2d_list) + len(other_point2d_list))
        self.point2d_list = list(set(self.point2d_list) | set(other_point2d_list))
        
    def union_with(self, other_MySet):
        #self.world_point = self.world_point * len(self.point2d_list) + other_MySet.world_point * len(other_MySet.point2d_list)/(len(self.point2d_list) + len(other_MySet.point2d_list))
        self.point2d_list = list(set(self.point2d_list) | set(other_MySet.point2d_list))
        
class GlobalSet(object):
    def __init__(self, threshold = 0.01):
        self.valid = {}
        self.set_list = {}
        self.set_index = {}
        self.list_ct = 0
        self.threshold = threshold
        
    def clear(self):
        self.valid.clear()
        self.set_list.clear()
        self.set_index.clear()
        self.list_ct = 0
        
    def getInfo(self):
        """
        return the number of 2d points(n_observations), n_points3d, legal sets
        """
        n_observations = 0
        n_points3d = 0
        legal_sets = []
        
        for i in self.set_list:
            if(self.valid[i]):
                n_observations += len(self.set_list[i].point2d_list)
                n_points3d += 1
                legal_sets.append(self.set_list[i])
        
        return n_observations, n_points3d, legal_sets

    def updateWorldPoints(self, update_world_pt):
        ct = 0
        for i in self.set_list:
            if(self.valid[i]):
                self.set_list[i].world_point = update_world_pt[ct].world_point
                ct+=1
#     def get_n_observations(self):
#         """
#         return the number of 2d points
#         """
#         n_observations = 0
#         for i in self.set_list:
#             if(self.valid[i]):
#                 n_observations += len(self.set_list[i].point2d_list)
#         return n_observations
    
    def show_list(self):
        for i in self.set_list.values():
            print(i.world_point,i.point2d_list)
    
    
    def check_threshold(self, set_idx, b):
        try:
            a = self.set_list[set_idx].world_point
        except KeyError:
            print("check_threshold go wrong")
            pdb.set_trace()
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2) < self.threshold
    
    def add(self, a_2d_point, a_3d_point):
        """
        a_2d_point is a tuple
        """
        try:
            idx = self.set_index[a_2d_point]
        except KeyError:
            idx = -1
        if(idx != -1):
            self.set_list[self.list_ct] = MySet(a_3d_point, list(a_2d_point))
            self.valid[self.list_ct] = True
            self.list_ct+=1
        else:
            if(self.valid[idx] and self.check_threshold(idx, a_3d_point)):
                self.set_index[a_2d_point] = idx
                self.set_list[idx].union(a_3d_point, list(a_2d_point))
            else:
                #not in threshold, discard this set
                self.valid[idx] = False
            
    def add2pts(self, a_list, a_3d_point):
        """
        a_list should be python list type
        Input form:
            a_list:
                image_index is same as camera index
                [(image_index1, x1, y1), (image_index2, x2, y2)]
            a_3d_point:
                array like([x, y, z], dtype=float32)
       
        a_set should a pair of points that match from two images.
        for example:
        After main program proccessed image 1,2
        if original MySet:
                [(1,x1,y1), (2,x'1,y'1)
                ,(1,x2,y2), (2,x'2,y'2)
                           .
                ,(1,xn,yn), (2,x'n,y'n)]
        Then main program is handling image 1,3
        The set [(1, x1, y1), (3, x2, y2)] should be added in.
        
        self.valid = False:
            Every 2d point pair should point to a 3d point. If they should be in a same set, there 3d point should not be 
            too far away. If their 3d points distance is not inside the threshold, discard this set.
        """
        try:
            idx1 = self.set_index[a_list[0]]
        except KeyError:
            idx1 = -1
            
        try:
            idx2 = self.set_index[a_list[1]]
        except KeyError:
            idx2 = -1
            
        if(idx1 == -1 and idx2 == -1):
            self.set_index[a_list[0]] = self.list_ct
            self.set_index[a_list[1]] = self.list_ct
            self.set_list[self.list_ct] = MySet(a_3d_point, a_list)
            self.valid[self.list_ct] = True
            self.list_ct += 1
        elif(idx1 == -1 and idx2 != -1):
            if(self.valid[idx2] and self.check_threshold(idx2, a_3d_point)):
                self.set_index[a_list[0]] = idx2
                self.set_list[idx2].union(a_3d_point, a_list)
            else:
                self.valid[idx2] = False
        elif(idx2 == -1 and idx1 != -1):
            if(self.valid[idx1] and self.check_threshold(idx1, a_3d_point)):
                self.set_index[a_list[1]] = idx1
                self.set_list[idx1].union(a_3d_point, a_list)
            else:
                self.valid[idx1] = False

        elif(idx1 != -1 and idx2 != -1):
            if(idx1 == idx2):
                if(self.valid[idx1] and self.check_threshold(idx1, a_3d_point)):
                    self.set_list[idx1].union(a_3d_point, a_list)
                else:
                    self.valid[idx2] = False
            else:
                if(self.valid[idx1] and self.valid[idx2] and self.check_threshold(idx1, a_3d_point)):
                    self.set_list[idx1].union_with(self.set_list[idx2])
                    for i in self.set_list[idx2].point2d_list:
                        self.set_index[i] = idx1                  
                    del self.set_list[idx2]
                else:
                    self.valid[idx2] = False
                    self.valid[idx1] = False
        else:
            print("should not trap here")
            pdb.set_trace()
                
                
            
