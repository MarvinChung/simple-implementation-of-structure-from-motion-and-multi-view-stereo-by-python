import numpy
import math
import pdb

class GlobalSets(object):
    def __init__(self, threshold = 10.0):
        #mask_list shows which sets_list are legal.
        #use mask_list to check which sets_list can use
        self.mask_list = {}
        self.sets_list = []
        
        #self.indices store which set the 2d point beyonds to.
        self.indices = {}
        
        self.world_points = {}
        self.threshold = threshold
    
    
    def get_n_observations(self):
        """
        return the number of 2d points
        """
        ct = 0
        for i, _set in enumerate(self.sets_list):
            if(self.mask_list[i]==1):
                ct += len(_set)
        return ct
    
    def add(self, a_set, a_3d_point):
        """
        a_set should be python set type
        Input form:
            a_set:
                image_index is same as camera index
                {(image_index1, x1, y1), (image_index2, x2, y2)}
            a_3d_point:
                array([x, y, z], dtype=int32)
       
        a_set should a pair of points that match from two images.
        for example:
        After main program proccessed image 1,2
        if global sets:
                {(1,x1,y1), (2,x'1,y'1)}
                {(1,x2,y2), (2,x'2,y'2)}
                           .
                {(1,xn,yn), (2,x'n,y'n)}
        Then main program is handling image 1,3
        there will be set {(1, x1, y1), (3, x2, y2)} should add to global sets.
        If both of the tuples have not been seen then it should be add to the global sets.
        If one of them have been seen then should be union to the specific set if satisfies the threshold test.
        
        merge pair at bundle adjustment:
            {world_points[i], sets_list[i]}
            
            where world_points[i] is an array
        
        Special case1:
            global sets:
                {(1,a,b), (2,x'1,y'1)}
                {(3,c,d), (4,x'1,y'1)}
            a_set:
                {(1,a,b), (3,c,d)}
            
            Then global sets:
                {(1,a,b), (2,x'1,y'1), (3,c,d), (4,x'1,y'1)}
                
        Special case2:
            global sets:
                {(0, 425, 82), (1, 421, 81)}
            a_set:
                {(0, 424, 81), (1, 421, 81)}
                
            This may happened due to the points are float originally when counting matches, then force to become int.
            These points will be seen as 1 point
            
            Then global sets:
                {(0, 424, 81), (0, 425, 82), (1, 421, 81)}
        """
        which_index = []
        print("a_set",a_set)
        #input("")
        for i in a_set:
            which_index.append(self.__find_a_element_in_which_set(i))
            
        if(which_index[0] is not None and which_index[1] is not None):
            if(self.__check_threshold(which_index[0], a_3d_point) and self.__check_threshold(which_index[1], a_3d_point)):
                self.__union_old_set(which_index[0], which_index[1])
                self.__union_new_set(which_index[0], a_set, a_3d_point)
        elif which_index[0] is not None:
            if(self.__check_threshold(which_index[0], a_3d_point)):
                self.__union_new_set(which_index[0], a_set, a_3d_point)
        elif which_index[1] is not None:
            if(self.__check_threshold(which_index[1], a_3d_point)):
                self.__union_new_set(which_index[1], a_set, a_3d_point)
        else:
            self.__add_in_sets_list(a_set, a_3d_point)
            
    #double underscore is to remind these methods should be private, although python does not have private method
    def __find_a_element_in_which_set(self, element_of_a_set):
        print('find', element_of_a_set)
        if(element_of_a_set not in self.indices):
            print("not seen yet")
            return None
        else:
            if self.mask_list[self.indices[element_of_a_set]]==0:
                print("wtf")
                pdb.set_trace()
            print("have been seen:",self.indices[element_of_a_set])
            return self.indices[element_of_a_set]
    
    def __add_in_sets_list(self, added_set, a_3d_point):
        """
        input a set with tuples
        ex:
         {(image_index,x,y), ......}
        """
        self.sets_list.append(added_set)
        print("add in sets list")
        
        
        #make find_which_set to be O(1)
        for i in added_set:
            self.indices[i] = len(self.sets_list)-1
            
        self.mask_list[len(self.sets_list)-1] = 1
        self.world_points[len(self.sets_list)-1] = a_3d_point  
    def __union_old_set(self, set_idx1, set_idx2):
        """
        Use __find_a_element_in_which_set(an element inside a set) to find set index.
        then set2 will be merged into set1
        """
        if(set_idx1 != set_idx2):
            try:
                print("union old set")
                input("")
                self.world_points[set_idx1] = (self.world_points[set_idx1]* len(self.sets_list[set_idx1]) + self.world_points[set_idx2]* len(self.sets_list[set_idx2]))/(len(self.sets_list[set_idx1])+len(self.sets_list[set_idx2]))
            except:
                print(set_idx1,"is inside:",set_idx1 in self.mask_list)
                print(set_idx2,"is inside:",set_idx2 in self.mask_list)
                raise
            #print("before union")
            #print(set_idx1," ",set_idx2)
            #print(self.sets_list[set_idx1])
            #print(self.sets_list[set_idx2])
            self.sets_list[set_idx1] = self.sets_list[set_idx1].union(self.sets_list[set_idx2])
            #print("after union")
            #print(self.sets_list[set_idx1])
            #print(self.sets_list[set_idx2])

            for i in self.sets_list[set_idx2]:
                self.indices[i] = set_idx1
            
            self.mask_list[set_idx2] = 0           
            #del self.world_points[set_idx2]
        
    def __union_new_set(self, set_idx, a_set, a_point):
        self.world_points[set_idx] = (self.world_points[set_idx] * len(self.sets_list[set_idx]) + a_point)/(len(self.sets_list[set_idx])+1)
                            
        self.sets_list[set_idx] = self.sets_list[set_idx].union(a_set)
        for i in a_set:
            self.indices[i] = set_idx
                   
    def __check_threshold(self, set_idx, b):
        return True
#         print("check threshold?",b)
#         print(self.sets_list[set_idx])
#         print("set_idx",set_idx)
#         try:
#             a = self.world_points[set_idx]
#         except KeyError:
#             pdb.set_trace()
#         return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2) < self.threshold