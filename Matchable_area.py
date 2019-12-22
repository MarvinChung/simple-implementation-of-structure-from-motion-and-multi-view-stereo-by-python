import cv2
import matplotlib.pyplot as plt
from skimage import filters

def non_Matchable_area(img , debug = False, grad_threshold = 0.01):
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges_x = filters.sobel_h(gray_img)
    edges_y = filters.sobel_v(gray_img)
    edges = filters.sobel(gray_img)
    if debug == True:
        plt.subplot(221)                 
        plt.imshow(edges_x)
        plt.title('sobel_x', size=20)
        
        plt.subplot(222)
        plt.imshow(edges_y)
        plt.title('sobel_y', size=20)
        
        plt.subplot(223)
        plt.imshow(edges)
        plt.title('sobel', size=20)

        plt.subplot(224)
        plt.title('gradient > 0.01')
        plt.imshow(edges > grad_threshold)
        plt.show()
    return (edges <= grad_threshold)

def Matchable_area(img , debug = False, grad_threshold = 0.01):
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges_x = filters.sobel_h(gray_img)
    edges_y = filters.sobel_v(gray_img)
    edges = filters.sobel(gray_img)
    if debug == True:
        plt.subplot(221)                 
        plt.imshow(edges_x)
        plt.title('sobel_x', size=20)
        
        plt.subplot(222)
        plt.imshow(edges_y)
        plt.title('sobel_y', size=20)
        
        plt.subplot(223)
        plt.imshow(edges)
        plt.title('sobel', size=20)

        plt.subplot(224)
        plt.title('gradient > 0.01')
        plt.imshow(edges > grad_threshold)
        plt.show()
    return (edges > grad_threshold)