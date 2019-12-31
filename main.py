#from PIL import Image
#from QuickSets import GlobalSets
from GlobalSet import GlobalSet
from argparse import ArgumentParser
from MVS import *
from utils import *
from SFM import *

def read_imgs(args):
    files = []
    print("read images from " + args.img_dir+"/*."+args.img_type)
    for file in glob.glob(args.img_dir+"/*."+args.img_type):      
        files.append(file)        
    files.sort()
    print(files)
    
    imgs = []
    for file in files:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
    return imgs

def main(args, threshold = 0.01, MIN_REPROJECTION_ERROR = 0.3):
    scale = args.scale
    imgs = read_imgs(args)

        
    global_set = GlobalSet(threshold = threshold)    
        
    StructureFromMotion(imgs[:2], global_set, args, MIN_REPROJECTION_ERROR)
    #Test2MethodsOfDensePointsWithTwoViewStereo(imgs, args)
    DensePointsWithMVS(imgs[:2], global_set, args) 

if __name__== "__main__":
    parser = ArgumentParser()
    parser.add_argument("-img_p", help="image directory", dest="img_dir", default=None)
    parser.add_argument("-par_p", help="parameter path", dest="par_path", default=None)
    parser.add_argument("-t", help="image file type", dest="img_type", default="ppm")
    parser.add_argument("-scale", help="scale", dest="scale", default=1, type=float)
    parser.add_argument("--debug", help="debug mode on", dest="debug", action='store_true')
    parser.add_argument("-Sequence", help="", dest="isSeq", default=1, type=int)
    parser.add_argument("-cell_size", help="", dest="cell_size", default=5, type=int)
    parser.add_argument("-desc_wid", help="", dest="desc_wid", default=5, type=int)
    args = parser.parse_args()
    try:
        main(args)
    except RuntimeError:
        print("")