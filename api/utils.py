import cv2
import os

orb = cv2.ORB_create()

def computeD(nparr):

    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    imgpath = os.path.join(static_dir, 'reference_images')  

    smallestD = float('inf')
    #imgpath = 'static'

    #img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    img = cv2.cvtColor(nparr, cv2.COLOR_BGR2GRAY)

    keypoints2, des2 = orb.detectAndCompute(img, None)

    #ORB feature extraction
    for filename in os.listdir(imgpath):
        if filename.endswith('.jpg'):

            image_path = os.path.join(imgpath, filename)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            keypoints, des1 = orb.detectAndCompute(img, None)
            
            #matching: bruteforce method
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x:x.distance)
            smallest = matches[0].distance

            if smallestD > smallest:
                smallestD = smallest
                name = filename
    
    smallestD = str(smallestD)

    return smallestD, name