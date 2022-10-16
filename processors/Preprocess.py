from typing import BinaryIO
import cv2
import numpy as np 

class Preprocess():
    def __init__(self, img):
        self.image = img.astype("uint8")

    def cvt_gray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def morphology(self, img):
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel = np.ones((3,3),np.uint8)
        morphology_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)        
        return morphology_img

    def binarization(self, img):
        ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # ret, thresh = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY)
        return thresh

    def noise_removal(self, img):
        return cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

    def auto_preprocess(self):
        # Otsu binarization
        # Sharpen? 
        # Remove Noise
        # 
        img = self.cvt_gray(self.image)
        # img = self.noise_removal(self.image)
        img = self.binarization(img)
        # img = self.morphology(binarized)
        return img
        