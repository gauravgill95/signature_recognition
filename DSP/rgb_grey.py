import numpy as np 
import cv2 as cv
from matplotlib import pyplot as plt

image = cv.imread('./uploads/input.png') 
cv.imshow('Original image',image)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imwrite( "./gray_image/input.jpg", gray);
img = gray
img = cv.bilateralFilter(img,9,75,75)
img_adptv_mean = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)
img_adptv_gauss = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)
titles = ['Original Image','Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img,img_adptv_mean,img_adptv_gauss]
for i in range(3):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()