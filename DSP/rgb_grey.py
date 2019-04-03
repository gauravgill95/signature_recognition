import numpy as np 
import cv2 as cv
from matplotlib import pyplot as plt

def dosomething(image):
	gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	img = gray
	img = cv.bilateralFilter(img,9,75,75)
	img_adptv_mean = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
	            cv.THRESH_BINARY,11,2)
	img_adptv_gauss = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
	            cv.THRESH_BINARY,11,2)
	img_adptv_gauss=  cv.bitwise_not(img_adptv_gauss)
	img_adptv_mean = cv.bitwise_not(img_adptv_mean)
	titles = ['Original Image','Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
	images = [img,img_adptv_mean,img_adptv_gauss]
	cv.imwrite( "./gray_image/input_mean.jpg", img_adptv_mean);
	cv.imwrite( "./gray_image/input_guass.jpg", img_adptv_gauss);
	for i in range(3):
	    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
	    plt.title(titles[i])
	    plt.xticks([]),plt.yticks([])
	#plt.show()
	# find contours
	binary = img_adptv_mean
	#cv.imshow('original',binary)
	#binary = (1+binary)%2
	cv.imshow('Inverted',binary)
	cv.waitKey(0)
	cv.destroyAllWindows()
	#print(cv.countNonZero(binary))
	(_, contours, _) = cv.findContours(binary, cv.RETR_EXTERNAL, 
	    cv.CHAIN_APPROX_SIMPLE)

	# print table of contours and sizes
	c_x = []
	c_y = []
	weights = []
	#print("Found %d objects." % len(contours))
	for (i, c) in enumerate(contours):
		#print("\tSize of contour %d: %d" % (i, len(c)))
		#print(cv.contourArea(c))	
		M = cv.moments(c)
		if M['m00'] !=0:
			#print(M['m00'])
			weights.append(M['m00'])
			c_x.append((M['m10']/M['m00']))
			c_y.append((M['m01']/M['m00']))
			cv.drawContours(img, contours, i, (0, 0, 255), 5)
			cv.namedWindow("output", cv.WINDOW_NORMAL)
			cv.imshow("output", img)
			cv.waitKey(0)
			cv.destroyAllWindows()
	c_x =np.array(c_x)
	c_y =np.array(c_y)
	weights = np.array(weights)
	weights_sum = np.sum(weights)
	centroid_x = 0
	centroid_y = 0 
	for i in range(len(weights)):
		centroid_x +=weights[i]*c_x[i] 
		centroid_y +=weights[i]*c_y[i]
	centroid_x /= weights_sum
	centroid_y /= weights_sum
	return weights_sum,centroid_x,centroid_y

image0 = cv.imread('./data/training/NISDCC-001_001_001_6g.PNG') 
image1 = cv.imread('./data/training/NISDCC-001_001_002_6g.PNG') 
image2 = cv.imread('./data/training/NISDCC-001_001_003_6g.PNG') 
image3 = cv.imread('./data/training/NISDCC-001_001_004_6g.PNG')
image4 = cv.imread('./data/training/NISDCC-001_001_005_6g.PNG')
for i in range(5):
	w,x,y =dosomething(eval('image'+str(i)))
	print(w," ",x," ",y)
 #draw contours over original image
# drawisplay original image with contours
#c
#cv.waitKey(0)	

