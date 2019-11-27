#!/usr/bin/env python
 
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
 
def build_filters():
 filters = []
 ksize = 31
 for theta in np.arange(0, np.pi, np.pi / 16):
 	kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
 	kern /= 1.5*kern.sum()
 	filters.append(kern)
 
 return filters
 
def process(img, filters):
 accum = np.zeros_like(img)
 for kern in filters:
 	fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
 	np.maximum(accum, fimg, accum)
 
 return accum
 
if __name__ == '__main__':
 import sys
 
 print (__doc__)
 try:
 	img_fn = sys.argv[1]
 except:
 	img_fn = 'test.png'
 
 img = cv2.imread(img_fn)
 if img is None:
 	print ('Failed to load image file:', img_fn)
 	sys.exit(1)
 
 # resized
 new_w, new_h = 640, 480
 ''' interpolation: 內插法選項設定，內差的意思是: 放大或縮小的過程中，刪除或者新增 pixels 的演算法'''
 resized_img = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_AREA)

 # median blur for de-noise
 # resized_img = cv2.medianBlur(resized_img,7) 

 filters = build_filters()

 start = time.process_time() 
 res1 = process(resized_img, filters)
 end = time.process_time()
 print ('elapsed time: ', "{:4.4f}".format(end-start))

 # median blur for de-noise
 # res1 = cv2.medianBlur(res1, 7) 

 cv2.imshow('Orignal', resized_img)
 cv2.imshow('Garbor',  res1)
 
 cv2.waitKey(0)
 cv2.destroyAllWindows()


 '''
 plt.subplot(121),plt.imshow(resized_img), plt.title('Original')
 plt.xticks([]), plt.yticks([])

 plt.subplot(122),plt.imshow(res1),plt.title('Feature')
 plt.xticks([]), plt.yticks([])

 plt.show()
 '''

 # save image
 # cv2.imwrite("/Users/simon/Desktop/MMS/resized.jpg", resized_img)
 # cv2.imwrite("/Users/simon/Desktop/MMS/texture.jpg", res1)

