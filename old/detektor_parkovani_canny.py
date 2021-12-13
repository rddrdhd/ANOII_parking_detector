#!/usr/bin/python
# groundtruth.txt - 
# parking_map_python.txt = souřadnice parkovacích míst
# pro detekci (např. detektor hran - histogram or. grad, sobel, canny) - můžeme dělat variantu i bez trénování , i s trénováním 
# 3stránkový (max 5)reeport o tom co jsme použili ajakou máme úspěšnost - co mi pomohlo, nepomohlo.....
# příště 21.9. uděláme počítadlo jak se daří našemu dektektoru
import sys
import cv2
import numpy as np
import math
import struct
from datetime import datetime
import glob
#TODO vykreslit do obrazu ?
#TODO co bychom s tím edge_image mohli udělat, abychom poznali že je tam auto? Pomocí nějaké jednoduché hodnoty - např. spočítat pixely - auta mají víc.
def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

def main(argv):
#vytvořím okno
    cv2.namedWindow("blur_image", 0)
    cv2.namedWindow("res_image", 0)
    cv2.namedWindow("edge_image", 0)

#načtu souřadnice
    pkm_file = open('parking_map_python.txt', 'r')
    pkm_lines = pkm_file.readlines()
    pkm_coordinates = []

    for line in pkm_lines:
        st_line = line.strip()
        sp_line = list(st_line.split(" "))
        pkm_coordinates.append(sp_line)

#projdu je přes testovací obrázky
    test_images = [img for img in glob.glob("test_images/*.jpg")]
    test_images.sort()
    for img in test_images:
		#přečtu celý obraz
        one_park_image = cv2.imread(img)
        for one_c in pkm_coordinates:
            pts = [((float(one_c[0])), float(one_c[1])),
                    ((float(one_c[2])), float(one_c[3])),
                    ((float(one_c[4])), float(one_c[5])),
                    ((float(one_c[6])), float(one_c[7]))] 
            #print(pts)
            #https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
			#vyrovnám a vyřežu parkovací místo podle bodů
            warped_image = four_point_transform(one_park_image, np.array(pts))
			#resize narovnaného obrazu
            res_image = cv2.resize(warped_image, (80, 80))
            #vymažu šum
            blur_image = cv2.GaussianBlur(res_image,(3,3),0)
            #převedu do černobílého
            gray_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2GRAY)
			#cannyho detektor hran - hodnoty jsou teď random
            edge_image = cv2.Canny(gray_image, 40, 120)           
            
			#zobrazím obrázky
            cv2.imshow('blur_image', blur_image)
            cv2.imshow('res_image', res_image) 
            cv2.imshow('edge_image', edge_image)
            n_white_pix = np.sum(edge_image == 255)
            point_1 = (int(one_c[0]),int(one_c[1]))
            point_2 = (int(one_c[2]),int(one_c[3]))
            point_3 = (int(one_c[4]),int(one_c[5]))
            point_4 = (int(one_c[6]),int(one_c[7]))
            if n_white_pix > 500:
                #print('Auto! Má totiž tolik pixelů hran::', n_white_pix)
                my_color=(0, 0, 255)
            else:
                #print('Prádné! Má totiž tolik pixelů hran:', n_white_pix)
                my_color=(0, 255, 0)
            #for point in [point_1, point_2, point_3, point_4]:
                #cv2.circle(one_park_image, point, radius=1, color=my_color, thickness=-1)
            cv2.circle(one_park_image,[x+20 for x in point_1], 10, my_color)
            #cv2.rectangle(one_park_image, point_1, point_3, my_color, 3)
            #cv2.waitKey(0)      
            #roi = img[y:y+h, x:x+w]
        cv2.imshow('one_park_image', one_park_image)
        key = cv2.waitKey(0)
        if key == 27: # exit on ESC
            break
    
if __name__ == "__main__":
   main(sys.argv[1:])     
