import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os  
import pickle
import argparse

def rectify(h):

	h = h.reshape((4,2))
	hnew = np.zeros((4,2),dtype = np.float32)
	
	hnew2 = np.zeros((4,2),dtype = np.float32)
	h2 = list(reversed(h))
	
	for i in range(len(h2)):
		hnew2[i] = h[i]
		
	add = h.sum(1)
	hnew[0] = h[np.argmin(add)]
	hnew[2] = h[np.argmax(add)]	   
	diff = np.diff(h,axis = 1)
	hnew[1] = h[np.argmin(diff)]
	hnew[3] = h[np.argmax(diff)]
	
	#print hnew
	#print hnew2
	
	return hnew

def save_obj(obj, name):
	with open('obj/'+ name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
	with open('obj/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)



def extraction_carte(image):

	#On transorme en noir et blanc
	image_grise = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	#On applique le blur - gaussian 
	image_blur = cv2.GaussianBlur(image_grise,(55,55),0)
	#image_blur = cv2.bilateralFilter ( image_grise, 15, 80, 80);
	#On applique un filtre passe-haut
	#imx = cv2.resize(image_blur,(1000,600))
	
	_,thresh = cv2.threshold(image_blur, 120, 255, cv2.THRESH_BINARY)
	#Extraction des contours
	imx = cv2.resize(thresh,(1000,600))

	contours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	#On classe le contour en ordre de grandeur -- On veut seulement garder les cartes, pas les formes
	contours = sorted(contours, key=cv2.contourArea,reverse=True)
	#cv2.drawContours(image, contours, -1, (0,255,0), 20)
	#imx = cv2.resize(image,(1000,600))
	#cv2.imshow('a',imx) 
	#cv2.waitKey()
	
	#Ici 0.3 est totalement abitraire..
	area_min = cv2.contourArea(contours[0]) * 0.3
	
	for card in contours:
		#Calcul du perimetre de la carte
		peri = cv2.arcLength(card,True)
		area = cv2.contourArea(card)

		#On veut seulement afficher le contour des cartes!
		if area < area_min:
			print "Break"
			break
		
		
		
		
		#On cherche a approximer un rectangle pour chaque cartes
		#Utilisation de Ramer-Douglas-Peucker_algorithm
		approx = cv2.approxPolyDP(card,0.02*peri,True)
		#print approx
		
		
		try:
			approx = rectify(approx);
		except ValueError:		
			print 'Attention la forme fournie est pas rectangulaire'
			#cv2.imshow('a',imx) 
			#cv2.waitKey()
		else:
			box = np.int0(approx)
			#print approx
			#cv2.drawContours(image,[box],0,(255,255,0),6)
			#imx = cv2.resize(image,(1000,600))
			#cv2.imshow('a',imx) 
			#cv2.waitKey()
			
			h = np.array([ [0,0],[449,0],[449,449],[0,449] ],np.float32)
			transform = cv2.getPerspectiveTransform(approx,h)
			warp = cv2.warpPerspective(image,transform,(450,450))
			
			#imx = cv2.resize(warp,(1000,600))
			#cv2.imshow('a',warp) 
			#cv2.waitKey()
			
			
			yield (warp,box)
		


def creer_database():

	sift = cv2.SIFT()

	database = {}
	for fn in os.listdir('./database'):
		image = cv2.imread("database/" + fn)
		card, _ = extraction_carte(image).next();
		
		kp, desc = sift.detectAndCompute(card,None)
		
		#imx=cv2.drawKeypoints(card,kp)
		#imx = cv2.resize(imx,(1000,600))
		#cv2.imshow('a',imx) 
		#cv2.waitKey()
		
		database[fn.split('.')[0]] = desc
	
	save_obj(database, "database")
	
	
	
		
def match_test(filename):
	
	sift = cv2.SIFT()
	bf = cv2.BFMatcher()
	
	
	database = load_obj("database")
	

	
	image = cv2.imread(filename)
	if image is None :
		print "Invalid image name"
		return
		
	#premiere loop pour les cartes dans l'image
	for i,(c,box) in enumerate(extraction_carte(image)):
		
		kp, desc = sift.detectAndCompute(c,None)
		max_match = 0;
		card="Aucun match"
		for key, value in database.iteritems():
			
			matches = bf.knnMatch(value,desc, k=2)

			# Apply ratio test
			good = []
			for m,n in matches:
				if m.distance < 0.4*n.distance:
					good.append([m])
			
			if len(good) > max_match:
				max_match = len(good)
				card = key
		
		
		#img=cv2.drawKeypoints(c,kp)
		#imx = cv2.resize(img,(1000,600))
		#cv2.imshow('a',img) 
		#cv2.waitKey()

		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.drawContours(image,[box],0,(0,255,0),20)
		cv2.putText(image,card,(box[0][0],box[0][1]), font, 6,(0,255,0),7)
		
	imx = cv2.resize(image,(1000,600))
	cv2.imshow('a',imx) 
	cv2.waitKey()

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Card Detector')
	parser.add_argument("--input", help="Name of the input picture")
	parser.add_argument("--createDb", help="Create the database", action="store_true")
	args = parser.parse_args()


	if(args.input is None and not args.createDb):
		print "Missing arguments"
	if(args.createDb):
		creer_database()
	if(args.input != None):
		match_test(args.input)

   

