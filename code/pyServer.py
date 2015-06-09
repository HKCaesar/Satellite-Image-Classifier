import zerorpc
import urllib
import cStringIO
import Image
import logging
logging.basicConfig()

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as lin
import scipy.signal as sig
from PIL import Image
import glob
import matplotlib.cm as cm
import itertools

class RPC(object):
	'''pass the method a name, it replies "Hello name!"'''
	def hello(self, name):
		return "Hello, {0}!".format(name)

	def location(self, location):

	def classify(self, url, patchSize):
		file = cStringIO.StringIO(urllib.urlopen(url).read())
		img = Image.open(file)
		w, h = img.size

		# Cut off Bing logo
		cropped = img.crop((0, 0, w, h-patchSize))

		smallPatchId, largePatchId, parents = self.patch_map(cropped, patchSize)
		patches = self.get_patches(img, patchSize)

		# print len(patches[0][0][0][0][0])

		labels = []

		for y in range(0, len(patches[0])):
			for x in range(0, len(patches[0][0])):
				labels.append(self.classifyLargePatch(patches[0][y][x]))

		#labelsSmall = {}

		# for i in range(0, len(smallPatchId)):
		# 	# Check if the small patch is in the border
		# 	if (i in parents):
		# 		p = parents[i]
		# 		weightParent = [ weights[p[0]], weights[p[1]], weights[p[2]], weights[p[3]] ]

		# 		labelsSmall[i] = majority_vote(smallPatchParents, weightParents)

		size = [w, h-patchSize, patchSize]

		return np.concatenate([size, labels]).tolist()


		#return labelsSmall		

	def majority_vote(self, patches, weightParents):
		votes = [0] * len(patches)

		for i in range(0, len(patches)):
			weights = weightParents[i]
			votes[0] += weights[0]
			votes[1] += weights[1]
			votes[2] += weights[2]

		return votes.argmax(axis=0)




	def patch_map(self, img, patchSize):
		largePatchSize = patchSize

		# Divide patch in 4 smaller patches
		smallPatchSize = largePatchSize / 2

		# Pixel width and height of image
		w, h = img.size

		# Id's of the patches
		smallPatchId = {}
		largePatchId = {}

		# Number of patches in x and y dimensions
		dimSmall = {'x': w / smallPatchSize, 'y': h / smallPatchSize}
		dimLarge = {'x': w / largePatchSize, 'y': h / largePatchSize}

		################# Map patch id's to (x,y)-coordinates ###################
		smallIndex = 0
		largeIndex = 0

		for y in range(0, h, smallPatchSize):
			for x in range(0, w, smallPatchSize):
				# To compensate for 'half large patches'
				if (
					h - y >= largePatchSize and
					w - x >= largePatchSize
				):
					largePatchId[largeIndex] = {'x': x, 'y': y}
					largeIndex += 1	

				smallPatchId[smallIndex] = {'x': x, 'y': y}
				smallIndex += 1


		######### Map small patch id's to large patch id's as parents ###########
		parents = {}
		index = 0

		''' 
		Starting from the second row, second small patch, because the outer
		border is ignored. Ends at the second to last row, second to last patch.		
		'''
		for i in range(dimSmall['x'] + 2, dimSmall['x']*(dimSmall['y']-1)):
			# Skip first and last column
			if (i % dimSmall['x'] > 1):
				parents[i-1] = (index, index + 1, dimLarge['x'] + index, dimLarge['x'] + index + 1)
				index += 1
			# To compensate for 'half large patches'
			if (i % dimSmall['x'] == 0):
				index += 1

		# print parents
		return smallPatchId, largePatchId, parents

	'''
		Divides the given image in patches with dimensions [patchSize x patchSize], returns 4-dim array of patches. 

		First dimension: y position of large patch
		Second dimension: x position of large patch
		Third dimension: y position of pixel in large patch
		Fourth dimension: x position of pixel in large patch
	'''
	def get_patches(self, img, patchSize):
		########### Load Input ############################################################################################################################
		# In this script I used the brightness to determine structures, instead of one RGB color:
		# this is determined by: 0.2126*R + 0.7152*G + 0.0722*B
		# Source: https://en.wikipedia.org/wiki/Relative_luminance

		dataPatchedF=[]

		data=img.convert('RGB')
		data= np.asarray( data, dtype="int32" )
		data=0.2126*data[:,:,0]+0.7152*data[:,:,1]+0.0722*data[:,:,2]
		Yamount=data.shape[0]/patchSize # Counts how many times the windowsize fits in the picture
		Xamount=data.shape[1]/patchSize # Counts how many times the windowsize fits in the picture
		dataPatchedF.append(np.array([[data[j*(patchSize/1):(j+1)*(patchSize/1),i*(patchSize/1):(i+1)*(patchSize/1)] for i in range(0,1*Xamount-0)] for j in range(0,1*Yamount-0)]))

		return dataPatchedF

	def classifyLargePatch(self, patch):
		# patch -> NN -> result [Label, weight]
		randomWeights = np.random.rand(3)
		
		return 1
		


def main():
	s = zerorpc.Server(RPC())
	s.bind("tcp://*:4242")
	s.run()

if __name__ == "__main__" : main()