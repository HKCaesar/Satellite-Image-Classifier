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
from itertools import product

class RPC(object):
	'''pass the method a name, it replies "Hello name!"'''
	def hello(self, name):
		return "Hello, {0}!".format(name)

	def classify(self, url, patchSize):
		img = self.crop_image(url, patchSize)
		w, h = img.size

		smallPatchId, largePatchId, parents = self.patch_map(img, patchSize)
		patches = self.get_patches(img, patchSize)

		matrix = self.classifyImage(img, patchSize)

		serialized = self.serialized_matrix(matrix)

		result = np.concatenate([[w, h, patchSize], serialized])

		return result.tolist()

	def crop_image(self, url, patchSize):
		file = cStringIO.StringIO(urllib.urlopen(url).read())
		img = Image.open(file)
		w, h = img.size
		
		# Cut off Bing logo
		return img.crop((0, 0, w, h-patchSize))


	def majority_vote(self, patches, weightParents):
		votes = [0] * len(patches)

		for i in range(0, len(patches)):
			weights = weightParents[i]
			votes[0] += weights[0]
			votes[1] += weights[1]
			votes[2] += weights[2]

		return votes.argmax(axis=0)


	def serialized_matrix(self, matrix):
		labels = []

		for y in range(0, matrix.shape[0]):
			for x in range(0, matrix.shape[1]):
				label = np.argmax(matrix[y,x])
				labels.append(label)

		return labels


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

	def classifyImage(self, img, patchSize):
		#################### Define pooling layers ###########################################################################
		P12=Pool_node(4)*(1.0/1.000) #factor 1000 added to lower values more
		P34=Pool_node(1)*(1.0/1.0) 

		#################### Define Convolution layers #######################################################################

		######### First C layer #########
		C1=[]
		Kernelsize=9
		## First Kernel

		# Inspiration: http://en.wikipedia.org/wiki/Sobel_operator
		# http://stackoverflow.com/questions/9567882/sobel-filter-kernel-of-large-size

		## First kernel
		Kernel=np.zeros((Kernelsize,Kernelsize))
		for i in range(0,Kernelsize):
			for j in range(0,Kernelsize):
				if(i==j):Kernel[i,j]=1
				if(i>j): Kernel[i,j]=5
				if(i<j): Kernel[i,j]=-5
					
		C1.append(Kernel*(1.0/100.0))


		Kernel=np.array([[4,3,2,1,0,-1,-2,-3,-4],
						 [5,4,3,2,0,-2,-3,-4,-5], 
						 [6,5,4,3,0,-3,-4,-5,-6],
						 [7,6,5,4,0,-4,-5,-6,-7], 
						 [8,7,6,5,0,-5,-6,-7,-8],
						 [7,6,5,4,0,-4,-5,-6,-7],
						 [6,5,4,3,0,-3,-4,-5,-6],
						 [5,4,3,2,0,-2,-3,-4,-5],
						 [4,3,2,1,0,-1,-2,-3,-4]])

		C1.append(Kernel*(1.0/100.0))

		## Fourth Kernel
		Kernel=np.zeros((Kernelsize,Kernelsize))
		Kernel[0:4,:]=-5
		Kernel[4:5,:]=0
		Kernel[5:9,:]=5    
		C1.append(Kernel*(1.0/100.0))
				

		## kernel
		Kernel=np.zeros((Kernelsize,Kernelsize))
		Kernel[3:6,3:6]=-2 
		Kernel[4,4]=8
		C1.append(Kernel*(1.0/100.0))

		## kernel
		Kernel=np.transpose(C1[2])
		C1.append(Kernel)


		######### Initialize output weights and biases #########

		# Initialisation, since this layer should be trained!
		N_branches=1
		ClassAmount=3 # Forest, City, Water


		import pickle
		file=open('W_michiel.txt','r')
		W=pickle.load(file)
		file=open('C2_michiel.txt','r')
		C2=pickle.load(file)
		file=open('bias_michiel.txt','r')
		bias=pickle.load(file)
		file=open('H3_bias_michiel.txt','r')
		H3_bias=pickle.load(file)

		####### Test phase on new images #######
		Error_Test=[]
		N_correct=0

		data=img.convert('RGB')
		data= np.asarray( data, dtype="int32" )
		data=0.2126*data[:,:,0]+0.7152*data[:,:,1]+0.0722*data[:,:,2]
		Yamount=data.shape[0]/patchSize # Counts how many times the windowsize fits in the picture
		Xamount=data.shape[1]/patchSize # Counts how many times the windowsize fits in the picture
			
			
		Patches=np.array([[data[y*patchSize:(y+1)*patchSize,  x*patchSize:(x+1)*patchSize] for x in range(0,Xamount)] for y in range(0,Yamount)]) 

		###### Chooses patch and defines label #####
		#for PP in range(0,len(Sequence)):
		forest=0
		city=0
		grass=0
		cityImg=[]
		forestImg=[]

		inputPatch=np.zeros((patchSize,patchSize))
		Classifier_array=np.zeros((len(Patches[:,0,0,0]),len(Patches[0,:,0,0]),3))
		for i in range(0,len(Patches[:,0,0,0])):
			for j in range(0,len(Patches[0,:,0,0])):
				inputPatch=Patches[i,j]
				### Layer 1 ###
				H1=[]
				H2=[]
				H3=[[np.zeros((4,4)) for b in range(0,N_branches)] for r in range(0,len(C1))]
				H4=[[np.zeros((4,4)) for b in range(0,N_branches)] for r in range(0,len(C1))]
				I_H3=np.ones((4,4))
				x=np.zeros(ClassAmount)
				f=np.zeros(ClassAmount)

				for r in range (0, len(C1)):
					H1.append(sig.convolve(inputPatch, C1[r], 'valid'))
					H2.append(Pool(H1[r], P12))
					for b in range(0,N_branches):
						H3[r][b]=Sigmoid(sig.convolve(H2[r], C2[r][b],'valid')-H3_bias[r][b]*I_H3)
						H4[r][b]=Pool(H3[r][b],P34)

				for k in range(0,ClassAmount):
					for r in range (0, len(C1)):
						for b in range(0,N_branches):
							x[k]=x[k]+np.inner(H4[r][b].flatten(),W[r][b,k].flatten())
					f[k]=Sigmoid(x[k]-bias[k])

				if(np.argmax(f)==0):
					forest+=1.0
					forestImg.append(inputPatch)
				if(np.argmax(f)==1):
					city+=1.0
					cityImg.append(inputPatch)
				if(np.argmax(f)==2):grass+=1.0     
				Classifier_array[i,j]=f/np.sum((f))

		return Classifier_array  
		


def main():
	s = zerorpc.Server(RPC())
	s.bind("tcp://*:4242")
	s.run()

########### Functions ############################################################################################################################

# Define Activitation functions, pooling and convolution functions (the rules)

def Sigmoid(x): 
	return (1/(1+np.exp(-x)))

def Sigmoid_dx(x):
	return np.exp(-x)/((1+np.exp(-x))**2)

def TanH(x):
	return (1-np.exp(-x))/(1+np.exp(-x))


def Pool(I,W):
	PoolImg=np.zeros((len(I)/len(W),len(I)/len(W))) # W must fit an integer times into I.
	for i in range(0,len(PoolImg)):
		for j in range(0,len(PoolImg)):
			SelAr=I[i*len(W):(i+1)*len(W),j*len(W):(j+1)*len(W)]
			PoolImg[i,j]=np.inner(SelAr.flatten(),W.flatten()) # Now this is just an inner product since we have vectors
	return PoolImg

# To automatically make Gaussian kernels
def makeGaussian(size, fwhm = 3, center=None):
	x = np.arange(0, size, 1, float)
	y = x[:,np.newaxis]

	if center is None:
		x0 = y0 = size // 2
	else:
		x0 = center[0]
		y0 = center[1]

	return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

# To automatically define pooling nodes
def Pool_node(N):
	s=(N,N)
	a=float(N)*float(N)
	return (1.0/a)*np.ones(s) 

if __name__ == "__main__" : main()
