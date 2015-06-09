import zerorpc
import urllib
import cStringIO
import Image
import logging
logging.basicConfig()

class RPC(object):
	'''pass the method a name, it replies "Hello name!"'''
	def hello(self, name):
		return "Hello, {0}!".format(name)

	def location(self, location):
		print location

	def crop_map(self, url, patchSize):
		file = cStringIO.StringIO(urllib.urlopen(url).read())
		img = Image.open(file)
		w, h = img.size

		# Cut off Bing logo
		cropped = img.crop((0, 0, w, h-patchSize))

		return self.patch_map(cropped, patchSize)

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
		smallIndex = 1
		largeIndex = 1

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
		index = 1

		''' 
		Starting from the second row, second small patch, because the outer
		border is ignored. Ends at the second to last row, second to last patch.		
		'''
		for i in range(dimSmall['x'] + 2, dimSmall['x']*(dimSmall['y']-1)):
			# Skip first and last column
			if (i % dimSmall['x'] > 1):
				parents[i] = (index, index + 1, dimLarge['x'] + index, dimLarge['x'] + index + 1)
				index += 1
			# To compensate for 'half large patches'
			if (i % dimSmall['x'] == 0):
				index += 1

		print parents
		return parents

def main():
	s = zerorpc.Server(RPC())
	s.bind("tcp://*:4242")
	s.run()

if __name__ == "__main__" : main()