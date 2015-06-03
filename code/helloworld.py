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

	def crop_map(self, url):
		file = cStringIO.StringIO(urllib.urlopen(url).read())
		img = Image.open(file)
		w, h = img.size
		cropped = img.crop((0, 0, w, h-48))

def main():
	s = zerorpc.Server(RPC())
	s.bind("tcp://*:4242")
	s.run()

if __name__ == "__main__" : main()