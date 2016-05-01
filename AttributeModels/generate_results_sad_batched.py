#------------------------------------------------------
# Generate results 
#------------------------------------------------------
import pylab as pltss
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io
import os.path

# Make sure that caffe is on the python path:
caffe_root = '/work/01932/dineshj/CS381V/caffe_install_scripts/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = 'deploy_sun_ws.prototxt'
PRETRAINED = 'caffe_sun_ws.caffemodel'

import os

caffe.set_mode_gpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load('sunours_mean.npy').mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

num_im = 5618
# num_im = 33
feat_vecs = zeros([num_im,4096])
results = zeros([num_im,42])
batch_size = 400

base_path = '/work/04001/ashishb/maverick/vision-project/data/SUN/SUN_WS/test/'

for j in range(num_im / batch_size + 1):
	print 'batch number', j+1, 'out of', num_im / batch_size + 1
	images = []
	start = j*batch_size
	for i in range (batch_size):
		im_num = start + i +1
		IMAGE_FILE = base_path + 'sun_ws_test_' + str(im_num) + '.jpg'
		if (os.path.isfile(IMAGE_FILE)):
			input_image = caffe.io.load_image(IMAGE_FILE)
			images.append(input_image)

	prediction = net.predict(images, oversample = False)
	# save into one major array 
	results[start:start+len(images),:] = prediction[:len(images), :]
	feat_vecs[start:start+len(images),:] = net.blobs['fc7'].data[:len(images), :]

# Print and Save 
print results.shape
print feat_vecs.shape

np.save('feat_vecs_test.npy', feat_vecs)
np.save('probs_test.npy', results)

# print 'Results Matrix  = ', results
scipy.io.savemat('probs_test', dict(results = results),appendmat=True,format='5',long_field_names=False,do_compression=False,oned_as='row')
