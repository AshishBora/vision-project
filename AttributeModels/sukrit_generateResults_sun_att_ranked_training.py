# --------------------------------------------------------------------------
# Generate Results for the SUN Attributes Ranked Training Dataset 
# --------------------------------------------------------------------------
import pylab as pltss
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io
import os.path

# -----------------------------------------------------
# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# -----------------------------------------------------
# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = 'deploy_sunours.prototxt'
PRETRAINED = 'caffe_sun_softmax_iter_48000.caffemodel'

import os

caffe.set_mode_gpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean = np.load(caffe_root + 'python/sunours_mean.npy').mean(1).mean(1),
                       channel_swap = (2,1,0),
                       raw_scale = 255,
                       image_dims = (256, 256))

results = zeros([2035,42])
k = 0
for i in xrange (1,4070,2):
	IMAGE_FILE = 'sun_attributes_ranked_training/SUN_TEST_' + str(i) + '.jpg'
	if (os.path.isfile(IMAGE_FILE)):
		input_image = caffe.io.load_image(IMAGE_FILE)
		# plt.imshow(input_image)
		# show()

		# predict takes any number of images, and formats them for the Caffe net automatically
		prediction = net.predict([input_image])
		# print 'predictions:', prediction[0]
		
		print '----------PROGRESS - DONE IMAGE = ', i

		# save into one major array 
		results[k,:] = prediction[0]
		k = k + 1
       
# ----------------------------------------------------- 
# Print and Save 
print 'Results Matrix Shape = ', results.shape
print 'Results Matrix  = ', results
scipy.io.savemat('results_sun_att_ranked_training',
				  dict(results = results),
				  appendmat = True,
				  format = '5',
				  long_field_names = False,
				  do_compression = False,
				  oned_as = 'row')










