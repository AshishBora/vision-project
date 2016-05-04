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

from datetime import datetime
import glob
import os

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


base_path = '/work/04001/ashishb/maverick/vision-project/data/SUN/images/'
imPaths = glob.glob(base_path + '*/*/*.jpg')


feat_vecs = np.zeros((1, 4096))
batch_size = net.blobs['data'].data.shape[0]
num_im = len(imPaths)

print(num_im)
print(batch_size)

iter_ = 0
max_iter = len(imPaths) / batch_size
startTime = datetime.now()
while imPaths:
    iter_ += 1
    print('iteration', iter_, 'out of', max_iter)
    ims = []
    images = []
    
    for i in range(batch_size):
        if imPaths:
	    temp = imPaths.pop()
            # print('image no.', (iter_-1)*batch_size + i + 1, 'out of', num_im)
	    print(temp)
            ims.append(temp)
        else:
            break

    for im in ims:
        image = caffe.io.load_image(im)
        images.append(image)
        
    # predict takes any number of images, and formats them for the Caffe net automatically
    prediction = net.predict(images, oversample = False)
    feat = net.blobs['fc7'].data
    feat_vecs = np.vstack([feat_vecs, feat])
    print datetime.now() - startTime


feat_vecs = feat_vecs[1:num_im+1, :]
print(feat_vecs.shape)
np.save('feat_vecs_SUN_full.npy', feat_vecs)
