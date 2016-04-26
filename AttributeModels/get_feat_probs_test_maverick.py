
import numpy as np
import caffe
import os
from datetime import datetime
import glob

# -----------------------------------------------------
# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = 'deploy_sun_ws.prototxt'
PRETRAINED = 'caffe_sun_ws.caffemodel'

print('Hi 1')
caffe.set_mode_gpu()

print('Hi 2')
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean = np.load('sunours_mean.npy').mean(1).mean(1),
                       channel_swap = (2,1,0),
                       raw_scale = 255,
                       image_dims = (256, 256))

print('Hi 3')
base_path = '/work/04001/ashishb/maverick/vision-project/data/SUN/SUN_WS/test/'
imPaths = glob.glob(base_path + '*.jpg')

print('Hi 4')
feat_vecs = np.zeros((1, 4096))
probs = np.zeros((1, 42))
batch_size = net.blobs['data'].data.shape[0]
num_im = len(imPaths)

print('Hi 5')
print(num_im)
print(batch_size)

GlobOrder = []
for im_path in imPaths:
    GlobOrder.append(int(im_path.split('_')[-1].split('.')[0]))

imPaths2 = []
for i in range(len(imPaths)):
    index = GlobOrder.index(i+1)
    imPaths2.append(imPaths[index])

imPaths = imPaths2

## FOR DEBUG ONLY
imPaths = imPaths[:40]

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
            print('image no.', (iter_-1)*batch_size + i + 1, 'out of', num_im)
            ims.append(imPaths.pop())
        else:
            break

    for im in ims:
        image = caffe.io.load_image(im)
        images.append(image)
        
    # predict takes any number of images, and formats them for the Caffe net automatically
    prediction = net.predict(images)
    feat = net.blobs['fc7'].data
    prob = net.blobs['prob'].data
    feat_vecs = np.vstack([feat_vecs, feat])
    probs = np.vstack([probs, prob])
    print datetime.now() - startTime

feat_vecs = feat_vecs[1:num_im+1, :]
probs = probs[1:num_im+1, :]

print(feat_vecs.shape)
print(probs.shape)

np.save('feat_vecs_test.npy', feat_vecs)
np.save('probs_test.npy', probs)