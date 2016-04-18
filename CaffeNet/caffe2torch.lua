require 'loadcaffe'

model = loadcaffe.load('deploy.prototxt', 'bvlc_reference_caffenet.caffemodel', 'cudnn')

torch.save('caffenet.t7', model)
