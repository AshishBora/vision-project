require 'loadcaffe'

sun_ws = loadcaffe.load('deploy_sun_ws.prototxt', 'caffe_sun_ws.caffemodel', 'cudnn')

torch.save('sun_ws.t7', sun_ws)
