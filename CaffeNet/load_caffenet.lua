require 'loadcaffe';

path = '/Users/ashish/caffe/models/bvlc_reference_caffenet/'
model = loadcaffe.load('deploy.prototxt', 'bvlc_reference_caffenet.caffemodel', 'cudnn')

print(model)

input = torch.rand(3,227,227);
output = model:forward(input:cuda());
print(output)
