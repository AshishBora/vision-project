require 'loadcaffe';

require 'image'

loadSize = {3, 256, 256}
sampleSize = {3, 227, 227}

local function loadImage(path)
   local input = image.load(path, 3, 'float')
   -- find the smaller dimension, and resize it to loadSize (while keeping aspect ratio)
   if input:size(3) < input:size(2) then
      input = image.scale(input, loadSize[2], loadSize[3] * input:size(2) / input:size(3))
   else
      input = image.scale(input, loadSize[2] * input:size(3) / input:size(2), loadSize[3])
   end
   return input
end

mean = {105.84042823, 114.99835384, 118.01661205}

-- function to load the image
local function preprocess(path)

	collectgarbage()
	local input = loadImage(path)
	local oH = sampleSize[2]
	local oW = sampleSize[3]
	local iH = input:size(2)
	local iW = input:size(3)
	local w1 = math.ceil((iW-oW)/2)
	local h1 = math.ceil((iH-oH)/2)
	local out = image.crop(input, w1, h1, w1+oW, h1+oH) -- center patch

	-- make the range [0, 255]
	out:mul(255)

	-- -- mean/std
	-- for i=1,3 do -- channels
	--    if mean then out[{{i},{},{}}]:add(-mean[i]) end
	--    if std then out[{{i},{},{}}]:div(std[i]) end
	-- end

	-- subtract mean values
	for i = 1,3 do -- channels
      out[{{i},{},{}}]:add(-mean[i])
   end

   print(out)

   -- swap channels
   red = out[{{1},{},{}}]:clone()
   out[{{1},{},{}}] = out[{{3},{},{}}]:clone()
   out[{{3},{},{}}] = red:clone()

   print(out)

   return out
end

path = '/Users/ashish/caffe/models/bvlc_reference_caffenet/'
model = loadcaffe.load('deploy.prototxt', 'bvlc_reference_caffenet.caffemodel', 'cudnn')
print(model)
model:evaluate()
model:cuda()

imFolder = '/work/04001/ashishb/maverick/vision-project/testImages/'
path = imFolder .. 'cat.jpg'
input = preprocess(path)
output = model:forward(input:cuda());
print(output)
