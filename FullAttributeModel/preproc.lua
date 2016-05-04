require 'image';
require 'cunn';

loadSize = {3, 256, 256}
sampleSize = {3, 227, 227}

function loadImage(path)
	local input = image.load(path, 3, 'float')
	-- find the smaller dimension, and resize it to loadSize (while keeping aspect ratio)
	if input:size(3) < input:size(2) then
		input = image.scale(input, loadSize[2], loadSize[3] * input:size(2) / input:size(3))
	else
		input = image.scale(input, loadSize[2] * input:size(3) / input:size(2), loadSize[3])
	end
	return input
end

-- np.load('sunours_mean.npy').mean(1).mean(1) gives this
-- score achived = 0.5790
-- mean = {112.5696779 ,  119.66566533,  122.03348502}

-- testing channel swapping
-- score achieved = 0.5808
mean = {122.03348502,  119.66566533,  112.5696779}

-- function to preprocess the image
function preprocess(path, max_crop_jitter, std)

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

	-- subtract mean values
	for i = 1,3 do -- channels
		out[{{i},{},{}}]:add(-mean[i])
	end

   -- swap channels
   red = out[{{1},{},{}}]:clone()
   out[{{1},{},{}}] = out[{{3},{},{}}]:clone()
   out[{{3},{},{}}] = red:clone()

   return out

end