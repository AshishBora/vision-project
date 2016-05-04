require 'image';

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

mean = {112.5696779 ,  119.66566533,  122.03348502}

function getTopLeftCorner( w1_mean, h1_mean, max_crop_jitter )
    w1 = math.ceil( torch.uniform( w1_mean * ( 1 - max_crop_jitter ), w1_mean * ( 1 + max_crop_jitter ) ) )
    h1 = math.ceil( torch.uniform( h1_mean * ( 1 - max_crop_jitter ), h1_mean * ( 1 + max_crop_jitter ) ) )
    return w1, h1
end

function addNoise( out, std_dev )
    local noise = torch.randn( sampleSize[1], sampleSize[2], sampleSize[3] )
    noise = noise:float()
    noise:mul( std_dev )
    out:add( noise )
    out[torch.lt( out, 0.0 )] = 0.0
    out[torch.gt( out, 1.0 )] = 1.0
end
-- function to preprocess the image
function preprocess(path, max_crop_jitter, std_dev)

	collectgarbage()
	local input = loadImage(path)
	local oH = sampleSize[2]
	local oW = sampleSize[3]
	local iH = input:size(2)
	local iW = input:size(3)
	local w1_mean = math.ceil((iW-oW)/2)
	local h1_mean = math.ceil((iH-oH)/2)
    w1, h1 = getTopLeftCorner( w1_mean, h1_mean, max_crop_jitter )
	local out = image.crop(input, w1, h1, w1+oW, h1+oH) -- center patch  
    
    addNoise( out, std_dev )
    
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

-- base_path = '../data/SUN/SUN_WS/training/aged/'
-- itorch.image({loadImage(base_path .. 'sun_ws_aged_400.jpg')})
-- itorch.image({preprocess(base_path .. 'sun_ws_aged_400.jpg', 1, 0.1 ) } )