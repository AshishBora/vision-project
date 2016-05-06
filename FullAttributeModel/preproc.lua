require 'torch'
require 'image';
require 'randomkit'

loadSize = {3, 256, 256}
sampleSize = {3, 227, 227}

-------- deprecated -------
-- function loadImage(path)
-- 	local img = image.load(path, 3, 'float')
-- 	-- find the smaller dimension, and resize it to loadSize (while keeping aspect ratio)
-- 	if img:size(3) < img:size(2) then
-- 		img = image.scale(img, loadSize[2], loadSize[3] * img:size(2) / img:size(3))
-- 	else
-- 		img = image.scale(img, loadSize[2] * img:size(3) / img:size(2), loadSize[3])
-- 	end
-- 	return img
-- end

mean = {122.03348502,  119.66566533,  112.5696779}

function getTopLeftCorner(w1_mean, h1_mean, max_crop_jitter)
    w1 = math.floor(torch.uniform(w1_mean * (1 - max_crop_jitter), w1_mean * (1 + max_crop_jitter)))
    h1 = math.floor(torch.uniform(h1_mean * (1 - max_crop_jitter), h1_mean * (1 + max_crop_jitter)))
    return w1, h1
end


-- randomkit implementation : This is 17 times slower!!
-- function addNoise(out, std_dev)
--     noise = torch.Tensor(sampleSize[1], sampleSize[2], sampleSize[3])

--     outfile = io.open('train_C.out', 'a')
--     outfile:write('getting noise... ')
    
--     randomkit.normal(noise, 0, std_dev)

--     outfile:write('got noise\n')
--     outfile:close()

--     noise = noise:float()
--     out:add(noise)
--     out[torch.lt(out, 0.0)] = 0.0
--     out[torch.gt(out, 1.0)] = 1.0
-- end


-- Stil takes up 50% of the time :(
function addNoise(out, std_dev)
    noise = torch.randn(sampleSize[1], sampleSize[2], sampleSize[3])
    noise = noise:float()
    noise:mul(std_dev)
    out:add(noise)
    out[torch.lt(out, 0.0)] = 0.0
    out[torch.gt(out, 1.0)] = 1.0
end


function swap_channels(img)
    local red = img[{{1},{},{}}]:clone()
    img[{{1},{},{}}] = img[{{3},{},{}}]:clone()
    img[{{3},{},{}}] = red:clone()
    return img
end


-- function to preprocess the image
function preprocess(img, max_crop_jitter, std_dev, hflip)

    -- collectgarbage()

    local oH = sampleSize[2]
    local oW = sampleSize[3]
    local iH = img:size(2)
    local iW = img:size(3)
    local w1_mean = math.floor((iW-oW)/2)
    local h1_mean = math.floor((iH-oH)/2)

    -- random cropping
    w1, h1 = getTopLeftCorner(w1_mean, h1_mean, max_crop_jitter)
	local out = image.crop(img, w1, h1, w1+oW, h1+oH) -- center patch  

    -- -- add gaussian noise
    -- addNoise(out, std_dev)

    -- randomly decide whether to horizontally flip the image
    if hflip then
        if torch.bernoulli() == 1 then
            image.hflip(img, img)
        end 
    end
	
    -- make the range [0, 255]
	out:mul(255)

	-- subtract mean values
	for i = 1, 3 do -- channels
		out[{{i},{},{}}]:add(-mean[i])
	end

    -- swap R and B channels
    out = swap_channels(out)

    return out

end

-- base_path = '../data/SUN/SUN_WS/training/aged/'
-- itorch.image({loadImage(base_path .. 'sun_ws_aged_400.jpg')})
-- itorch.image({preprocess(base_path .. 'sun_ws_aged_400.jpg', 1, 0.1) })