require 'torch'
require 'nngraph'
require 'nn'
require 'lmdb'
require 'image'

dofile('func_lib.lua')
dofile('preproc.lua')

-- function to get an example for training C
get_example_C = function(reader, attrs, num_im)

    -- randomly select two images from different classes
    local y = torch.randperm(num_im)

    local images = {}
    images[1] = reader:get(y[1])
    images[2] = reader:get(y[2])

    -- randomly select one of those to be given to model B
    local label = torch.bernoulli() + 1
    images[3] = images[label]:clone()

    -- randomly select one of the images to ask about its attrsibute
    local ques = torch.Tensor(42):fill(0)
    ques[attrs[y[torch.bernoulli() + 1]]] = 1
    ques = ques:view(1, -1)

    -- target is probability of label = 2
    local target = label - 1

    -- random crops and gaussian noise
    -- serialize
    for i = 1, 3 do
        images[i] = preprocess(images[i], 1, 0.1)
        images[i] = images[i]:view(1, -1)
        images[i]:float()
        -- print(images[i]:size())
    end

    -- print(ques:size())
    input = torch.cat({images[1], images[2], images[3], ques:float()}, 2)
    return input, target

end