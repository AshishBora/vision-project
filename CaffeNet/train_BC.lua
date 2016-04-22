require 'cudnn';
require 'cunn';
require 'image';
require 'loadcaffe';

-- function to get an example for training C
function getCtrainExample(trainset, base_path)
    -- randomly select two images from different classes
    local y = torch.randperm(#trainset)
    local im1_Path = trainset[y[1]][torch.random(1, #trainset[y[1]])]
    local im2_Path = trainset[y[2]][torch.random(1, #trainset[y[2]])]

    -- randomly select one of those to be given to model B
    local label = torch.bernoulli() + 1

    -- randomly select one of the images to ask about its class
    local ques = torch.Tensor(1000):fill(0)
    ques[y[torch.bernoulli() + 1]] = 1

    -- target is just remapping label
    -- label = 1  =>  target = -1 
    -- label = 2  =>  target = 1
    -- This is necessary for MaxMarginCriterion
    local target = 2*label-3

    -- Testing
    -- print(im1_Path)
    -- print(im2_Path)
    -- print(y[1], y[2])
    -- print(label)
    -- print(ques[y[1]], ques[y[2]])
    
    local image_data = {}
    image_data[1] = preprocess(base_path .. im1_Path)
    image_data[2] = preprocess(base_path .. im2_Path)
    image_data[3] = image_data[label]:clone()
    local input = {image_data[1], image_data[2], image_data[3], ques}
    
    return {input, target}
end

-- Use a typical generic gradient update function
function trainStep(model, input, target, criterion, learningRate)
    local pred = model:forward(input)
    local err = criterion:forward(pred, target)
    local gradCriterion = criterion:backward(pred, target)

    model:zeroGradParameters()
    model:backward(input, {gradCriterion[1]:cuda(), gradCriterion[2]:cuda()})
    model:updateParameters(learningRate)

    print('pred = ', pred[1][1], pred[2][1])
    print('err = ', err)
end

-- get some essential functions
dofile('string_split.lua')
dofile('getImPaths.lua')

-- Laod the original model and creat BC model
dofile('torchModel_BC.lua')

-- get the list of images to be used for training
list_file_path = '/work/04001/ashishb/maverick/data/listfiles/train_listfile_100.txt'
base_path = '/work/04001/ashishb/maverick/data/'
trainset = getImPaths(list_file_path)

crit = nn.MarginRankingCriterion(0.1)
lr = 0.01
for i = 1, 5 do
    example = getCtrainExample(trainset, base_path)
    input = example[1]
    target = example[2]
    trainStep(BC_model, {input[1]:cuda(), input[2]:cuda(), input[3]:cuda(), input[4]:cuda()}, target, crit, lr)
end



