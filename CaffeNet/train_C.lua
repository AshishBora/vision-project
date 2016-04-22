require 'image';
require 'loadcaffe';
require 'cudnn';
-- require 'cunn';

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
function accumulate(model, input, target, criterion, batch_size)
    local pred = model:forward(input)
    local err = criterion:forward(pred, target)
    local gradCriterion = criterion:backward(pred, target)
    model:backward(input, {gradCriterion[1]:cuda(), gradCriterion[2]:cuda()}, 1/batch_size)

    -- print('pred = ', pred[1][1], pred[2][1])
    return err
end


-- function to evalaute the model
function evalPerf(model, criterion, testset, base_path, test_iter)

    print('Testing... ')
    local test_loss = 0
    local test_pred_err = 0

    -- FOR DEBUGGING only
    -- set the random seed so that same batch is chosen always. Make sure error goes down
    torch.manualSeed(3489208)
    print('Hi 0')
    for j = 1, test_iter do
        example = getCtrainExample(testset, base_path)
        print('Hi 1')
        input = example[1]
        print('Hi 2')
        target = example[2]
        print('Hi 3')
        local pred = model:forward({input[1]:cuda(), input[2]:cuda(), input[3]:cuda(), input[4]:cuda()})
        print('Hi 4')
        local samp_loss = criterion:forward(pred, target)
        print('Hi 5')
        local pred_err = 0
        print('Hi 6')
        if samp_loss > 0 then
            pred_err = 1
        end
        print('Hi 7')
        test_pred_err = test_pred_err + pred_err
        print('Hi 8')
        test_loss = test_loss + samp_loss
        print('Hi 9')
    end
    print('average test_loss = ', test_loss/test_iter)
    print('average test_pred_err = ', test_pred_err/test_iter)

end

-- get some essential functions
print('Running string split... ')
dofile('string_split.lua')

print('Running getImPaths... ')
dofile('getImPaths.lua')

-- Laod the original model and creat BC model
print('Loading pretrained model... ')
dofile('torchModel_BC.lua')
print('done')

-- get the list of images to be used for training
print('Loading image paths and labels... ')
train_listfile_path = '/work/04001/ashishb/maverick/data/listfiles/train_listfile_100.txt'
val_listfile_path = '/work/04001/ashishb/maverick/data/listfiles/val_listfile.txt'

trainset = getImPaths(train_listfile_path)
testset = getImPaths(val_listfile_path)

base_path = '/work/04001/ashishb/maverick/data/'
print('done')

-- put everything in evaluate mode
BC_model:evaluate()
-- put C in training model
C_model:training()

crit = nn.MarginRankingCriterion(0.1)
lr = 0.01
batch_size = 50
max_train_iter = 100
test_interval = 10
test_iter = 1000
lr_stepsize = 20
gamma = 0.1
-- TO DO : Add weight decay

print('Training... ')
for i = 1, max_train_iter do
    BC_model:zeroGradParameters()
    local batch_err = 0

    -- FOR DEBUGGING only
    -- set the random seed so that same batch is chosen always. Make sure error goes down
    torch.manualSeed(0)

    for j = 1, batch_size do
        example = getCtrainExample(trainset, base_path)
        input = example[1]
        target = example[2]
        local err = accumulate(BC_model, {input[1]:cuda(), input[2]:cuda(), input[3]:cuda(), input[4]:cuda()}, target, crit, batch_size)
        batch_err = batch_err + err
        -- print('err = ', err)
    end
    BC_model:updateParameters(lr)
    print('batch_err = ', batch_err)

    if i % test_interval == 0 then
        evalPerf(BC_model, crit, testset, base_path, test_iter)
    end

    if i % lr_stepsize == 0 then
        lr = lr * gamma
    end

end