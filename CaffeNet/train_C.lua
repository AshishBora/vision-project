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
    -- outfile:write(im1_Path)
    -- outfile:write(im2_Path)
    -- outfile:write(y[1], y[2])
    -- outfile:write(label)
    -- outfile:write(ques[y[1]], ques[y[2]])
    
    local image_data = {}
    image_data[1] = preprocess(base_path .. im1_Path)
    image_data[2] = preprocess(base_path .. im2_Path)
    image_data[3] = image_data[label]:clone()
    local input = {image_data[1], image_data[2], image_data[3], ques}
    
    return {input, target}
end


-- Use a typical generic gradient update function
function accumulate(model, input, target, criterion, eval_criterion,  batch_size)
    local pred = model:forward(input)
    local loss = criterion:forward(pred, target)
    local gradCriterion = criterion:backward(pred, target)
    model:backward(input, {gradCriterion[1]:cuda(), gradCriterion[2]:cuda()}, 1/batch_size)
    
    local eval_loss = eval_criterion:forward(pred, target)
    local pred_err = 0
    if(eval_loss > 0) then
        pred_err = 1
    end
    -- outfile:write('pred = ', pred[1][1], pred[2][1])
    return loss, pred_err
end


-- function to evalaute the model
function evalPerf(model, criterion, testset, base_path, test_iter)

    outfile = io.open("train_C.out", "a")
    outfile:write('Testing... ')
    outfile:close()

    local test_loss = 0
    local test_pred_err = 0

    -- FOR DEBUGGING only
    -- set the random seed so that same batch is chosen always. Make sure error goes down
    -- torch.manualSeed(3489208)

    for j = 1, test_iter do
        example = getCtrainExample(testset, base_path)
        input = example[1]
        target = example[2]
        local pred = model:forward({input[1]:cuda(), input[2]:cuda(), input[3]:cuda(), input[4]:cuda()})
        local samp_loss = criterion:forward(pred, target)
        local pred_err = 0
        if samp_loss > 0 then
            pred_err = 1
        end
        test_pred_err = test_pred_err + pred_err
        test_loss = test_loss + samp_loss
    end

    outfile = io.open("train_C.out", "a")
    outfile:write('average test_loss = ', test_loss/test_iter, ', ')
    outfile:write('average test_pred_err = ', test_pred_err/test_iter, '\n')
    outfile:close()
end

-- get some essential functions
outfile = io.open("train_C.out", "w")
outfile:write('Running string split... ')
dofile('string_split.lua')
outfile:write('done\n')
outfile:close()

outfile = io.open("train_C.out", "a")
outfile:write('Running getImPaths... ')
dofile('getImPaths.lua')
outfile:write('done\n')
outfile:close()

-- Laod the original model and creat BC model
outfile = io.open("train_C.out", "a")
outfile:write('Loading pretrained model... ')
dofile('torchModel_BC.lua')
outfile:write('done\n')
outfile:close()

-- get the list of images to be used for training
outfile = io.open("train_C.out", "a")
outfile:write('Loading image paths and labels... ')
train_listfile_path = '/work/04001/ashishb/maverick/data/listfiles/train_listfile_100.txt'
val_listfile_path = '/work/04001/ashishb/maverick/data/listfiles/val_listfile.txt'

trainset = getImPaths(train_listfile_path)
testset = getImPaths(val_listfile_path)

base_path = '/work/04001/ashishb/maverick/data/'
outfile:write('done\n')
outfile:close()

-- put everything in evaluate mode
BC_model:evaluate()
-- put C in training mode
C_model:training()

crit = nn.MarginRankingCriterion(0.1)
eval_crit = nn.MarginRankingCriterion(0.0)
lr = 0.01
batch_size = 250
max_train_iter = 100
test_interval = 20
test_iter = 1000
lr_stepsize = 50
gamma = 0.1
snapshot_interval = 50
snapshot_prefix = './'
-- TO DO : Add weight decay

outfile = io.open("train_C.out", "a")
outfile:write('Training... \n')
outfile:close()

for i = 1, max_train_iter do

    -- initial testing
    if i == 1 then
        evalPerf(BC_model, eval_crit, testset, base_path, test_iter)
    end

    BC_model:zeroGradParameters()
    local batch_loss = 0

    -- FOR DEBUGGING only
    -- set the random seed so that same batch is chosen always. Make sure error goes down
    -- torch.manualSeed(0)

    local train_pred_err = 0
    for j = 1, batch_size do
        example = getCtrainExample(trainset, base_path)
        input = example[1]
        target = example[2]
	local loss = 0
	local pred_err = 0
        loss, pred_err = accumulate(BC_model, {input[1]:cuda(), input[2]:cuda(), input[3]:cuda(), input[4]:cuda()}, target, crit, eval_crit, batch_size)
        batch_loss = batch_loss + loss
	train_pred_err = train_pred_err + pred_err;
        -- outfile:write('loss = ', loss)
    end
    C_model:updateParameters(lr)

    outfile = io.open("train_C.out", "a")
    outfile:write('Iteration no. ', i, ', lr = ', lr, ', average batch_loss = ', batch_loss/batch_size, ', Training Error = ', train_pred_err/batch_size, '\n')
    outfile:close()


    if i % test_interval == 0 then
        evalPerf(BC_model, eval_crit, testset, base_path, test_iter)        
    end

    if i % lr_stepsize == 0 then
        lr = lr * gamma
    end

    if i % snapshot_interval == 0 then
        snapshot_filename = snapshot_prefix .. 'BC_model__' .. tostring(i) .. '.t7'
        torch.save(snapshot_filename, BC_model)
    end

end
