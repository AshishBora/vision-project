require 'nngraph';


function createFullModel(B_model, C_model)
    local image_feat1 = nn.Identity()();
    local image_feat2 = nn.Identity()();
    local image_feat3 = nn.Identity()();
    local question = nn.Identity()();

    local confidence = B_model({question, image_feat3});
    local scores = C_model({image_feat1, image_feat2, question, confidence});
    
    nngraph.annotateNodes();
    return nn.gModule({image_feat1, image_feat2, image_feat3, question}, {scores});
end

-- function to get an example for training C
function getCtrainExample(set, labels)

    -- randomly select two images from different classes
    local y = torch.randperm((#set)[1])

    local im_feat = {}
    im_feat[1] = set[y[1]]
    im_feat[2] = set[y[2]]

    -- randomly select one of those to be given to model B
    local label = torch.bernoulli() + 1
    im_feat[3] = im_feat[label]:clone()

    local ques = torch.Tensor(42):fill(0)
    -- randomly select one of the images to ask about its class
    ques_label = torch.bernoulli() + 1
    ques[labels[y[ques_label]]] = 1

    -- target is just remapping label
    -- label = 1  =>  target = 1 
    -- label = 2  =>  target = -1
    -- This is necessary for MaxMarginCriterion
    local target = 3 - 2*label

    -- Testing
    -- outfile:write(im1_Path)
    -- outfile:write(im2_Path)
    -- outfile:write(y[1], y[2])
    -- outfile:write(label)
    -- outfile:write(ques[y[1]], ques[y[2]])
    
    local input = {im_feat[1], im_feat[2], im_feat[3], ques, ques_label}
    
    return {input, target}
end


-- Use a typical generic gradient update function
function accumulate(model, input, target, criterion, eval_criterion, batch_size)
    local pred = model:forward(input)
    local loss = criterion:forward(pred, torch.Tensor{target})
    local gradCriterion = criterion:backward(pred, torch.Tensor{target})
    model:backward(input, gradCriterion, 1/batch_size)
    
    local eval_loss = eval_criterion:forward(pred, torch.Tensor{target})
    local pred_err = 0
    if(eval_loss > 0) then
        pred_err = 1
    end
    -- outfile:write('pred = ', pred[1][1], pred[2][1])
    return loss, pred_err
end


-- function to evalaute the model
function evalPerf(model, criterion, set, labels, iter)

    outfile = io.open("train_C.out", "a")
    outfile:write('Testing... ')
    outfile:close()

    local test_loss = 0
    local test_pred_err = 0

    -- FOR DEBUGGING only
    -- set the random seed so that same batch is chosen always. Make sure error goes down
    -- torch.manualSeed(3489208)

    for j = 1, iter do
        example = getCtrainExample(set, labels)
        input = example[1]
        target = example[2]
        local pred = model:forward({input[1], input[2], input[3], input[4]})
        local samp_loss = criterion:forward(pred, torch.Tensor{target})
        local pred_err = 0
        if samp_loss > 0 then
            pred_err = 1
        end
        test_pred_err = test_pred_err + pred_err
        test_loss = test_loss + samp_loss
    end

    outfile = io.open("train_C.out", "a")
    outfile:write('average test_loss = ', test_loss/iter, ', ')
    outfile:write('average test_pred_err = ', test_pred_err/iter, '\n')
    outfile:close()
end

-- get some essential functions
-- outfile = io.open("train_C.out", "w")
-- outfile:write('Running string split... ')
-- dofile('string_split.lua')
-- outfile:write('done\n')
-- outfile:close()

-- outfile = io.open("train_C.out", "a")
-- outfile:write('Running getImPaths... ')
-- dofile('getImPaths.lua')
-- outfile:write('done\n')
-- outfile:close()

-- Laod the original model and creat BC model
outfile = io.open("train_C.out", "w")
outfile:write('Loading pretrained model... ')

B_model = torch.load('B_model_nn.t7')
C_model = torch.load('C_model_debug.t7')
BC_model = createFullModel(B_model, C_model)
-- ABC_model = createFullModel(A_model, B_model, C_model, encoders);

-- convert to double
BC_model:double()

outfile:write('done\n')
outfile:close()

-- read preprocessed feature vectors and labels
feat_vecs = torch.load('feat_vecs.t7')
labels = torch.load('labels.t7')

num_im  = 10
feat_vecs = feat_vecs[{{1, num_im}}]
labels = labels[{{1, num_im}}]
-- TO DO : random shuffle of data
-- y = torch.randperm(#feat_vecs)

-- generate trainset and testset
-- percentage of images in the train set
train_perc = 0.80
trainset_size = torch.round((#feat_vecs)[1] * train_perc)
trainset = feat_vecs[{{1, trainset_size}}]
train_labels = labels[{{1, trainset_size}}]
testset = feat_vecs[{{trainset_size+1, (#feat_vecs)[1]}}]
test_labels = labels[{{trainset_size+1, (#feat_vecs)[1]}}]

-- put the model in evalaute mode except for C
BC_model:evaluate()
C_model:training()

crit = nn.MarginCriterion(0.1)
eval_crit = nn.MarginCriterion(0.0)
lr = 0.0001
batch_size = 256
max_train_iter = 5000
test_interval = 20
test_iter = 1000
lr_stepsize = 100
gamma = 1
snapshot_interval = 100
snapshot_prefix = './'
-- TO DO : Add weight decay

outfile = io.open("train_C.out", "a")
outfile:write('Training... \n')
outfile:close()

for i = 1, max_train_iter do

    -- initial testing
    if i == 1 then
        evalPerf(BC_model, eval_crit, testset, test_labels, test_iter)
    end

    BC_model:zeroGradParameters()
    local batch_loss = 0

    -- FOR DEBUGGING only
    -- set the random seed so that same batch is chosen always. Make sure error goes down
    -- torch.manualSeed(0)

    local train_pred_err = 0
    for j = 1, batch_size do
        example = getCtrainExample(trainset, train_labels)
        input = example[1]
        target = example[2]
        
        label = (3 - target)/2
        ques_label = input[5]
        if label == ques_label then
            confidence = torch.Tensor{1}
        else
            confidence = torch.Tensor{0}
        end
        
        local loss = 0
        local pred_err = 0
        -- FOR DEBUG ONLY : GIVING SAME INPUT        
        loss, pred_err = accumulate(C_model, {input[1], input[2], input[4], confidence}, target, crit, eval_crit, batch_size)        
        batch_loss = batch_loss + loss
        train_pred_err = train_pred_err + pred_err;
        -- outfile:write('loss = ', loss)
    end

    -- update parameters only for C
    C_model.modules[10]:updateParameters(lr)

    outfile = io.open("train_C.out", "a")
    outfile:write('Iteration no. ', i, ', lr = ', lr, ', average batch_loss = ', batch_loss/batch_size, ', Training Error = ', train_pred_err/batch_size, '\n')
    outfile:close()


    if i % test_interval == 0 then
        evalPerf(BC_model, eval_crit, testset, test_labels, test_iter)        
    end

    if i % lr_stepsize == 0 then
        lr = lr * gamma
    end

    if i % snapshot_interval == 0 then
        snapshot_filename = snapshot_prefix .. 'BC_model__' .. tostring(i) .. '.t7'
        torch.save(snapshot_filename, BC_model)
    end

end