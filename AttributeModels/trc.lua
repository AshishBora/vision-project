require 'nngraph';
require 'torch';


function createFullModel(B_model, C_model)
    local input = nn.Identity()();

    local image_feat1 = nn.Narrow(2, 1, 4096)(input);
    local image_feat2 = nn.Narrow(2, 4097, 4096)(input);
    local image_feat3 = nn.Narrow(2, 8193, 4096)(input);
    local question = nn.Narrow(2, 12289, 42)(input);

    local confidence = B_model({question, image_feat3});
    local scores = C_model({image_feat1, image_feat2, question, confidence});
    
    nngraph.annotateNodes();
    return nn.gModule({input}, {scores});
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

    -- randomly select one of the images to ask about its class
    local ques = torch.Tensor(42):fill(0)
    ques[labels[y[torch.bernoulli() + 1]]] = 1

    -- target is probability of label = 2
    local target = label - 1
    local input = torch.cat({im_feat[1], im_feat[2], im_feat[3], ques})

return {input, target}
end



function do_weight_decay(model, wd)
    lin_modules = model:findModules('nn.Linear');
    local i = 0;
    for i = 1,#lin_modules do
        m = torch.mul(lin_modules[i].weight, wd);
        lin_modules[i].gradWeight = lin_modules[i].gradWeight + m;
    end
end



function get_pred_err(prob, target)

    local pred = 2
    if(prob[1] < 0.5) then
        pred = 1
    end

    local pred_err = 0
    if pred ~= target+1 then
        pred_err = 1
    end

    return pred_err
end



function get_total_pred_err(probs, targets)
    local total_pred_err = 0;
    local i = 0;
    for i = 1, (#probs)[1] do
        total_pred_err = total_pred_err + get_pred_err(probs[i], targets[i]);
    end
    return total_pred_err/(#probs)[1];
end



-- Use a typical generic gradient update function
function accumulate(model, inputs, targets, criterion, eval_criterion,  wd)
    local probs = model:forward(inputs)
    local loss = criterion:forward(probs, targets)
    local gradCriterion = criterion:backward(probs, targets)
    -- model:backward(inputs, gradCriterion, 1/((#inputs)[1]))
    model:backward(inputs, gradCriterion)
    do_weight_decay(model, wd)
    local pred_err = get_total_pred_err(probs, targets)
    -- print('prob = ', prob)
    return loss, pred_err
end


-- function to evalaute the model
function evalPerf(model, criterion, set, labels, iter)

    outfile = io.open("train_C.out", "a")
    outfile:write('Testing... \n')


    -- FOR DEBUGGING only
    -- set the random seed so that same batch is chosen always. Make sure error goes down
    -- torch.manualSeed(3489208)
    inputs, targets = nextBatch(set, labels, iter);

    local probs = model:forward(inputs)
    local test_loss = criterion:forward(probs, targets)
    local test_pred_err = get_total_pred_err(probs, targets)

    outfile:write('average test_loss = ', test_loss, ', ')
    outfile:write('average test_pred_err = ', test_pred_err, '\n')
    outfile:close()

end



function nextBatch(trainset, train_labels, batchSize)
    local inputs = torch.Tensor(batchSize, 12330);
    local targets = torch.Tensor(batchSize);
    local i = 0;
    for i = 1, batchSize do
        example = getCtrainExample(trainset, train_labels);
        inputs[i] = example[1];
        targets[i] = example[2];
    end
    inputs:double();
    targets:double();
    return inputs, targets
end



-------------------------  Create model --------------------------
outfile = io.open("train_C.out", "w")
outfile:write('Creating model... ')

B_model = torch.load('B_model_nn.t7')

C_model_old = torch.load('C_model__1500_init.t7')
C_model = torch.load('C_model.t7')
C_model.modules[13].modules[1].weight = C_model_old.modules[10].modules[1].weight:clone()
C_model.modules[13].modules[3].weight = C_model_old.modules[10].modules[3].weight:clone()
C_model.modules[13].modules[1].bias = C_model_old.modules[10].modules[1].bias:clone()
C_model.modules[13].modules[3].bias = C_model_old.modules[10].modules[3].bias:clone()

BC_model = createFullModel(B_model, C_model)
--graph.dot(BC_model.fg, 'BCM')
crit = nn.BCECriterion() -- define loss
C_model_old = nil -- garbage collection

BC_model:double() -- convert to double
BC_model:evaluate() -- put the model in evalaute mode

outfile:write('done\n')
outfile:close()



---------------- Read preprocessed feature vectors and labels ------------------------
outfile = io.open("train_C.out", "a")
outfile:write('Preparing data... ')

feat_vecs = torch.load('feat_vecs.t7')
labels = torch.load('labels.t7')

-- randomly shuffle data
feat_vecs_temp = feat_vecs:clone()
labels_temp = labels:clone()
y = torch.randperm((#feat_vecs)[1])
for i = 1, ((#feat_vecs)[1]) do
    feat_vecs[i] = feat_vecs_temp[y[i]]
    labels[i] = labels_temp[y[i]]
end

-- garbage collection
feat_vecs_temp = nil
labels_temp = nil

-- generate trainset and testset
train_perc = 0.80 -- percentage of images to be taken in the train set
trainset_size = torch.round((#feat_vecs)[1] * train_perc)
trainset = feat_vecs[{{1, trainset_size}}]
train_labels = labels[{{1, trainset_size}}]
testset = feat_vecs[{{trainset_size+1, (#feat_vecs)[1]}}]
test_labels = labels[{{trainset_size+1, (#feat_vecs)[1]}}]

outfile:write('done\n')
outfile:close()



---------------- Define hyper parameters ------------------------
lr = 0.2
attr_lr = 0.5
batch_size = 512
max_train_iter = 10000
test_interval = 50
test_iter = 1000
lr_stepsize = 200
gamma = 0.7
attr_gamma = 0.7
wd = 0
snapshot_interval = 100
snapshot_prefix = './'
snapshot = true
B_model.modules[2].modules[2].p = 0.2 -- dropout rate
C_model.modules[2].modules[2].p = 0.2  -- dropout rate
C_model.modules[7].modules[2].p = 0.2  -- dropout rate


----------------  Start training ------------------------

BC_model:training() -- put the model in training mode

outfile = io.open('train_C.out', 'a')
outfile:write('Training with snapshotting ')
if snapshot then
    outfile:write('enabled... \n')
else
    outfile:write('disabled... \n')
end
outfile:close()

-- random initialization of C's fully connected layer
C_model.modules[2]:reset(0.01);
C_model.modules[7]:reset(0.01);
-- divide weights to get in trainable area
div_fact = 0.3
C_model.modules[13].modules[1].weight:mul(div_fact)
C_model.modules[13].modules[3].weight:mul(div_fact)
C_model.modules[13].modules[1].bias:mul(div_fact)
C_model.modules[13].modules[3].bias:mul(div_fact^2)

for i = 1, max_train_iter do

    collectgarbage()

    -- initial testing
    if i == 1 then
        evalPerf(BC_model, crit, testset, test_labels, test_iter)
    end

    BC_model:zeroGradParameters()
    local batch_loss = 0

    -- FOR DEBUGGING only
    -- set the random seed so that same batch is chosen always. Make sure error goes down
    -- torch.manualSeed(214325)

    inputs, targets = nextBatch(trainset, train_labels, batch_size);
    batch_loss, train_pred_err = accumulate(BC_model, inputs, targets, crit, crit,  wd);

    -- update parameters for only a few layers in C
    C_model.modules[2]:updateParameters(attr_lr)
    C_model.modules[7]:updateParameters(attr_lr)
    C_model.modules[13]:updateParameters(lr)

    BC_model:clearState(); -- reduce memory usage

    local grad_norm = torch.norm(C_model.modules[2].modules[3].gradWeight)

    outfile = io.open("train_C.out", "a")
    outfile:write('iter ', i, ', lr: ', lr, ', attr_lr: ', attr_lr)
    outfile:write(', batch_loss: ', batch_loss, ', train_err: ', train_pred_err)
    outfile:write(', grad_norm: ', grad_norm, '\n')
    outfile:close()

    if i % test_interval == 0 then
        evalPerf(BC_model, crit, testset, test_labels, test_iter)        
    end

    if i % lr_stepsize == 0 then
        lr = lr * gamma;
        attr_lr = attr_lr * attr_gamma;
    end

    if snapshot and (i % snapshot_interval == 0) then

        outfile = io.open("train_C.out", "a")

        outfile:write('Snapshotting C_model... ')
        snapshot_filename_C = snapshot_prefix .. 'C_model__' .. tostring(i) .. '.t7'
        C_model:clearState()
        torch.save(snapshot_filename_C, C_model)
        outfile:write('done\n')

        outfile:close()

    end

end
