require 'nngraph';
require 'torch';


function create_BC(B_model, C_model)
    local input = nn.Identity()();

    local im_size = 227*227*3
    local image1_srlz = nn.Narrow(2, 1, im_size)(input);
    local image2_srlz = nn.Narrow(2, im_size+1, im_size)(input);
    local image3_srlz = nn.Narrow(2, 2*im_size+1, im_size)(input);
    local question = nn.Narrow(2, 3*im_size+1, 42)(input);

    local image1 = nn.View(3, 227, 227)(image1_srlz)
    local image2 = nn.View(3, 227, 227)(image2_srlz)
    local image3 = nn.View(3, 227, 227)(image3_srlz)

    local confidence = B_model({question, image3});
    local scores = C_model({image1, image2, question, confidence});
    
    nngraph.annotateNodes();
    return nn.gModule({input}, {scores});
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
    local pred = 1
    if(prob[1] < 0.5) then
        pred = 0
    end
    local pred_err = 0
    if pred ~= target then
        pred_err = 1
    end
    return pred_err
end



function get_total_pred_err(probs, targets)

    local pred = torch.Tensor(probs:size())
    pred:fill(1)
    pred[torch.lt(probs:double(), 0.5)] = 0

    local pred_err = torch.ne(pred, targets)
    return torch.mean(pred_err:double())

end



-- Use a typical generic gradient update function
function accumulate(model, inputs, targets, crit, wd)
    local probs = model:forward(inputs:cuda())
    local loss = crit:forward(probs:cuda(), targets:cuda())
    local gradCriterion = crit:backward(probs:cuda(), targets:cuda())
    -- model:backward(inputs, gradCriterion, 1/((#inputs)[1]))
    model:backward(inputs:cuda(), gradCriterion:cuda())
    do_weight_decay(model, wd)
    local pred_err = get_total_pred_err(probs, targets)
    -- print('prob = ', prob)
    return loss, pred_err
end


-- function to evalaute the model
function evalPerf(model, crit, get_example, reader, iter, attrs, num_im)

    outfile = io.open('train_C.out', 'a')
    outfile:write('Testing... \n')
    outfile:close()

    -- FOR DEBUGGING only
    -- set the random seed so that same batch is chosen always. Make sure error goes down
    -- torch.manualSeed(3489208)
    inputs, targets = nextBatch(get_example, reader, iter, attrs, num_im)
    -- print(collectgarbage('count'))
    -- print(inputs:size())

    model:evaluate()
    local probs = model:forward(inputs:cuda())
    model:training()
    -- print(collectgarbage('count'))

    local test_loss = crit:forward(probs:cuda(), targets:cuda())
    local test_pred_err = get_total_pred_err(probs, targets)
    -- print(collectgarbage('count'))

    outfile = io.open('train_C.out', 'a')
    outfile:write('test_loss = ', test_loss, ', ')
    outfile:write('test_pred_err = ', test_pred_err, '\n')
    outfile:close()

end



function nextBatch(get_example, reader, batch_size, attrs, num_im)
    local inputs = torch.Tensor(batch_size, 3*227*227*3+42);
    local targets = torch.Tensor(batch_size);
    local i = 0;
    for i = 1, batch_size do
        inputs[i], targets[i] = get_example(reader, attrs, num_im)
        -- print(i, collectgarbage('count'))
    end
    inputs:cuda()
    targets:cuda()
    return inputs, targets
end