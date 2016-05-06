require 'image';
require 'cunn';
require 'cudnn';
require 'nngraph';

----------------- get training functions --------------
dofile('training_funcs.lua')
dofile('get_example_C.lua')


------------- Load the LMDBs and get the reader ----------------
train_lmdb = lmdb.env{
                Path = './lmdb/train_lmdb',
                Name = 'train_lmdb'
            }

train_lmdb:open()
train_stat = train_lmdb:stat()
-- printTable(train_stat) -- Current status
train_num = train_stat['entries']
train_reader = train_lmdb:txn(true) -- Read-only transaction
train_attrs = torch.load('./lmdb/train_attrs.t7')


val_lmdb = lmdb.env{
                Path = './lmdb/val_lmdb',
                Name = 'val_lmdb'
            }

val_lmdb:open()
val_stat = val_lmdb:stat()
-- printTable(val_stat) -- Current status
val_num = val_stat['entries']
val_reader = val_lmdb:txn(true) -- Read-only transaction
val_attrs = torch.load('./lmdb/val_attrs.t7')



-------------------------  Create model --------------------------
outfile = io.open("train_C.out", "w")
outfile:write('Creating model... ')

B_model = torch.load('B_model.t7')
C_model = torch.load('C_model.t7')
BC_model = create_BC(B_model, C_model)
--graph.dot(BC_model.fg, 'BCM')
crit = nn.BCECriterion() -- define loss

BC_model:double() -- convert to double
BC_model:evaluate() -- put the model in evalaute mode
BC_model:cuda() -- put the model on cuda
crit:cuda()

outfile:write('done\n')
outfile:close()


-------------------- easy access to layers --------------------

-- verify this
B_conv = B_model.modules[2]
B_fc = B_model.modules[3]

C_conv1 = C_model.modules[2]
C_fc1 = C_model.modules[3]
C_conv2 = C_model.modules[8]
C_fc2 = C_model.modules[9]

C_cmpr = C_model.modules[15]


-------------------- share parameters --------------------
-- print(collectgarbage("count"))

C_conv1 = B_conv:clone('weight', 'bias')
C_conv2 = B_conv:clone('weight', 'bias')

-- print(collectgarbage("count"))
BC_model:clearState()
-- print(collectgarbage("count"))

---------------- Define hyper parameters ------------------------
lr = 0.2
attr_lr = 0.5

-- batch_size = 512
batch_size = 128

max_train_iter = 10000
test_interval = 50

-- val_iter = 1000
val_iter = 128

lr_stepsize = 200
gamma = 0.7
attr_gamma = 0.7
wd = 0
snapshot_interval = 100
snapshot_prefix = './'
snapshot = true

-- verify this
B_fc.modules[2].p = 0.2 -- dropout rate
C_fc1.modules[2].p = 0.2  -- dropout rate
C_fc2.modules[2].p = 0.2  -- dropout rate

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


for i = 1, max_train_iter do
    
    mem = collectgarbage("count")

    if i == 1 then  -- initial testing
        evalPerf(BC_model, crit, get_example_C, val_reader, val_iter, val_attrs, val_num)
    end
    BC_model:zeroGradParameters()
    local batch_loss = 0
    -- FOR DEBUGGING only
    -- set the random seed so that same batch is chosen always. Make sure error goes down
    -- torch.manualSeed(214325)
    inputs, targets = nextBatch(get_example_C, train_reader, batch_size, train_attrs, train_num)

    -- outfile = io.open("train_C.out", "a")
    -- outfile:write(inputs:size())
    -- outfile:close()

    batch_loss, train_pred_err = accumulate(BC_model, inputs, targets, crit,  wd)
    local grad_norm = torch.norm(C_fc1.modules[3].gradWeight)

    -- update parameters for only a few layers in C
    C_fc1:updateParameters(attr_lr)
    C_fc2:updateParameters(attr_lr)
    C_cmpr:updateParameters(lr)

    BC_model:clearState(); -- reduce memory usage

    outfile = io.open("train_C.out", "a")
    outfile:write('iter ', i, ', lr: ', lr, ', attr_lr: ', attr_lr)
    outfile:write(', batch_loss: ', batch_loss, ', train_err: ', train_pred_err)
    outfile:write(', grad_norm: ', grad_norm, ', mem:', mem, '\n')
    outfile:close()

    if i % test_interval == 0 then
        evalPerf(BC_model, crit, get_example_C, val_reader, val_iter, val_attrs, val_num)
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

-- close the LMDBs
train_reader:abort()
val_reader:abort()

train_lmdb:close()
val_lmdb:close()
