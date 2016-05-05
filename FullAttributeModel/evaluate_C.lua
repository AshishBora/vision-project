require 'cunn';
require 'cudnn';
require 'nngraph';

dofile('get_example_C.lua')
dofile('training_funcs.lua')
dofile('func_lib.lua')

train_lmdb = lmdb.env{
                Path = './lmdb/train_lmdb',
                Name = 'train_lmdb'
            }

train_lmdb:open()
stat = train_lmdb:stat()
printTable(stat) -- Current status
train_num = stat['entries']
train_reader = train_lmdb:txn(true) -- Read-only transaction
train_attrs = torch.load('./lmdb/train_attrs.t7')

B_model = torch.load('B_model.t7')
C_model = torch.load('C_model.t7')
BC_model = create_BC(B_model, C_model)
crit = nn.BCECriterion() -- define loss

BC_model:cuda()
crit:cuda()

inputs, targets = nextBatch(get_example_C, train_reader, 2, train_attrs, train_num)
probs = BC_model:forward(inputs:cuda())

print(probs)
print(targets)

gradCriterion = crit:backward(probs:cuda(), targets:cuda())
BC_model:backward(inputs:cuda(), gradCriterion:cuda())

train_reader:abort()
train_lmdb:close()