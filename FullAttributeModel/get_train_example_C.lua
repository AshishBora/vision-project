-- require 'cudnn'
require 'torch'
require 'nngraph'
require 'nn'
-- require 'cunn'
require 'lmdb'
require 'image'

dofile('func_lib.lua')
dofile('preproc.lua')

-- function to get an example for training C
function get_train_example_C(reader, attrs, num_im)

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

    -- target is probability of label = 2
    local target = label - 1

    -- serialize
    for i = 1, 3 do
    	images[i] = images[i]:view(-1)
    	images[i]:float()
    	print(images[i]:size())
    end

    print(ques:size())

    input = torch.cat({images[1], images[2], images[3], ques:float()})

    return input, target
end


train_lmdb = lmdb.env{
Path = './lmdb/train_lmdb',
Name = 'train_lmdb'
}

train_lmdb:open()
stat = train_lmdb:stat()
-- printTable(stat) -- Current status
num_train = stat['entries']
train_reader = train_lmdb:txn(true) --Read-only transaction
attrs = torch.load('./lmdb/attrs.t7')

input, target = get_train_example_C(train_reader, attrs, num_train)
print(input:size())
print(target)

train_reader:abort()
train_lmdb:close()

