dofile('get_example_C.lua')

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

input, target = get_example_C(train_reader, train_attrs, train_num)
print(input:size())
print(target)

train_reader:abort()
train_lmdb:close()