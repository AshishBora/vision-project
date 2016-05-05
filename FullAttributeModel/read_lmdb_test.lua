require 'lmdb'
require 'image'

dofile('func_lib.lua')

local db = lmdb.env{
Path = './train_lmdb',
Name = 'train_lmdb'
}

db:open()
printTable(db:stat()) -- Current status


local reader = db:txn(true) --Read-only transaction


-------Read-------
for i = 1, 10 do
    print(i)
    img = reader:get(i)
    image.display(img)
end

reader:abort()

db:close()