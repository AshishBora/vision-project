require 'lmdb'
require 'image'
require 'torch'

function create_lmdb(base_path, list_file_path, lmdb_path, lmdb_name, attrs_path)

    attrs = {}

    db = lmdb.env{
        Path = lmdb_path,
        Name = lmdb_name
    }

    db:open()
    txn = db:txn() -- Write transaction

    i = 1
    for line in io.lines(list_file_path) do
        words = line:split(' ')
        im_Path, attr = words[1], words[2]
        attrs[i] = tonumber(attr)

        path = base_path .. im_Path
        img = image.load(path, 3, 'float')
        txn:put(i, img)
        i = i + 1
        if i % 100 == 0 then
        	print('Processed', i, 'images')
            txn:commit()
            txn = db:txn() -- Write transaction
        end
    end

    txn:commit()
    db:close()

    attrs = torch.Tensor(attrs)
    torch.save(attrs_path, attrs)
end
