require 'lmdb'
require 'image'
require 'torch'

dofile('create_lmdb.lua')

-- project_dir = '/Users/ashish/vision-project/'
project_dir = '/work/04001/ashishb/maverick/vision-project/'
base_path = project_dir .. 'data/SUN/SUN_WS/training/'
list_file_path = './val_listfile.txt'
lmdb_path = './val_lmdb'
lmdb_name = 'val_lmdb'
attrs_path = './val_attrs.t7'

create_lmdb(base_path, list_file_path, lmdb_path, lmdb_name, attrs_path)