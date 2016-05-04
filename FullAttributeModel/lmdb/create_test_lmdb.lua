require 'lmdb'
require 'image'
require 'torch'

dofile('create_lmdb.lua')

project_dir = '/Users/ashish/vision-project/'
base_path = project_dir .. 'data/SUN/SUN_WS/test/'

list_file_path = './test_listfile.txt'
lmdb_path = './test_lmdb'
lmdb_name = 'test_lmdb'
attrs_path = './attrs_fake.t7'

create_lmdb(base_path, list_file_path, lmdb_path, lmdb_name, attrs_path)