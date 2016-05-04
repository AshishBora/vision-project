require 'lmdb'
require 'image'
require 'torch'

dofile('create_lmdb.lua')

project_dir = '/Users/ashish/vision-project/'
base_path = project_dir .. 'data/SUN/SUN_WS/training/'
list_file_path = './train_listfile.txt'
lmdb_path = './train_lmdb'
lmdb_name = 'train_lmdb'
labels_path = './labels.t7'

create_lmdb(base_path, list_file_path, lmdb_path, lmdb_name, labels_path)