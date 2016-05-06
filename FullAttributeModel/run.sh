#luajit caffe2torch.lua
#luajit print_sun_ws.lua
#luajit evaluate_sun_ws.lua

#cd lmdb;
#luajit create_train_lmdb.lua
#luajit create_val_lmdb.lua
#luajit create_test_lmdb.lua

#luajit create_B.lua
#luajit print_B.lua
#luajit evaluate_B.lua

#luajit create_C.lua
#luajit print_C.lua
#luajit evaluate_C.lua

luajit trc.lua
