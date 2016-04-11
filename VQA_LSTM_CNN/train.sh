# Run train_val on torch
th eval.lua -input_img_h5 data_img.h5 -input_ques_h5 data_prepro.h5 -input_json data_prepro.json -model_path lstm.t7
