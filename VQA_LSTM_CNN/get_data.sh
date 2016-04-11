wget https://filebox.ece.vt.edu/~jiasenlu/codeRelease/vqaRelease/train_only/data_train_val.zip

wget https://filebox.ece.vt.edu/~jiasenlu/codeRelease/vqaRelease/train_only/pretrained_lstm_train_val.t7.zip

unzip data_train_val.zip 
unzip pretrained_lstm_train_val.t7.zip

rm data_train_val.zip 
rm pretrained_lstm_train_val.t7.zip

th eval.lua -input_img_h5 data_img.h5 -input_ques_h5 data_prepro.h5 -input_json data_prepro.json -model_path lstm.t7
