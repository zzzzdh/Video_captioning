python3 main.py \
--image_root ../data/Jieshan/Dataset_small_addvaltest \
--caption_path ../data/Jieshan/Dataset_small_addvaltest/annotation \
--vocab_path ./data/vocab_dailyreport_small.pkl \
--caption_model lstm \
--max_tokens 15 \
--model_path ../data/Jieshan/Run_OneList_addvaltest_moreepoch \
--num_epochs 120 \
--batch_size 24 \
--save_step 2 \
-lr 0.005 \
--image_model ResNet50 \
--drop_prob_lm 0.1 \
--ff_size 2048 \
--num_layers 3 \
--use_bert 1 \
--use_img 1 \
--phase train \
--num_workers 2 \
--vis 1 


--checkpoint /home/cheer/Project/DailyReport/data/Jieshan/Run_OneList_addvaltest/ResNet50/lstm/model_img_bert/best_model.ckpt


#whether to finetune_cnn
--finetune_cnn 1 \
--use_img 1 \

/home/cheer/Project/DailyReport/data/Jieshan/Dataset_OnePlaylist
--bert_filename ../data/Jieshan/Dataset_OnePlaylist/binary_features 


small dataset
5506/764/??
--use_img 1 \ 

--use_no_aoa_data 1 \
--use_img 1 \

6254/784/780
