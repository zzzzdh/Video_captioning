import argparse


# img size
# img feature size
# bert feature size
# trans embed size
# end token
# loss function
def get_opt():
	parser = argparse.ArgumentParser(description='PyTorch Convolutional Image Captioning Model')

	# Data settings
	parser.add_argument('--image_root', type=str, default= './data/coco/',\
			help='directory containing coco dataset train2014, val2014, & annotations')
	parser.add_argument('--bert_filename', type=str, default= "bert_filenamen.json", help='bert_filename path')
	parser.add_argument('--caption_path', type=str, default= "../../DATASET/annotation_ui2code/captions_train.json", help='caption_train/val/test.json path')
	parser.add_argument('--vocab_path', type=str, default= "data/vocab.pkl", help='vocabulary path')
	parser.add_argument('--checkpoint', type=str, help='checkpoint')
	parser.add_argument('--use_coco', type=bool, default= False, help='test code with coco data')
	
	# General setting
	parser.add_argument('--phase', type=str, default='test', help='whether to train or not')
	parser.add_argument('--image_model', type=str, default = "ResNet50", help='which model to encode img, options: VGG16/19, ResNet18/34/50/101/152')
	parser.add_argument('--caption_model', type=str, help='which model to decode img, options: convcap, lstm, transformer')
	parser.add_argument('--max_tokens', type=int, default= 15, help='max_tokens')
	parser.add_argument('--split', type=str, help='which split')
	parser.add_argument('--num_layers', type=int, default=3,\
			    help='depth of convcap network(3) or number of layers in lstm(1)')
	parser.add_argument('--embed_size', type=int , default=2048, help='dimension of word embedding vectors')
	parser.add_argument('--finetune_cnn', type=bool , default=False, help='whether to finetune ResNet')

	# ablation setting
	parser.add_argument('--use_bert', type=bool, default= False, help='whether to use bert')
	parser.add_argument('--bert_size', type=int, default=64*768,help='size for bert feature')
	parser.add_argument('--use_img', type=bool, default= False, help='whether to use img feature')
	parser.add_argument('--img_fatures_size', type=int, default=2048,help='embedding size for img')
	parser.add_argument('--frame_per_video', type=int, default=8,help='frame_per_video')
	parser.add_argument('--use_no_aoa_data', type=bool, default= False, help='whether to use no aoa bert feature')

	# Optimization settings
	parser.add_argument('--model_path', type=str, default="model_result", help='output directory to save models & results')
	parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
	parser.add_argument('--num_epochs', type=int, default=30, help='number of training epochs')
	parser.add_argument('--batch_size', type=int, default=32, help='number of images per training batch')
	parser.add_argument('--num_workers', type=int, default=4, help='pytorch data loader threads')
	parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5,\
			    help='learning rate for convcap(5e-5), for lstm(0.001)')
	parser.add_argument('-st', '--lr_step_size', type=int, default=10,\
			    help='epochs to decay learning rate after')

	# convcap
	parser.add_argument('--attention', dest='attention', action='store_true', \
			    help='Use this for convcap with attention (by default set)')
	parser.add_argument('--no-attention', dest='attention', action='store_false', \
			    help='Use this for convcap without attention')

    # lstm
	parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')

	# transformer
	#parser.add_argument('--att_size', type=int , default=7, help='attention size for transfomer')
	parser.add_argument('--use_bn', type=int, help="whether to use batch normalzation when embedding attention vector of img")
	parser.add_argument('--drop_prob_lm', type=float, help="dropout rate for language model")
	parser.add_argument('--ff_size', type=int, default = 2048, help="feed forward size for transformer")
	
	# evaluation
	parser.add_argument('--log_step', type=int , default=1000, help='step size for prining log info')
	parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')

	parser.add_argument('-sc', '--score_select', type=str, default='CIDEr',\
			    help='metric to pick best model')
	parser.add_argument('--beam_size', type=int, default=1, \
			    help='beam size to use for test') 
	parser.add_argument('--vis', type=bool, default=False,\
			    help='whether to visualize results')
	
	parser.set_defaults(attention=True)
	args = parser.parse_args()
	
	return args

