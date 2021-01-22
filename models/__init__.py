import os

import numpy as np
import torch

from .lstm import DecoderRNN
from .convcap import convcap
from .Transformer import Transformer

def setup(args):
	#lstm
	if args.caption_model == 'lstm':
		args.input_features_size = 0
		# hard code
		if args.use_img:
			args.input_features_size += 2048 * args.frame_per_video 
		if args.use_bert:
			args.input_features_size += 64*128*args.frame_per_video 
		model = DecoderRNN(args.input_features_size, args.embed_size, args.hidden_size, args.vocab_len, args.num_layers, args.max_tokens)
	# convolutional caption
	elif args.caption_model == 'convcap':
		model = convcap(args.numwords, args.embed_size, args.num_layers, is_attention=args.attention)
	# Transformer
	elif args.caption_model == 'transformer':
		model = Transformer(args)
	else:
		raise Exception("Caption model not supported: {}".format(args.caption_model))

	'''
	# check compatibility if training is continued from previously saved model
	if getattr(args, "checkpoint", None) is not None:
		# check if all necessary files exist 
		assert os.path.isdir(args.start_from)," %s must be a a path" % args.start_from
		assert os.path.isfile(os.path.join(args.start_from,"infos_"+args.id+".pkl")),"infos.pkl file does not exist in path %s"%args.start_from
		model.load_state_dict(torch.load(os.path.join(args.model_path, 'best_model.ckpt')))
	'''
	return model
