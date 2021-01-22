import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F
import copy
from multiprocessing import Pool


class ImgFeats(nn.Module):
	def __init__(self, args):
		"""Load the pretrained ResNet-101 and replace top fc layer."""
		super(ImgFeats, self).__init__()
		self.args = args
		self.finetune_cnn = getattr(args, "finetune_cnn", False)
		self.caption_model = args.caption_model
		self.image_model = args.image_model
		self.use_bert = args.use_bert
		self.use_img = args.use_img
		self.bert_size = args.bert_size
		self.features_size = 0

		self.avg_pooling1x1 = nn.AdaptiveAvgPool2d((1,1))
		# if self.caption_model !="lstm" and self.finetune_cnn == False:
		# 	self.linear_img = nn.Linear(2048, 512)

		# self.linear_bert = nn.Linear(768, 32) #!!!! open when use lstm
		### img model
		if self.use_img:
			# original_model = self.get_original_model(self.image_model)
			# # input --> layer before avg pool
			# if 'ResNet' in self.image_model:
			# 	self.base_module = nn.Sequential(*list(original_model.children())[:-2], nn.ReLU())
			# elif 'VGG' in self.image_model:
			# 	# 7*7*512
			# 	self.base_module = nn.Sequential(*list(original_model.features.children()), nn.ReLU())		

			if self.image_model in ['ResNet50', 'ResNet101', 'ResNet152']:
				self.features_size += 7*7*2048
			else:
				self.features_size += 7*7*512

			if self.caption_model == "lstm":
				self.features_size = 2048
		if self.use_bert:
			if self.caption_model == "lstm":
				self.features_size += 32*64
			else:
				self.features_size += self.bert_size  # is it 768*64
			
	# def get_original_model(self, name):
	# 	if self.image_model == "VGG16":
	# 		original_model = models.vgg16(pretrained=True) # 7*7*512
	# 	elif self.image_model == "VGG19":
	# 		original_model = models.vgg19(pretrained=True) # 7*7*512
	# 	elif self.image_model == 'ResNet18':
	# 		original_model = models.resnet18(pretrained=True) # 7*7*512
	# 	elif self.image_model == "ResNet34": 
	# 		original_model = models.resnet34(pretrained=True) # 7*7*512
	# 	elif self.image_model == "ResNet50":
	# 		original_model = models.resnet50(pretrained=True) # 7*7*2048
	# 	elif self.image_model == "ResNet101":
	# 		original_model = models.resnet101(pretrained=True) # 7*7*2048
	# 	elif self.image_model == "ResNet152":
	# 		original_model = models.resnet152(pretrained=True) # 7*7*2048
	# 	else:
	# 		raise ValueError("Invalid image model. Please choose from VGG16/19 or ResNet18/34/50/101/152")
	# 	return original_model

	def forward(self, images=None, bert_features=None):
		"""Extract feature vectors from input images."""
		if self.use_img:
			x = images
			if self.caption_model == "lstm":
				x = x.resize(x.size(0), 7, 7, -1)
				x = self.avg_pooling1x1(x).squeeze()
				print("!:", x.size())
				print(x.size())
			# [batch_size, 2048, 7,7] -> [batch_size, 2048*7*7]
			features = x.view(x.size(0), -1)
			# print(features.size())

		if self.args.use_bert:
			# bert_features = self.bert_embedding(bert_features)
			# if self.caption_model == "lstm": #####!!!!! open when use lstm
			# 	bert_features = self.linear_bert(bert_features)
			bert_features = bert_features.view(bert_features.size(0), -1)

			# print(features.size())
			# print(bert_features.size())
			if self.use_img:
				features = torch.cat([features, bert_features], dim = 1)
			else:
				features = bert_features

		features = features.view(-1, 8, self.features_size)

		if self.caption_model == "lstm":
			# we maybe need to map the dimension to the embed size of lstm
			return features
		elif self.caption_model == "convcap":
			return att, features
		elif self.caption_model == "transformer":
			return features
