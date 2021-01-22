import os, nltk, json
import numpy as np
import pickle as pk


from PIL import Image
from dataloaders.build_vocab import Vocabulary
from pycocotools.coco import COCO

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
from torchvision import transforms
import random
# from gensim.models import FastText as ft

# from imgaug import augmenters as iaa
# import imgaug as ia

from torch.utils.data import DataLoader
# from prefetch_generator import BackgroundGenerator

# class DataLoaderX(DataLoader):
# 	def __iter__(self):
# 		return BackgroundGenerator(super().__iter__)

# class ImgAugTransform:
#   def __init__(self):
#     self.aug = iaa.Sequential([
#         iaa.Scale((224, 224)),
#         iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
#         iaa.Affine(rotate=(-20, 20), mode='constant'),
#         iaa.Sometimes(0.25,
#                       iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
#                                  iaa.CoarseDropout(0.1, size_percent=0.5),
# 				 iaa.Sharpen(alpha=0.5)])),
#         iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True),
# 	iaa.Sometimes(0.15,iaa.WithChannels(0, iaa.Add((10, 100)))),
# 	iaa.Sometimes(0.05,iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"))
#     ])
      
#   def __call__(self, img):
#     img = np.array(img)
#     return self.aug.augment_image(img)


# self.img_transforms = transforms.Compose([
# 		        ImgAugTransform(),
# 		        lambda x: Image.fromarray(x),
# 			transforms.ToTensor(),
# 			transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], 
# 					     std  = [ 0.229, 0.224, 0.225 ])
# 			])


class AccessibilityDataset(Dataset):
	"""Loads train/val/test splits of our dataset"""

	def __init__(self, args, vocab, split, img_size = (224,224)):
		self.max_tokens = args.max_tokens
		self.root = args.image_root
		self.frame_per_video = args.frame_per_video
		flag_val = False

		# if split in ["validation", "test"]:
		# 	flag_val = True
		# 	split = "train"
		# gt captioning
		caption_json = os.path.join(args.caption_path, "{}_list.json".format(split))
		self.caption_json = caption_json
		self.data = COCO(self.caption_json)
		self.ids = list(self.data.anns.keys())
		self.bert_filename = args.bert_filename
		self.bert_size = args.bert_size
		self.use_no_aoa_data = args.use_no_aoa_data


		self.ids.sort()
		#self.ids = self.ids[:6000] # test code

		# if flag_val:
		# 	self.ids = self.ids[:1000]

		# self.image_features = {}
		imgf_filename = os.path.join(self.root, "{}_Images_resnet50_features".format(split))
		bertf_filename = os.path.join(self.root, "{}_binary_features".format(split))
		if args.use_no_aoa_data:
			bertf_filename += "_no_aoa"
		with open(imgf_filename +"_OnePlayList_addvaltest.pkl", "rb") as f:
			self.image_features = pk.load(f)
		with open(bertf_filename+"_OnePlayList_addvaltest.pkl", "rb") as f:
			self.bert_features = pk.load(f)


		# whether to use cnn feature
		self.use_img = args.use_img
		# whether to use bert feature
		self.use_bert = args.use_bert

		if split == "train":
			random.shuffle(self.ids)
		
		# <pad>:0, <start>:1, <end>:2, <unk>:3
		self.vocab = vocab
		self.split = split
		self.img_seq_len = args.frame_per_video
		self.numwords = len(self.vocab)
		print(('[DEBUG] #words in wordlist: %d' % (self.numwords)))

		# Image preprocessing, normalization for the pretrained resnet
		self.img_size = img_size
		self.img_transforms = transforms.Compose([
			transforms.Resize((224,224)),
			# transforms.RandomCrop((224,224)),
			# transforms.ColorJitter(hue=.05, saturation=.05),
			#transforms.RandomHorizontalFlip(), 
			transforms.ToTensor(),
			transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], 
					     std  = [ 0.229, 0.224, 0.225 ])
		])

		self.img_transforms_test = transforms.Compose([
					transforms.Resize(img_size),
					transforms.ToTensor(),
					transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], 
							     std  = [ 0.229, 0.224, 0.225 ])
					])

	def subsequent_mask(self, size):
	    "Mask out subsequent positions."
	    attn_shape = (size, size)
	    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
	    return torch.from_numpy(subsequent_mask) == 0

	def __getitem__(self, idx):
		data = self.data
		vocab = self.vocab

		ann_id =self.ids[idx] # idx
		first_img_id = data.anns[ann_id]['image_id']

		imgs = []
		img_filenames = []
		for i in range(self.frame_per_video):
			img_id = first_img_id + i
			path = data.loadImgs(img_id)[0]['file_name']
			# path = os.path.join(self.root, path)
			img_filenames.append(path)

		if self.use_img:
			for path in img_filenames:
				# # preprocess img
				# img = Image.open(path).convert('RGB')
				# if self.split == "train":
				# 	img = self.img_transforms(img)
				# else:
				# 	img = self.img_transforms_test(img)

				npyfilename = path.replace("Images/", "Images_resnet50_features/").replace(".jpg", ".npy")
				# npyfilename = os.path.join("/media/cheer/UI/Project/DailyReport", npyfilename)
				img = self.image_features[npyfilename]
				# img = torch.Tensor(np.load(npyfilename))

				imgs.append(img)

			imgs = torch.stack(imgs, dim = 0)
		else:
			imgs =torch.ByteTensor(1).zero_()

		#### get caption
		caption = data.anns[self.ids[idx]]['caption']
		# Convert caption (string) to word ids.
		words = nltk.tokenize.word_tokenize(str(caption).lower())

		# token2idx
		target = torch.LongTensor(self.max_tokens).zero_()
		# tgt_mask: this is for cnn
		tgt_mask = torch.ByteTensor(self.max_tokens).zero_()

		# add <start>
		words = ['<start>'] + words
		# cut words longer than (max_token-1)
		length = min(len(words), self.max_tokens-1)
		tgt_mask[:(length+1)] = 1
		words = words[:length]
		words += ['<end>']

		tmp = [vocab(token) for token in words]
		# use the modified data
		caption = " ".join(words[1:-1])
		target[:length+1] = torch.LongTensor(tmp)
		
		# tgt_mask_transformer for transformer
		tgt_mask_transformer = (target != 0).unsqueeze(-2)
		# print(length)
		tgt_mask_transformer = tgt_mask_transformer & Variable(self.subsequent_mask(target.size(-1)).type_as(tgt_mask_transformer.data))
		# print("tgt_mask_transformer:", tgt_mask_transformer)
		# add <end>
		length += 1

		data = [imgs, caption, target, tgt_mask, tgt_mask_transformer, first_img_id, length]
		
		if self.use_bert:
			current_bert_features = []
			for path in img_filenames:
				if self.use_no_aoa_data:
					bert_path = path.replace("Images/", "binary_features_no_aoa/").replace(".jpg", ".npy")
				else:
					bert_path = path.replace("Images/", "binary_features/").replace(".jpg", ".npy")
				
				# bert_path = os.path.join("/media/cheer/UI/Project/DailyReport", bert_path)
				current_bert_features.append(self.bert_features[bert_path].view(-1))
				# current_bert_features.append(torch.Tensor(np.load(bert_path)).view(-1))
			current_bert_features = torch.stack(current_bert_features, dim = 0).float()
			data.append(current_bert_features)
		else:	
			data.append(torch.ByteTensor(1).zero_())
		return data

	def __len__(self):
		return len(self.ids)

def collate_fn(data):
	"""Creates mini-batch tensors from the list of tuples (image, caption).
	
	We should build custom collate_fn rather than using default collate_fn, 
	because merging caption (including padding) is not supported in default.

	Args:
		data: list of tuple (image, caption). 
			- image: torch tensor of shape (3, 256, 256).
			- caption: torch tensor of shape (?); variable length.

	Returns:
		images: torch tensor of shape (batch_size, 3, 256, 256).
		targets: torch tensor of shape (batch_size, padded_length).
		lengths: list; valid length for each padded caption.
	"""
	# Sort a data list by caption length (descending order).
	data.sort(key=lambda x: x[6], reverse=True)
	#images, captions, targets, sentence_masks, tgt_mask_transformers, first_img_ids, lengths, current_bert_features  = zip(*data)

	new_data = []
	for idx, item in enumerate(zip(*data)):
		if idx in [0,7]:
			item = torch.cat(item, 0)
		elif idx not in [1, 5, 6]:
			item = torch.stack(item, 0)
		else:
			# 1:captions 5:img_ids 6:lengths
			item = list(item)
		new_data.append(item)
	return new_data

def get_loader(args, vocab, split, shuffle=True):
	"""Returns torch.utils.data.DataLoader for custom dataset."""
	# caption dataset
	dataset = AccessibilityDataset(args, vocab, split)
	
	# Data loader for our dataset
	# This will return (images, captions, lengths) for each iteration.
	# images: a tensor of shape (batch_size, 3, 224, 224).
	# captions: a tensor of shape (batch_size, padded_length).
	# lengths: a list indicating valid length for each caption. length is (batch_size).
	data_loader = DataLoader(dataset=dataset, 
						  batch_size=args.batch_size,
						  shuffle=shuffle,
						  num_workers=args.num_workers,
						  collate_fn=collate_fn, drop_last=True, pin_memory=True)
	return data_loader


