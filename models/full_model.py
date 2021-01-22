import argparse, os, time, pickle, random, sys, json
import numpy as np
from tqdm import tqdm 

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

from dataloaders.data_loader import get_loader 
#from dataloaders.data_loader_coco import get_loader as get_loader_coco
from dataloaders.build_vocab import Vocabulary
from utils.evaluate import language_eval
from utils.opts import get_opt
from utils.optim import NoamOpt, LabelSmoothing
from utils.lr_scheduler import LR_Scheduler

from models import image_models
import models

class UnBlindModel(object):
    def __init__(self, args):
        super(UnBlindModel, self).__init__()
        self.args = args

        self.num_epochs = args.num_epochs
        self.max_tokens = args.max_tokens
        self.split = args.split
        self.batch_size = args.batch_size
        self.caption_model = args.caption_model
        self.log_step = args.log_step
        self.score_select = args.score_select
        self.save_step = args.save_step
        self.image_model = args.image_model
        self.caption_model = args.caption_model
        self.img_fatures_size = args.img_fatures_size
        self.bert_size = args.bert_size
        self.lr = args.learning_rate
        self.use_coco = args.use_coco

        # Device configuration
        self.device = torch.device('cuda')# if torch.cuda.is_available() else 'cpu')
        print("device:", self.device)

        # Load vocabulary
        with open(args.vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        
        if self.use_coco:
            if args.phase == "train":
                self.train_loader = get_loader_coco(args, self.vocab, "train2014", shuffle=True) 
                self.val_loader = get_loader_coco(args, self.vocab, "val2014", shuffle=False) 
            self.test_loader = get_loader_coco(args, self.vocab, "val2014", shuffle=False) 
        else:
            # Build data loader
            if args.phase == "train":
                self.train_loader = get_loader(args, self.vocab, "train", shuffle=True) 
                self.val_loader = get_loader(args, self.vocab, "validation", shuffle=False) 
            self.test_loader = get_loader(args, self.vocab, "test", shuffle=False) 

        self.numwords = self.test_loader.dataset.numwords
        self.vocab_len = len(self.vocab)
        print("vocab_len:", self.vocab_len)
        self.idx2word = self.vocab.idx2word

        self.args.numwords = self.numwords
        self.args.vocab_len = self.vocab_len

        # Create result model directory
        self.model_path = os.path.join(args.model_path, self.image_model, self.caption_model, "model")

        if args.use_coco:
            self.model_path += "_coco"
        if args.use_img:
            self.model_path += "_img"
            if args.finetune_cnn:
                self.model_path += "_finetune"
        if args.use_bert:
            self.model_path += "_bert"
            if args.use_no_aoa_data:
                self.model_path += "_no_aoa"
        self.args.model_path = self.model_path

        print(self.model_path)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.bestmodel_path = os.path.join(self.model_path, 'best_model.ckpt')
        self.args.model_path =self.bestmodel_path

        self.vis_path = os.path.join(self.model_path, "vis")

        if not os.path.exists(self.vis_path):
            os.makedirs(self.vis_path)

        self.writer = SummaryWriter(log_dir=self.model_path)

        # one-hot
        self.trans_input_fatures_size = 0
        if self.args.use_img:
            self.trans_input_fatures_size += self.img_fatures_size
        if self.args.use_bert:
            self.trans_input_fatures_size += self.bert_size
        print("# args.trans_input_fatures_size:", self.trans_input_fatures_size)
        self.args.trans_input_fatures_size = self.trans_input_fatures_size
        self.build_model()

        # criterion
        if self.caption_model == "transformer":
            self.criterion = LabelSmoothing(size=self.args.vocab_len, padding_idx=0, smoothing=0.1)
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.CrossEntropyLoss()
        self.criterion.to(self.device)
        
        # optimizer
        self.params = list(self.decoder.parameters()) 
        if args.caption_model == "lstm":
            self.params += list(self.encoder.linear_bert.parameters())
        # if args.finetune_cnn:
        #     self.params += list(self.encoder.parameters())
        # else:
        #     self.params += list(self.encoder.linear_bert.parameters()) ###!!!! open when use lstm
        if self.caption_model != "transformer":
            self.optimizer = torch.optim.Adam(self.params, lr=self.lr)
            # optimizer = torch.optim.RMSprop(params, lr=args.learning_rate)
            #self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.args.lr_step_size, gamma=.1)
            self.scheduler = LR_Scheduler('poly', self.lr, self.num_epochs, iters_per_epoch = len(self.train_loader))
        else:
            # d_model = embed_size
            self.optimizer = NoamOpt(self.args.embed_size, 1, 4000,  \
                    torch.optim.Adam(self.params, lr=0, betas=(0.9, 0.98), eps=1e-9))

        # resume model
        self.start_epoch = 0
        self.bestscore = 0
        # check if all necessary files exist 
        if getattr(self.args, "checkpoint", None) is not None:
            print("[INFO] Loading best model from checkpoint:", self.args.checkpoint)
            checkpoint = torch.load(self.args.checkpoint)
            self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])

            # if self.caption_model == "transformer":
            #     self.optimizer.optimizer.load_state_dict((checkpoint['optimizer']))
            # else:
            #     self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = checkpoint["epoch"] + 1
            if self.score_select == checkpoint["score_select"]:
                self.bestscore = checkpoint[self.score_select]
                if isinstance(self.bestscore ,dict):
                    self.bestscore = self.bestscore[self.score_select]

    # Build the models
    def build_model(self):
        self.encoder = image_models.image_model(self.args)
        self.args.features_size = self.encoder.features_size
        self.decoder = models.setup(self.args)

        self.encoder.to(self.device)
        self.encoder.train()
        self.decoder.to(self.device)
        self.decoder.train()


    def forward(self,images, targets, lengths=None, bert_features=None, trans_tgt_masks=None):
        if self.caption_model == "lstm":
            features = self.encoder(images, bert_features)
            outputs = self.decoder(features, targets, lengths)
            return outputs

        elif self.caption_model == "convcap":
            imgsfeats, imgsfc7 = self.encoder(images, bert_features)
            _, _, feat_h, feat_w = imgsfeats.size()

            wordact, attn = self.decoder(imgsfeats, imgsfc7, targets)
            attn = attn.view(self.batch_size, self.max_tokens, feat_h, feat_w)
            wordact = wordact[:,:,:-1]

            return wordact, attn, feat_h, feat_w

        elif self.caption_model == "transformer":
            trans_tgt_masks = trans_tgt_masks.to(self.device)

            # t5 = time.time()
            features = self.encoder(images, bert_features)
            # t6= time.time()
            # print("img encoder:", t6-t5)
            outputs = self.decoder(features, targets[:, :-1], trans_tgt_masks[:,:-1, :-1])
            # t7= time.time()
            # print("decoder:", t7-t6)

            # reshape  --> (batch_size*seq_len, vocab_len)
            outputs = outputs.contiguous().view(-1, outputs.size(-1))
            return outputs
        else:
            raise NotImplementedError("This model is not implemented! PLease use transformer/lstm/convcap")

    def compute_loss(self, outputs, targets, lengths, masks=None, attn=None, wordact=None, feat_h=None, feat_w=None):
        if self.caption_model == "lstm":
            targets_pad = pack_padded_sequence(targets, lengths, batch_first=True)[0]
            loss = self.criterion(outputs, targets_pad)

        elif self.caption_model == "convcap":
            targets = targets[:,1:]
            masks = masks[:,1:].contiguous()

            wordact_t = wordact.permute(0, 2, 1).contiguous().view(\
                        self.batch_size*(self.max_tokens-1), -1)
            wordclass_t = targets.contiguous().view(\
                          self.batch_size*(self.max_tokens-1), 1)

            maskids = torch.nonzero(masks.view(-1)).numpy().reshape(-1)

            addition = 0
            if self.args.attention:
                addition = (torch.sum(torch.pow(1. - torch.sum(attn, 1), 2))) \
                           /(self.batch_size*feat_h*feat_w)

            loss = self.criterion(wordact_t[maskids, ...], \
                       wordclass_t[maskids, ...].contiguous().view(maskids.shape[0])) +\
                       addition
        elif self.caption_model == "transformer":
            targets = targets[:, 1:].contiguous().view(-1)
            # targets_pad = pack_padded_sequence(targets, lengths, batch_first=True)[0]
            # loss = self.criterion(outputs, targets)
            # reshape --> (batch_size*seq_len)
            loss = self.criterion(outputs, targets)/sum(lengths)
            # print("criterion:", torch.argmax(outputs, dim=1))
            # print("target:", targets)

        return loss

    def train(self):
        total_step = len(self.train_loader)
        bestscore = self.bestscore

        for epoch in range(self.start_epoch, self.num_epochs):
            print("\n==>Epoch:", epoch)
            loss_train = 0
            print(len(self.train_loader))
            # iter_end = time.time()
            for iteration, current_batch in enumerate(tqdm(self.train_loader)):
                # t0 = time.time()
                if self.caption_model != "transformer":
                    # update lr
                    #self.scheduler.step() 
                    self.scheduler(self.optimizer, iteration, epoch, bestscore)   
                # random.shuffle(self.train_loader.dataset.ids)
                #self.writer.add_scalar("Overall/learning_rate", self.lr, epoch*total_step + iteration)

                # t1 = time.time()
                # print("shuffle:", t1-t0)
                # print("end to start:", t1-iter_end)
                iter_start = time.time()
                images, captions, targets, masks, trans_tgt_masks, img_ids, lengths, bert_features = current_batch
                # t2 = time.time()
                # print("Reading data:", t2-t1)

                images = images.to(self.device)
                targets = targets.to(self.device)
                if self.args.use_bert:
                    bert_features = bert_features.to(self.device)

                # t3 = time.time()
                # print("Feature put to cuda:", t3-t2)


                if self.caption_model == "lstm":
                    outputs = self.forward(images, targets, lengths, bert_features)
                    loss = self.compute_loss(outputs, targets, lengths)

                elif self.caption_model == "convcap":
                    wordact, attn, feat_h, feat_w = self.forward(images, targets, bert_features=bert_features)
                    loss = self.compute_loss(None, targets, None, masks, attn, wordact, feat_h, feat_w)

                elif self.caption_model == "transformer":

                    outputs = self.forward(images, targets, lengths, bert_features, trans_tgt_masks)
                    # t4 = time.time()
                    loss = self.compute_loss(outputs, targets, lengths)
                    # print("compute loss:", time.time()-t4)

                self.writer.add_scalar("Loss/train", loss.item(), epoch*total_step + iteration)
                loss_train = loss_train + loss.item()

                # t9 = time.time()
                self.decoder.zero_grad()
                self.encoder.zero_grad()
                loss.backward()
                self.optimizer.step()

                del outputs, loss, images, captions, targets, masks, trans_tgt_masks, img_ids, lengths, bert_features
                # print("backward:", time.time()-t9)
                # print("all time:", time.time() - t1)

                iter_end = time.time()

            # Print log info
            self.writer.add_scalar("Loss/total_train_loss", loss_train, epoch)

            # Save the model checkpoints & evaluation
            # Run on validation and obtain score
            scores = self.test('validation', epoch, iteration)  
            score = scores[0][self.score_select]

            for m,s in scores[0].items():
                self.writer.add_scalar("Acc/{}".format(m), s, epoch)

            if (epoch+1) % self.save_step == 0:
                # print()
                if(score > bestscore):
                    print("[INFO] save model")
                    save_path = self.save_model(epoch, iteration, scores[0])
            
                    bestscore = score
                    print(('[DEBUG] Saving model at epoch %d with %s score of %f'\
                        % (epoch, self.score_select, score)))
                    os.system('cp %s %s' % (save_path, self.bestmodel_path))

                
        # Run on validation and obtain score
        scores = self.test('validation', epoch, iteration) 
        score = scores[0][self.score_select]
        for m,s in scores[0].items():
            self.writer.add_scalar("Acc/each_save_interval/{}".format(m), s, epoch*total_step + iteration)

        print("[INFO] save last model")
        save_path = self.save_model(epoch, iteration, score)

        if (score > bestscore):
            bestscore = score
            print(('[DEBUG] Saving model at epoch %d with %s score of %f'\
                % (epoch, self.score_select, score)))
            os.system('cp %s %s' % (save_path, self.bestmodel_path))

        # test
        self.test("test", epoch, iteration)

    def save_model(self, epoch, iteration, score):
        save_path = os.path.join(self.model_path, 'model-{}-{}.ckpt'.format(epoch, iteration))
        if self.caption_model == "transformer":
            optim_state_dict = self.optimizer.optimizer.state_dict()
        else:       
            optim_state_dict = self.optimizer.state_dict()

        save_data = {'epoch': epoch,
                'decoder_state_dict': self.decoder.state_dict(),
                'encoder_state_dict': self.encoder.state_dict(),
                'optimizer': optim_state_dict,
                'iteration': iteration, 
                'score_select': self.score_select,
                self.score_select: score,
                }
        # score = {self.score_select:score}
        # save_data.update(score)
        torch.save(save_data, save_path)
        return save_path


    def test(self, split, epoch, iteration):
        if split == "validation":
            current_dataloader = self.val_loader ### need to modify to validation
            # split = "train"
        elif split == 'test':
            # if self.use_coco:
            #     split = "val2014"
            # else:
            #     split="train"
            current_dataloader = self.test_loader
            # split = "train"

        print("[INFO] Evaluate {} with {} images ({} batches)".format(split.upper(), \
               len(current_dataloader.dataset.ids),len(current_dataloader)))

        self.encoder.eval()
        self.decoder.eval()

        pred_captions = []
        for i, current_batch in enumerate(tqdm(current_dataloader)):
            images, captions, _, _, _, img_ids, _, bert_features = current_batch

            if self.args.use_bert:
                bert_features = bert_features.to(self.device)

            images = images.to(self.device)

            if self.caption_model == "lstm":
                features = self.encoder(images, bert_features)
                sentence_ids = self.decoder.sample(features).cpu().numpy()

                # Convert word_ids to words
                for j in range(self.batch_size):
                    sampled_caption = []
                    word_raw_id = []
                    for word_id in sentence_ids[j]:
                        word = self.idx2word[word_id]
                        word_raw_id.append(word_id)
                        if word == '<end>':
                            break
                        sampled_caption.append(word)
                    word_raw_id = word_raw_id[1:]
                    sentence = ' '.join(sampled_caption[1:])
                    word_raw_id = [str(raw) for raw in word_raw_id]
                    pred_captions.append({'image_id': img_ids[j], 'caption': sentence, "gt_caption":captions[j]})
        
            elif self.caption_model == "convcap":
                imgsfeats, imgsfc7 = self.encoder(images, bert_features)
                _, featdim, feat_h, feat_w = imgsfeats.size()

                wordclass_feed = np.zeros((self.batch_size, self.max_tokens), dtype='int64')
                wordclass_feed[:,0] = self.vocab('<start>') 

                outcaps = np.empty((self.batch_size, 0)).tolist()

                for j in range(self.max_tokens-1):
                    wordclass = Variable(torch.from_numpy(wordclass_feed)).cuda()

                    wordact, _ = self.decoder(imgsfeats, imgsfc7, wordclass)

                    wordact = wordact[:,:,:-1]
                    # batch_size*max_token_len-1, vocab_len
                    wordact_t = wordact.permute(0, 2, 1).contiguous().view(self.batch_size * (self.max_tokens-1), -1)

                    wordprobs = F.softmax(wordact_t, dim=1).cpu().data.numpy()
                    wordids = np.argmax(wordprobs, axis=1)

                    word_raw_id = [[]]*self.batch_size
                    for k in range(self.batch_size):
                        word = self.idx2word[wordids[j+k*(self.max_tokens-1)]]
                        outcaps[k].append(word)
                        word_raw_id[k].append(wordids[j+k*(self.max_tokens-1)])
                        if(j < self.max_tokens-1):
                            wordclass_feed[k, j+1] = wordids[j+k*(self.max_tokens-1)]
                
                for j in range(self.batch_size):
                    num_words = len(outcaps[j]) 
                    if '<end>' in outcaps[j]:
                        num_words = outcaps[j].index('<end>')
                    outcap = ' '.join(outcaps[j][:num_words])
                    
                    current_word_raw_id = word_raw_id[k]#[:num_words]
                    current_word_raw_id = [str(raw) for raw in current_word_raw_id]
                    pred_captions.append({'image_id': img_ids[j], 'caption': outcap, "gt_caption":captions[j]})

            elif self.caption_model == "transformer":
                features = self.encoder(images, bert_features)
                sentence_ids = self.decoder.evaluate(features, self.max_tokens).cpu().numpy()
        
                # Convert word_ids to words
                for j in range(self.batch_size):
                    sampled_caption = []
                    word_raw_id = []
                    for word_id in sentence_ids[j]:
                        word = self.idx2word[word_id]
                        word_raw_id.append(word_id)
                        if word == '<end>':
                            break
                        sampled_caption.append(word)
                    sentence = ' '.join(sampled_caption[1:])
                    word_raw_id = word_raw_id[1:]
                    word_raw_id = [str(raw) for raw in word_raw_id]
                    # print("predicted:", sentence)
                    # print("gt:", captions[j])
                    pred_captions.append({'image_id': img_ids[j], 'caption': sentence, "gt_caption":captions[j]})

            del images, captions,img_ids,bert_features
        print(pred_captions[0:2])
        # Calculate scores
        scores = language_eval(self.args, pred_captions, self.model_path, split, epoch, iteration)

        metrics = list(scores[0].keys())
        metrics.sort()

        for m in metrics:
            print("{}: {}".format(m.upper(), scores[0][m]))

        if split == "test":# and self.args.vis:
            print("[INFO] visualizing...")
            vis_folder = self.vis_path 
            target = os.path.join(vis_folder, "imgs")
            if not os.path.exists(target):
                os.makedirs(target)
            data = current_dataloader.dataset.data
            '''
            # save img
            for pred in pred_captions:
                img_id = pred["image_id"]
                path = data.loadImgs(img_id)[0]['filename']
                img_path = os.path.join(args.image_root, split, path)
                os.system("cp {} {}".format(img_path, target))
            '''
            # in order to save space, we use the original location of img to show them
            gt = {}
            for k in range(len(pred_captions)):
                pred = pred_captions[k]
                img_id = pred["image_id"]
                path = data.loadImgs(img_id)[0]['file_name']
                # need absolute path
                img_path = os.path.join(self.args.image_root, split, path)
                pred_captions[k]["img_path"] = img_path
            with open(os.path.join(vis_folder, "vis_gt.json"),"w") as f:
                json.dump(pred_captions, f)
        
        self.encoder.train() 
        self.decoder.train()
        
        return scores

    def test_beam(self, split = "test"):
        raise NotImplementedError("Currently not support beam size > 1") 