import os, argparse

from models.full_model import UnBlindModel 
from dataloaders.build_vocab import Vocabulary
from utils.opts import get_opt

if __name__ == '__main__':
    """Train model and run inference on coco test set to output metrics"""
    args = get_opt()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    #model = UnBlindModel(batchsize = 12, )sss
    model = UnBlindModel(args)
    if args.phase == 'train':
        model.train()
    else:
        if args.beam_size == 1:
            scores = model.test("test", 21, 31)
        else:
            scores = model.test_beam('test')
            
    model.writer.close()
