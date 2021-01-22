import matplotlib; matplotlib.use('Agg')
import os, argparse

from train import train 
from test import test
from test_beam import test_beam 
from opts import get_opt

def main():
	"""Train model and run inference on coco test set to output metrics"""

	args = get_opt()
	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

	if(args.is_train == 1):
		if not os.path.exists(args.model_dir):
			os.makedirs(args.model_dir)
		train(args)

	bestmodelfn = os.path.join(args.model_dir, 'model.pth')
	if(os.path.exists(bestmodelfn)):
		if(args.beam_size == 1):
			scores = test(args, 'test', modelfn=bestmodelfn)
		else:
			scores = test_beam(args, 'test', modelfn=bestmodelfn)

		print('TEST set scores')
		for k, v in list(scores[0].items()):
			print(('%s: %f' % (k, v)))
	else:
		raise Exception('No checkpoint found %s' % bestmodelfn)

if __name__ == '__main__': 
	main()
