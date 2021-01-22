from .image_models import ImgFeats as trans_ImgFeats
from .image_models_lstm import ImgFeats as lstm_ImgFeats 

def image_model(args):
    if args.caption_model == "lstm":
        return lstm_ImgFeats(args)
    elif args.caption_model == "transformer":
        return trans_ImgFeats(args)
    else:
        raise("wrong caption_model")
