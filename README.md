# Video_captioning

## Programming Video Captioning with Action-Aware Multimodal Transformer

This is the repository of programming video captioning, which use natural language to describe programming workflows.

You can download the [dataset]() or construct your own dataset follwing the instructions. As the dataset is very large, we only upload part of it.

![](/examples/result.jpg)


## Please check more examples at 
https://htmlpreview.github.io/?https://github.com/zzzzdh/Video_captioning/blob/master/examples.html

## How to use

### Prepare data

Note: You need to change your own path for each process.

#### Video data
1. Download [video dataset](https://drive.google.com/file/d/1J3gCS_4JWfinZepA3dnzV0sp0bHHCIqv/view?usp=sharing) or prepare your own video data. The videos have to include subtitles.
2. Clip video and parse captions `python3 clip_video_ffmpeg.py`. By doing this, you can get the frames and subtitles.
3. Extract code from frames by [Google OCR](https://cloud.google.com/vision/docs/ocr), run `python3 google_ocr.py`. You may need to have developer account for using Google OCR, or you can use [tesseract](https://github.com/tesseract-ocr/tesseract) instead.

#### Code data
1. You can prepare your own code data for training BERT or download the [code dataset]() directly.
2. Run `python3 prepare_code_data.py` and parse the project to code, which looks like sentences.
3. Move to `bert` and fine tune BERT by `bash run_train.sh`.
4. Extrac code feature by `bash batch_extract_feature.sh`. The input is OCRed data from video data step 2.

### Generate training data
1. run `python3 generate_feature.py` to generate binary features.
2. run `python3 generate_coco_annotation.py` to generate training list.

### Lunch Training
1. run `train_transformer.sh`

### Evaluation
1. Move to `evaluation` and run `python3 eval.py`
