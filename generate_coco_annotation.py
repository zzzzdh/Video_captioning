import os 
import json
import numpy as np
import math
from glob import glob

data_dir = '/home/cheer/Project/DailyReport/data'
stride = 8

def resample(start, end, image_len):
  input_list = np.arange(start, end + 1)
  if len(input_list) in range(int(stride / 2) + 1, stride):
    sample_list = np.append(input_list, np.random.choice(input_list, stride - len(input_list)))
    sample_list = np.sort(sample_list)
  else:
    input_list = np.repeat(input_list, math.ceil(stride / len(input_list)))
    sample_list = np.random.choice(input_list, stride)
  return np.clip(np.sort(sample_list), 1, image_len - 1)

def main():
  sample_list = []
  video_list = os.listdir(os.path.join(data_dir, 'binary_features'))
  #video_list = glob(os.path.join(data_dir, 'binary_features', 'java_0*'))
  for video in video_list:
    sample_list.append(video.split('/')[-1])
  video_list = sample_list
  image_id = 0
  annotation_id = 0
  images = []
  annotations = []
  for video in video_list[0:10]:
    image_list = os.listdir(os.path.join(data_dir, 'Images', video))
    image_len = len(image_list)
    with open(os.path.join(data_dir, 'Captions', video + '.txt'), 'r') as caption_file:
      captions = caption_file.readlines()
    for caption in captions:
      start = int(caption.split()[0])
      end = int(caption.split()[1])
      text = ' '.join(caption.split()[2:])
      resample_list = resample(start, end, image_len)
      annotation = {}
      annotation['id'] = annotation_id
      annotation['image_id'] = image_id
      annotation['caption'] = text
      annotations.append(annotation)
      annotation_id += 1
      for sample in resample_list:
        img = {}
        img['id'] = image_id
        img['file_name'] = os.path.join('Images', video, '{:05}'.format(sample) + '.jpg')
        images.append(img)
        image_id += 1
  with open(os.path.join(data_dir, 'annotation_list', 'test_list.json'), 'w') as json_file:
    json.dump({'images':images, 'annotations':annotations}, json_file)

  images = []
  annotations = []
  for video in video_list[-10:]:
    image_list = os.listdir(os.path.join(data_dir, 'Images', video))
    image_len = len(image_list)
    with open(os.path.join(data_dir, 'Captions', video + '.txt'), 'r') as caption_file:
      captions = caption_file.readlines()
    for caption in captions:
      start = int(caption.split()[0])
      end = int(caption.split()[1])
      text = ' '.join(caption.split()[2:])
      resample_list = resample(start, end, image_len)
      annotation = {}
      annotation['id'] = annotation_id
      annotation['image_id'] = image_id
      annotation['caption'] = text
      annotations.append(annotation)
      annotation_id += 1
      for sample in resample_list:
        img = {}
        img['id'] = image_id
        img['file_name'] = os.path.join('Images', video, '{:05}'.format(sample) + '.jpg')
        images.append(img)
        image_id += 1
  with open(os.path.join(data_dir, 'annotation_list', 'validation_list.json'), 'w') as json_file:
    json.dump({'images':images, 'annotations':annotations}, json_file)

  images = []
  annotations = []
  for video in video_list[8:-8]:
    image_list = os.listdir(os.path.join(data_dir, 'Images', video))
    image_len = len(image_list)
    with open(os.path.join(data_dir, 'Captions', video + '.txt'), 'r') as caption_file:
      captions = caption_file.readlines()
    for caption in captions:
      start = int(caption.split()[0])
      end = int(caption.split()[1])
      text = ' '.join(caption.split()[2:])
      resample_list = resample(start, end, image_len)
      annotation = {}
      annotation['id'] = annotation_id
      annotation['image_id'] = image_id
      annotation['caption'] = text
      annotations.append(annotation)
      annotation_id += 1
      for sample in resample_list:
        img = {}
        img['id'] = image_id
        img['file_name'] = os.path.join('Images', video, '{:05}'.format(sample) + '.jpg')
        images.append(img)
        image_id += 1
  with open(os.path.join(data_dir, 'annotation_list', 'train_list.json'), 'w') as json_file:
    json.dump({'images':images, 'annotations':annotations}, json_file)
        

if __name__ == '__main__':
  main()
