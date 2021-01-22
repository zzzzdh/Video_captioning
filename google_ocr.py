from google.cloud import vision
import io
import json
import cv2
import os
import numpy as np
from tqdm import tqdm

data_dir = '/home/cheer/Project/DailyReport/data'

def make_dir(folder_name):
  save_path = os.path.join(data_dir, 'OCR')
  if not os.path.exists(os.path.join(save_path, folder_name)):
    print ('create folder {}'.format(folder_name))
    os.makedirs(os.path.join(save_path, folder_name))
  else:
    print ('folder {} exist'.format(folder_name))
  return os.path.join(save_path, folder_name)

def find_line_box(vertice_list):
  vertice_list = np.array(vertice_list)
  x_list = np.sort(vertice_list[:, 0])
  y_list = np.sort(vertice_list[:, 1])
  x_min = x_list[0]
  x_max = x_list[-1]
  y_min = y_list[0]
  y_max = y_list[-1]
  return np.array([x_min, y_min, x_max, y_max])
  
def detect_text(folder):
  client = vision.ImageAnnotatorClient()
  save_path = make_dir(folder)
  image_list = os.listdir(os.path.join(data_dir, 'Images', folder))
  for image_name in tqdm(image_list):
    with io.open(os.path.join(data_dir, 'Images', folder, image_name), 'rb') as image_file:
      content = image_file.read()
    word_count_list = []
    total_count = 1
    text_line_index = 0
    line_list = []
    image = vision.types.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    if len(texts) == 0:
      line_data = {}
      line_data['id'] = 0
      line_data['text'] = 'None'
      line_data['vertice'] = {'x_min':0, 'y_min':0, 'x_max':0, 'y_max':0}
      line_list.append(line_data)
      with open(os.path.join(save_path, image_name.replace('.jpg', '.json')), 'w') as json_file:
        json.dump({'lines':line_list}, json_file)
      continue
    text_lines = texts[0].description.split('\n')
    for text_line in text_lines:
      word_count_list.append(len(text_line.split()))
    for word_count in word_count_list:
      vertice_list = []
      vertice_line = texts[total_count:total_count + word_count]
      for text in vertice_line:
        for vertex in text.bounding_poly.vertices:
          vertice_list.append([vertex.x, vertex.y])
      if len(vertice_list):
        line_box = find_line_box(vertice_list)
        line_data = {}
        line_data['id'] = text_line_index
        line_data['text'] = text_lines[text_line_index]
        line_data['vertice'] = {'x_min':int(line_box[0]), 'y_min':int(line_box[1]), 'x_max':int(line_box[2]), 'y_max':int(line_box[3])}
        line_list.append(line_data)
      total_count += word_count
      text_line_index += 1
    with open(os.path.join(save_path, image_name.replace('.jpg', '.json')), 'w') as json_file:
      json.dump({'lines':line_list}, json_file)

  
if __name__=="__main__":
  folder = 'java_0_48'
  detect_text(folder)
