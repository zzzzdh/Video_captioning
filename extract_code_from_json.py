import os
import json
import re
import javalang
from tqdm import tqdm

data_dir = '/home/cheer/Project/DailyReport/data'

json_path = '/home/cheer/Project/DailyReport/data/OCR/java_0_23/00002.json'

def get_distance(box, line):
  box_x = int(box.split()[5])
  box_y = int(box.split()[6])
  line_x = (line['vertice']['x_max'] + line['vertice']['x_min']) / 2
  line_y = (line['vertice']['y_max'] + line['vertice']['y_min']) / 2
  return pow(box_x - line_x, 2) + pow(box_y - line_y, 2)

def sort_distance(val):
  return val['distance']

def sort_id(val):
  return val['id']

def extract(folder_list):
  for folder in tqdm(folder_list):
    text_list = []
    with open(os.path.join(data_dir, 'Annotations', folder + '.txt')) as box_file:
      box_list = box_file.readlines()
    for box in box_list:
      line_data = []
      tokenized_line = []
      result = []
      text_result = ''
      count = 0
      with open(os.path.join(data_dir, 'OCR', folder, box.split()[0] + '.json'), 'r') as json_file:
        json_data = json.load(json_file)
      for line in json_data['lines']:
        line_data.append({'id':line['id'], 'distance':get_distance(box, line), 'text':line['text']})
      line_data.sort(key = sort_distance)
      for line in line_data:
        token_list = []
        try:
          tokens = list(javalang.tokenizer.tokenize(line['text']))
          for token in tokens:
            if isinstance(token, javalang.tokenizer.String):
              value = "'String'"
            else:
              value = token.value
            token_list.append(value)
        except:
          pass
        line.update({'text':' '.join(token_list)})
        tokenized_line.append(line)
      for line in tokenized_line:
        if count < 64 and len(line['text'].split()):
          result.append(line)
          count += len(line['text'].split())
      result.sort(key = sort_id)
      for line in result:
        text_result = text_result + ' ' + line['text']
      text_result = text_result + '\n'
      text_list.append(text_result)
    with open(os.path.join(data_dir, 'OCR_text', folder + '.txt'), 'w') as text_file:
      text_file.writelines(text_list)

def extract_all(folder_list):
  for folder in tqdm(folder_list):
    image_list = os.listdir(os.path.join(data_dir, 'OCR', folder))
    text_list = []
    for image in image_list:
      tokenized_line = []
      result = []
      text_result = ''
      count = 0
      with open(os.path.join(data_dir, 'OCR', folder, image), 'r') as json_file:
        json_data = json.load(json_file)
      for line in json_data['lines']:
        token_list = []
        try:
          tokens = list(javalang.tokenizer.tokenize(line['text']))
          for token in tokens:
            if isinstance(token, javalang.tokenizer.String):
              value = "'String'"
            else:
              value = token.value
            token_list.append(value)
        except:
          pass
        tokenized_line.append(' '.join(token_list))
      for line in tokenized_line:
        if count < 64 and len(line.split()):
          result.append(line)
          count += len(line.split())
      for line in result:
        text_result = text_result + ' ' + line
      text_result = text_result + '\n'
      text_list.append(text_result)
    with open(os.path.join(data_dir, 'OCR_text_all', folder + '.txt'), 'w') as text_file:
      text_file.writelines(text_list)  

          
def start():
  folder_list = os.listdir(os.path.join(data_dir, 'OCR'))
  extract_list = []
  for folder in folder_list:
    if len(os.listdir(os.path.join(data_dir, 'OCR', folder))) == len(os.listdir(os.path.join(data_dir, 'Images', folder))) and re.search('java_0.', folder):
       extract_list.append(folder)
  extract(extract_list)
  #extract_all(extract_list)


if __name__ == '__main__':
  start()
