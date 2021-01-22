import json
import os
import re
import numpy as np
from tqdm import tqdm

json_path = '/home/cheer/Project/DailyReport/data/code_features'
save_path = '/home/cheer/Project/DailyReport/data/binary_features'

def make_dir(folder_name):
  if not os.path.exists(os.path.join(save_path, folder_name)):
    print ('create folder {}'.format(folder_name))
    os.makedirs(os.path.join(save_path, folder_name))
  else:
    print ('folder {} exist'.format(folder_name))
  return os.path.join(save_path, folder_name)

def generate():
  json_list = []
  all_list = os.listdir(json_path)
  #for item in all_list:
  #  if re.search('java_0.', item):
  #    json_list.append(item)
  for json_file in tqdm(all_list):
    save_p = make_dir(os.path.splitext(json_file)[0])
    with open(os.path.join(json_path, json_file), 'r') as j_file:
      json_list = list(j_file)
    f = 0
    for json_str in json_list:
      data = json.loads(json_str)
      feature_matrix = np.zeros((64, 768), dtype = float)
      i = 0
      for feature in data['features']:
        for layer in feature['layers']:
          feature_matrix[i] = layer['values']
          i += 1
      np.save(os.path.join(save_p, '{:05}'.format(f)), feature_matrix)
      f += 1
      

if __name__ == '__main__':
  generate()
