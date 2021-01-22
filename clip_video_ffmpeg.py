import glob
import os
import webvtt
from tqdm import tqdm

video_path = '/media/cheer/disk2/youtube/videos/java'
output_path = '/home/cheer/Project/DailyReport/data/Images'
language = 'java'
playlist_list = [1, 2, 3, 5, 8, 12, 14, 26, 30, 32, 33, 34, 38, 41, 42, 44, 46, 47, 48]

'''
1, 2, 3, 5, 8, 12, 14, 26, 30, 32, 33, 34, 38, 41, 42, 44, 46, 47
'''

def parse_vtt(file_name, save_dir):
  file_name = file_name.replace('.mp4', '.en.vtt')
  i = 1
  caption_list = []
  text = ''
  start = ''
  end = ''
  for caption in webvtt.read(file_name):
    if i % 2 == 0:
      text = text + ' ' + caption.text
      caption_list.append('{} {} {}\n'.format(int(caption.start.split(':')[0])*3600 + int(caption.start.split(':')[1])*60 + int(float(caption.start.split(':')[2])), int(caption.end.split(':')[0])*3600 + int(caption.end.split(':')[1])*60 + int(float(caption.end.split(':')[2])), text))
      text = '' 
    else:
      text = text + caption.text
    i += 1
  with open(save_dir.replace('Images', 'Captions') + '.txt', 'w') as caption_file:
    caption_file.writelines(caption_list)

def make_dir(playlist, video_number):
  folder_name = language + '_' + str(playlist) + '_' + str(video_number)
  if not os.path.exists(os.path.join(output_path, folder_name)):
    print ('create folder {}'.format(folder_name))
    os.makedirs(os.path.join(output_path, folder_name))
  else:
    print ('folder {} exist'.format(folder_name))
  return os.path.join(output_path, folder_name)

def clip(playlist, file_name, video_number):
  if os.path.exists(file_name.replace('.mp4', '.en.vtt')):
    save_dir = make_dir(playlist, video_number)
    parse_vtt(file_name, save_dir)
    cmd = "ffmpeg -i '" + file_name + "' -r 1 " + save_dir + "/%05d.jpg"
    os.system(cmd)

def start():
  for playlist in playlist_list:
    file_list = glob.glob(os.path.join(video_path, str(playlist), '*.mp4'))
    video_number = 0
    for file_name in tqdm(file_list):
      clip(playlist, file_name, video_number)
      video_number += 1

if __name__ == '__main__':
  start()
