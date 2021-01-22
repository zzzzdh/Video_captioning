import csv
import os
import re
import random
import javalang

data_path = '/home/cheer/Project/DailyReport/data'
output_path = '/home/cheer/Project/DailyReport/data/Code/Github_repo'
#vocab = '/home/cheer/Project/DailyReport/data/Code/vocab.txt'
#vocab_list = []

def download():
  with open(os.path.join(data_path, 'Code', 'github.csv'), 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
      cmd = 'git clone https://github.com/' + row[2] + '.git ' + os.path.join(output_path, row[2].replace('/', '_')) 
      os.system(cmd)

def clean(java_file):
  code_list = []
  clean_list = []
  result = []
  try:
    with open(java_file, 'r') as j_file:
      code = j_file.readlines()
    for line in code:
      line = line.strip()
      if len(line) and not re.match('/\*', line) and not re.match('//', line) and not re.match('\*', line):
        code_list.append(line.replace('.', ' '))
  except:
    pass
  for i in range(len(code_list)):
    if not code_list[i] == '}' and not code_list[i] == '};':
      clean_list.append(code_list[i])
    else:
      clean_list[-1] = clean_list[-1] + code_list[i]
  i = 0
  if len(clean_list):
    result.append(clean_list[0])
  for line in clean_list:
    if len(result[i].split()) < 50 and len(line.split(' ')) < 10:
      result[i] = result[i] + ' ' + line
    else:
      result[i] = result[i] + '\n'
      result.append(line)
      i += 1
  result.append('\n')
  with open(os.path.join(data_path, 'Code', 'code.txt'), 'a') as code_file:
    code_file.writelines(result)

def generate_ast(java_file):
  result = []
  i = 0
  line_count = 0
  token_count = 0
  max_len = random.randint(30, 60)
  try:
    with open(java_file, 'r') as j_file:
      lines = j_file.readlines()
    code = ''
    for line in lines:
      code = code + ' ' + line
    tokens = list(javalang.tokenizer.tokenize(code))
    token_count = len(tokens)
    if len(tokens):
      result.append(tokens[0].value)
    del tokens[0]
    for token in tokens:
      if isinstance(token, javalang.tokenizer.String):
        value = "'String'"
      else:
        value = token.value
      #if value not in vocab_list:
      #  vocab_list.append(value)
      if len(result[i].split()) < max_len:
        result[i] = result[i] + ' ' + value
      else:
        result[i] = result[i] + '\n'
        result.append(value)
        i += 1
        line_count += 1
    result.append('\n')
    result.append('\n')
    line_count += 1
    print (java_file)
    with open(os.path.join(data_path, 'Code', 'code.txt'), 'a') as code_file:
      code_file.writelines(result)
  except:
    pass
  return line_count, token_count

    
def extract_java():
  file_count = 0
  line_count = 0
  token_count = 0
  for root, dirs, files in os.walk(output_path):
    for name in files:
      if re.search('.\.java', name):
        file_count += 1
        l_count, t_count = generate_ast(os.path.join(root, name)) 
        line_count += l_count
        token_count += t_count
  print ('files:{}, lines:{}, tokens:{}'.format(file_count, line_count, token_count))
  
  #with open(vocab, 'w') as vocab_file:
  #  vocab_file.writelines(vocab_list) 


if __name__ == '__main__':
  #download()
  extract_java()
