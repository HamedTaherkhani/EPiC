import os
import transformers
import torch

os.environ['TRANSFORMERS_CACHE'] = '/home/hamedth/projects/def-hemmati-ac/hamedth/hugging_face'

print(torch.cuda.get_device_name(1))
print(torch.cuda.device_count())

