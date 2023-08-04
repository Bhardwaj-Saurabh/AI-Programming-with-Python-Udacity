import torch
from torchvision import models

import json

def save_checkpoint(path, model, optimizer, classifier):
    checkpoint = {'model': models.vgg19(pretrained=True),
              'input_size': 25088,
              'output_size': 102,
              'classifier': classifier,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx
             }

    torch.save(checkpoint, path)
    
def load_checkpoint(filepath):
    model_info = torch.load(filepath)
    model = model_info['model']
    model.classifier = model_info['classifier']
    model.load_state_dict(model_info['state_dict'])
    model.class_to_idx = model_info['class_to_idx']
    optimizer = model_info['optimizer']
    return model

def load_cat_names(filename):
    with open(filename) as f:
        category_names = json.load(f)
    return category_names