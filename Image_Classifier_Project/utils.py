import torch
from torchvision import models
import json
import argparse

def save_checkpoint(path, model, optimizer, args):
    checkpoint = {'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx,
              'arch': args.arch,
              'optimizer': optimizer.state_dict(),
             }

    torch.save(checkpoint, path)
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == 'vgg19':
        model = models.vgg19(pretrained=True)
    else:
        model = models.densenet121(pretrained = True)

    optimizer = checkpoint['optimizer']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def load_cat_names(filename):
    with open(filename) as f:
        category_names = json.load(f, strict=False)
    return category_names