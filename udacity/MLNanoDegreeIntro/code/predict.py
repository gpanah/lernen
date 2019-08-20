import argparse
import torch
import datetime
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from PIL import Image
import torchvision.transforms as transforms
import numpy as n
import json

def parseInput():
    parser = argparse.ArgumentParser(description='Train a model based on a dataset')
    parser.add_argument('image_path', type=str, help='The path to a single image.')
    parser.add_argument('checkpoint_path', type=str, help='The path to a pytorch checkpoint file.')
    parser.add_argument('--top_k', default=5, type=int, help='The number of most likely classes to return.')
    parser.add_argument('--category_names',type=str, help='A mapping file of categories to names.')
    parser.add_argument('--gpu', default=True, type=bool, help='Should the gpu be used for training?')
    return parser.parse_args()

def load_checkpoint(path, device):
    checkpoint = torch.load(path)
    model = getattr(models, checkpoint['model_arch'])(pretrained=True)
    model.to(device)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['class_to_idx']
                             
def process_image(image):
    #Resize
    size = 256, 256
    im = Image.open(image)
    im.thumbnail(size)
    
    #Crop
    im = im.crop((6, 6, 250, 250))
    
    #Convert to Tensor with adjusts the order of the color channel and converts the 0-255 values to 0-1
    pil2tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    tensor = normalize(pil2tensor(im))
    return tensor

def convertIndexToClass(class_to_idx, value):
    return list(class_to_idx.keys())[list(class_to_idx.values()).index(value)]

def predict(model, class_to_idx, args):
    with torch.no_grad():
        tensor = process_image(args.image_path)
        tensor = tensor.cuda() if next(model.parameters()).is_cuda else tensor
        result = model(tensor.unsqueeze_(0))
        ps = torch.exp(result)
        probs, classes = ps.topk(args.top_k)
        preds = []
        for i in range(0,len(probs.data.cpu().numpy()[0])):
            preds.append(([convertIndexToClass(class_to_idx, idx) for idx in classes.data.cpu().numpy()[0]][i], probs.data.cpu().numpy()[0][i]))
        return preds
    
if __name__ == '__main__':
    args = parseInput()
    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    model, class_to_idx = load_checkpoint(args.checkpoint_path, device)
    predictions = predict(model, class_to_idx, args)
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        predictions = [(cat_to_name[pred[0]], pred[1]) for pred in predictions]
    print(predictions)