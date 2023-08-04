import argparse
import torch
from torchvision import transforms
from PIL import Image
from utils import load_checkpoint, load_cat_names

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', action='store', default='../check.pth')
    parser.add_argument('--top_k', dest='top_k', default='5')
    parser.add_argument('--pathname', dest='pathname', default='flowers/test/1/image_06743.jpg')
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store', default='gpu')
    return parser.parse_args()

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    image_pil = Image.open(image)
   
    adjustments = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                     ])
    
    image = adjustments(image_pil)
    return image

def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path)
    image = image.unsqueeze(0) 
    image = image.to(device)
    
    # to make sure no drop; otherwise, the output is random
    
    with torch.no_grad ():
        output = model.forward(image)
        
    output_prob = torch.exp(output)
    
    probs, indices = output_prob.topk(topk)
    
    probs   = probs.to('cpu').numpy().tolist()[0]
    indices = indices.to('cpu').numpy().tolist()[0]
    
    # Displays the category name
    index_to_class = {val:cat_to_name[k] for k, val in model.class_to_idx.items()}  

    classes = [index_to_class[i] for i in indices]
    
    return probs, classes

def main(): 
    args = parse_args()
    device = args.gpu
    model = load_checkpoint(args.checkpoint)
    cat_to_name = load_cat_names(args.category_names)
    
    img_path = args.pathname
    top_p, classes = predict(img_path, model, device, topk=5)
    labels = [cat_to_name[str(index)] for index in classes]
    print('File selected: ' + img_path)
    
    print(labels)
    print(top_p)
    
    # According to user this prints out top k classes and probs(top_p) 
    i=0 
    
    while i < len(labels):
        print("{} with a probability of {}".format(labels[i], top_p[i]))
        i += 1

if __name__ == "__main__":
    main()