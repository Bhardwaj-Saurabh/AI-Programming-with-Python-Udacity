import argparse

import sys
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim

from collections import OrderedDict
from utils import save_checkpoint

# check GPU if availabale for MAC M1 
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# check GPU if availabale for MAC M1 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Hyper parameters
img_dim_H = 256
img_dim_W = 256

epochs = 10
print_every = 25
batch_size_train = 32
batch_size_val = 32
batch_size_test = 10

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--data_dir', action='store')
    parser.add_argument('--arch', dest='arch', default='vgg19')
    parser.add_argument('--epochs', dest='epochs', default='3')
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.001')
    parser.add_argument('--use_cuda', action='store', default='False')
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth")
    return parser.parse_args()

def train(n_epochs, trainloader, validloader, model, optimizer, criterion, use_cuda):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    running_train_loss = []
    running_val_loss = []

    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.
        av_loss_train = 0

        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(trainloader):
            # move to GPU
            if use_cuda:
                data, target = data.to(device), target.to(device)
                
            ## find the loss and update the model parameters accordingly
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            ## record the average training loss
            #train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
            train_loss += loss.item() * len(target)
            av_loss_train = train_loss / ((batch_idx+1) * batch_size_train)

            stats = f'Epoch {epoch}\t Loss: {loss.item()}\t avg_Loss: {av_loss_train}'

            # Print training statistics (on same line).
            print('\r' + stats, end="")
            sys.stdout.flush()

            # Print training statistics (on different line).
            if (batch_idx+1) % print_every == 0:
                print('\r' + stats)

        ######################
        # validate the model #
        ######################
        model.eval()
        val_loss = 0.
        accuracy = 0 

        for batch_idx, (data, target) in enumerate(validloader):
            # move to GPU
            if use_cuda:
                data, target = data.to(device), target.to(device)
            ## find the loss and update the model parameters accordingly
            ## update the average validation loss
            with torch.no_grad():
                output = model(data)
                
                v_loss = criterion(output, target) # loss calculation

                output = torch.exp(output) # get exponents of output layer
                
                val_loss += v_loss.item()

                equals = (target.data == output.max(dim = 1)[1])
                accuracy += equals.float().mean().item() 

        #validation_loss = av_loss_vall / len(validloader)
        val_accuracy = (accuracy / len(validloader)) * 100
                
        # Get training/validation statistics
        stats = 'Epoch [%d/%d]\t  Val Loss: %.4f\t\t\t Val Accuracy: %.2f' % (epoch, n_epochs, val_loss/len(validloader), val_accuracy)   # valid_loss,

        # Print validation statistics (on new line).
        print(stats)
        sys.stdout.flush()

        # store losses for later use
        running_train_loss.append(av_loss_train)              #train_loss
        running_val_loss.append(val_loss/len(validloader))    #valid_loss

            
def main():
    args = parse_args()
    
    data_dir = '../flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    

    train_transform = transforms.Compose([transforms.Resize((img_dim_H,img_dim_W)),
                                            transforms.CenterCrop(224),
                                            transforms.RandomHorizontalFlip(0.5),
                                            transforms.RandomVerticalFlip(0.5),
                                            transforms.RandomRotation(45),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                std=(0.229, 0.224, 0.225))])

    valid_transform = transforms.Compose([transforms.Resize((img_dim_H,img_dim_W)),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                            std=(0.229, 0.224, 0.225))])

    test_transform = transforms.Compose([transforms.Resize((img_dim_H,img_dim_W)),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                        std=(0.229, 0.224, 0.225))])

    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size_train, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size_val, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_test, shuffle=False)
   
    model = getattr(models, args.arch)(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
    

    # define classifier for transfer learning
    classifier = nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(25088, 2048)),
                        ('relu', nn.ReLU()),
                        ('fc2', nn.Linear(2048, 256)),
                        ('relu', nn.ReLU()),
                        ('fc3', nn.Linear(256, 102)),
                        ('output', nn.LogSoftmax(dim=1))
                        ]))


    # Update the classifier in the model    
    model.classifier = classifier
    # Define the loss
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.learning_rate))
    epochs = int(args.epochs)
    class_index = train_data.class_to_idx
    # Get the gpu settings
    use_cuda = args.use_cuda
    train(epochs, train_loader, valid_loader, model, optimizer, criterion, use_cuda=use_cuda)
    model.class_to_idx = class_index
    # New save location
    path = args.save_dir  
    save_checkpoint(path, model, optimizer, classifier)

if __name__ == "__main__":
    main()