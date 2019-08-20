import argparse
import torch
import datetime
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

def parseInput():
    parser = argparse.ArgumentParser(description='Train a model based on a dataset')
    parser.add_argument('data_dir', type=str, help='The directory containing the training and validation data.')
    parser.add_argument('--save_dir', default='checkpoints', type=str, help='The directory to save the checkpoint to.')
    parser.add_argument('--arch', default='vgg16',type=str, help='The model architecture to use (e.g. vgg16).')
    parser.add_argument('--learning_rate', default=0.003, type=float, help='The learning rate to use for training.')
    parser.add_argument('--hidden_units', default=256, type=int, help='The number of hidden units to use.')
    parser.add_argument('--epochs', default=3, type=int, help='The number of epochs to use for training.')
    parser.add_argument('--gpu', default=True, type=bool, help='Should the gpu be used for training?')
    return parser.parse_args()

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_valid_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    
    return train_data, trainloader, valid_data, validloader
def take_checkpoint(model, optimizer, class_to_idx, epoch, arch):
    checkpoint = {'model_state_dict': model.state_dict(),
                  'classifier': model.classifier,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'class_to_idx': class_to_idx, 
                  'epoch': epoch,
                  'model_arch': arch
                 }
    torch.save(checkpoint, f'{"".join(str(datetime.datetime.now()).split())}.pth')
    
def train(model, trainloader, train_data, validloader, device, args):
    model.to(device) 
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    steps = 0
    running_loss = 0
    print_every = 5

    for epoch in range(args.epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{args.epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {test_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
        print(f'End of Epoch {epoch + 1} taking checkpoint')
        take_checkpoint(model, optimizer, train_data.class_to_idx, epoch, args.arch)       
        
def setupModel(args, train_dataa):
    model = getattr(models, args.arch)(pretrained=True)
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    inputFeatures = model.classifier[0].in_features
    outputPossibilites = len(train_data.class_to_idx.keys())
    model.classifier = nn.Sequential(nn.Linear(inputFeatures, args.hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(args.hidden_units, outputPossibilites),
                                     nn.LogSoftmax(dim=1))
     
    return model

if __name__ == '__main__':
    args = parseInput()
    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    # load data
    train_data, trainloader, valid_data, validloader = load_data(args.data_dir)
    model = setupModel(args, train_data)
    train(model, trainloader, train_data, validloader, device, args)
   
    