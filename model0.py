import torch
import torch.nn
import torch.optim
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets
import os
import matplotlib.pyplot as plt
import numpy as np

#=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)

    plt.pause(0.001)  # pause a bit so that plots are updated

def visualize_model(model, num_images=36):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    class_names = data_generator['train'].classes
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                #ax = plt.subplot(num_images//2, 2, images_so_far)
                ax = plt.subplot()
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[predictions[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

#=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+

def set_parameter_requires_grad(model, extracting): # small function for freezing layers
    # takes in model and a boolean as input values
    # model: pretrained model from torchvision.models
    # boolean: True = freeze layers, False = unfreeze layers
    if extracting:
        for param in model.parameters(): # if extracting, iterate over all model parameters (to freeze them)
            param.requires_grad = False # to freeze the parameters so that the gradients are not computed in backward() (gradient descent)

def train_model(model, data_loaders, criterion, optimizer, epochs=12): # big function to train model
    for epoch in range(epochs): # iterate through number of epochs (training loops)

        # printing debug information
        print("Epoch %d / %d" % (epoch, epochs-1)) # our progress
        print("-"*15) # looks nice

        # iterate through training and validation phases
        for phase in ["train", "val"]: # literal folder names in directory
            if phase == "train":
                model.train()
            else: # validation phase
                model.eval()

            running_loss = 0.0
            correct = 0 # total number of correct guesses

            # iterate over inputs and labels in data_loaders (an object that handles the loading of data from directory)
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad() # zero the parameter gradients for optimizer (every training loop)

                # FORWARD
                # track history if only in train
                with torch.set_grad_enabled(phase=="train"):
                    outputs = model(inputs) # feed forward pass
                    loss = criterion(outputs, labels) # calculate los
                    _, predictions = torch.max(outputs, 1) # get prediction class for outputs

                    # BACKWARD
                    # if training phase, we want to perform backpropagation and step the optimizer
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                correct += torch.sum(predictions == labels.data) # total number of correct predictions

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_acc = correct.double() / len(data_loaders[phase].dataset)

            # basic print statement for loss function and accuracy (after epoch); both to 4 decimal places
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase,
                                                       epoch_loss, epoch_acc))

#=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+

if __name__ == "__main__": # main loop for tying everything together

    root_dir = "data/sigh" # location of dataset

    # we are currently working with a small dataset
    # we will use data transformations (help turn 1 image into many) to increase the amount of data our network sees
    image_transforms = {
        "train": transforms.Compose([#transforms.RandomRotation((-270, 270)),
                 transforms.Resize((224, 224)), # resnet model expects 224x224 images
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])]),
        "val": transforms.Compose([#transforms.RandomRotation((-270, 270)),
               transforms.Resize((224, 224)),
               transforms.ToTensor(),
               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
    }

    # data_generator: object that maps the transforms to the images we load from the directory
    # opens up image folder dataset & performs the transforms by iterating over the keys in "train" and "val" folders
    data_generator = {k: torchvision.datasets.ImageFolder(os.path.join(root_dir, k),
                      image_transforms[k]) for k in ["train", "val"]}

    # data_generator needs a data_loader
    data_loader = {k: torch.utils.data.DataLoader(data_generator[k], batch_size=2, # takes data_generator as input
                   shuffle=True, num_workers=4) for k in ["train", "val"]} # num_workers is our desired number of dedicated threads

    # instantiating our device (local gpu) and the pretrained model (resnet18)
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(0)
    model = models.resnet18(pretrained=True) # resnet18 --> 18 layers in network

    # ! FREEZING [Y/N]!
    # set_parameter_requires_grad(model, True) --> Extracting = True ==> Freezing (FEATURE EXTRACTION)
    # set_parameter_requires_grad(model, False) --> Extracting = False ==> Not Freezing (FINE-TUNING)
    set_parameter_requires_grad(model, False)

    # the (binary) linear classifier that we will stick on top of our pretrained resnet18
    num_features = model.fc.in_features # fc = fully connected
    # we have 2 classes, so we have 2 outputs
    # again, this is just the fully connected final layer
    model.fc = torch.nn.Linear(num_features, 2) # nn.Linear(num_features, len(class_names))
    model.to(device) # send to our device

    criterion = torch.nn.CrossEntropyLoss() # the loss we use for multiclass classification
    # optimize model parameters with a learning rate of 0.001 using the Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # catching potential errors
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad is True:
            params_to_update.append(param)
            print("\t", name)

    # TRAINING
    train_model(model, data_loader, criterion, optimizer)
    # fc.weight and fc.bias = the parameters to update when freezing=True

    # VISUALIZING PREDICTIONS
    visualize_model(model)

    # SAVING MODEL
    PATH = r"C:\Users\stuar\PycharmProjects\neural_net"
    # when saving a model for inference, it is only necessary to save the trained model’s learned parameters
    torch.save(model.state_dict(), os.path.join(PATH, "model1.pth")) # saves .pth file in PATH



