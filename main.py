# https://github.com/pytorch/examples/blob/master/mnist/main.py

import torch
import torch.optim as optim

from torchvision import datasets, transforms

import torch.utils

import numpy as np

torch.manual_seed(3)


def train_epoch(model, trainloader, criterion, device, optimizer):
    """
    Train the model on the current epoch.
    :param model: Predefined model (e.g. onelinear)
    :param trainloader: Loads training data
    :param criterion: loss function
    :param device: device to run on
    :param optimizer: SGD optimizer
    :return: losses for logging
    """
    model.train()  # IMPORTANT!!! Set the model to training mode

    losses = []
    for batch_idx, data in enumerate(trainloader):
        inputs = data[0].to(device)
        labels = data[1].to(device)

        outputs = model(inputs) # run model on inputs
        loss = criterion(outputs, labels) # calculate losses
        losses.append(loss.item())

        optimizer.zero_grad()  # reset accumulated gradients
        loss.backward()  # compute new gradients
        optimizer.step()  # apply new gradients to change model parameters

    return losses # for logging: accumulated losses


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates model with eval data
    :param model: custom model
    :param dataloader: eval data loader
    :param criterion: loss function
    :param device: device to run on
    :return: evaluation (accurracy, avg loss)
    """
    model.eval()  # IMPORTANT!!! put model into eval mode

    with torch.no_grad():  # do not record computations for computing the gradient

        datasize = 0
        accuracy = 0
        avgloss = 0
        for ctr, data in enumerate(dataloader):

            # print ('epoch at',len(dataloader.dataset), ctr)

            inputs = data[0].to(device)
            outputs = model(inputs)

            labels = data[1]

            # computing some loss
            cpuout = outputs.to('cpu') # cpu is important
            if criterion is not None:
                curloss = criterion(cpuout, labels)
                avgloss = (avgloss * datasize + curloss) / (datasize + inputs.shape[0])

            # for computing the accuracy
            labels = labels.float()
            _, preds = torch.max(cpuout, 1)  # get predicted class
            accuracy = (accuracy * datasize + torch.sum(preds == labels)) / (datasize + inputs.shape[0])

            datasize += inputs.shape[0]  # update datasize used in accuracy comp

    if criterion is None:
        avgloss = None

    return accuracy, avgloss


def train_modelcv(dataloader_cvtrain, dataloader_cvtest, model, criterion, optimizer, scheduler, num_epochs, device):
    """
    Train the actual model
    :param dataloader_cvtrain: Loads training data
    :param dataloader_cvtest: Loads eval data
    :param model: Our defined model (e.g. onelinear)
    :param criterion: Loss function
    :param optimizer: SGD to not have to optimize all values
    :param scheduler: not used currently
    :param num_epochs: How many times we want to go over the dataset
    :param device: What device we want to use
    :return: trained values (best epoch, best measure, best weights)
    """
    best_measure = 0
    best_epoch = -1

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1)) # run for every epoch
        print('-' * 10)

        losses = train_epoch(model, dataloader_cvtrain, criterion, device, optimizer) # train actual model on epoch
        # scheduler.step()
        measure, _ = evaluate(model, dataloader_cvtest, criterion=None, device=device) # evaluate what came out of it

        print(' perfmeasure', measure.item())

        # store current parameters because they are the best or not?
        if measure > best_measure:  # > or < depends on higher is better or lower is better?
            bestweights = model.state_dict()
            best_measure = measure
            best_epoch = epoch
            print('current best', measure.item(), ' at epoch ', best_epoch)

    return best_epoch, best_measure, bestweights


class onelinear(torch.nn.Module):
    def __init__(self, dims, numout):
        super().__init__()  # initialize base class

        self.bias = torch.nn.Parameter(data=torch.zeros(numout), requires_grad=True) # set zero bias
        self.w = torch.nn.Parameter(data=torch.randn((dims, numout)),
                                    requires_grad=True)  # random init shape must be (dims, numout), requires_grad to True

    def forward(self, x):
        # compute the prediction over batched input x

        # print(x.size()) # (batchsize,dims)
        # print(self.w.size())

        v = x.view((-1, 28 * 28))  # flatten the image to (batchsize,dims), -1 allows to guess the number of elements
        y = self.bias + torch.mm(v, self.w) # w * x + b

        return y


class areallyoldschoolneuralnet(torch.nn.Module):  # google for Nirvana Dumb :)
    def __init__(self, indims, numcl):
        super().__init__()

        # your code here
        self.fc1 = None  # for one neural network layer
        # you may need to define more linear layers

        # for a better model: convolutions, (dropout)

    def forward(self, x):
        v = x.view((-1, 28 * 28))  # flattens the (batch, 28, 28) into a (batch, 28*28)

        # your code here

        return None


def run():
    # parameters
    batches = 32
    maxnumepochs = 3 # how many passes through the data should be done?

    # device=torch.device("cuda:0")
    device = torch.device("mps") # optimize for apple silicon (swap for "cpu" otherwise)

    dataTransformer = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)) # dataset weights
        ])

    dataset = {
        'trainingValues': datasets.FashionMNIST('./data', train=True, download=True, transform=dataTransformer),
        'testingValues': datasets.FashionMNIST('./data', train=False, download=True, transform=dataTransformer)
    }
    numberOfClasses = 10
    totalDimensions = 784

    dataloaders = {
        'train': torch.utils.data.DataLoader(dataset['trainingValues'], batch_size=batches, shuffle=False,
                                             sampler=torch.utils.data.sampler.SubsetRandomSampler(np.arange(50000))),
        'val': torch.utils.data.DataLoader(dataset['trainingValues'], batch_size=batches, shuffle=False,
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               np.arange(50000, 60000))),
        'test': torch.utils.data.DataLoader(dataset['testingValues'], batch_size=batches, shuffle=False)
    }

    # model ( simple linear regression model )
    model = onelinear(totalDimensions, numberOfClasses).to(device)

    # loss - cross entropy loss that penalizes wrong guesses
    loss = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')

    lrates = [0.01, 0.001] # learning rate - vector points in direction where loss is greatest, move in opposite direction by x learning rate

    best_hyperparameter = None
    weights_chosen = None
    bestmeasure = None

    for lr in lrates:  # try a few learning rates

        print('\n\n\n###################NEW RUN##################')
        print('############################################')
        print('############################################')

        # optimizer here, because of lr,
        # applies the computed gradients to change the trainable parameters of the model.
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)  # which parameters to optimize during training?
        # SGD utilizes that we do not have to look at the entire dataset, but just at a batch, making training faster
        # momentum basically utilizes the last optimizationss to ensure they all roughly go in the same direction

        # train on train and eval on val data
        best_epoch, best_perfmeasure, bestweights = train_modelcv(dataloader_cvtrain=dataloaders['train'],
                                                                  dataloader_cvtest=dataloaders['val'], model=model,
                                                                  criterion=loss, optimizer=optimizer, scheduler=None,
                                                                  num_epochs=maxnumepochs, device=device)

        if best_hyperparameter is None:
            best_hyperparameter = lr
            weights_chosen = bestweights
            bestmeasure = best_perfmeasure
        elif best_perfmeasure > bestmeasure:
            best_hyperparameter = lr
            weights_chosen = bestweights
            bestmeasure = best_perfmeasure

    # end of for loop over hyperparameters here!
    model.load_state_dict(weights_chosen)

    accuracy, _ = evaluate(model=model, dataloader=dataloaders['test'], criterion=None, device=device) # evaluate best values again

    print('accuracy val', bestmeasure.item(), 'accuracy test', accuracy.item())


if __name__ == '__main__':
    run()



