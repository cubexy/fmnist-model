# https://github.com/pytorch/examples/blob/master/mnist/main.py

import torch
import torch.optim as optim

from torchvision import datasets, transforms

import torch.utils

import numpy as np

torch.manual_seed(3)


def train_epoch(model, trainloader, criterion, device, optimizer):
    model.train()  # IMPORTANT!!!

    losses = []
    for batch_idx, data in enumerate(trainloader):
        inputs = data[0].to(device)
        labels = data[1].to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()  # reset accumulated gradients
        loss.backward()  # compute new gradients
        optimizer.step()  # apply new gradients to change model parameters

    return losses


def evaluate(model, dataloader, criterion, device):
    model.eval()  # IMPORTANT!!!

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
            cpuout = outputs.to('cpu')
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
    best_measure = 0
    best_epoch = -1

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        losses = train_epoch(model, dataloader_cvtrain, criterion, device, optimizer)
        # scheduler.step()
        measure, _ = evaluate(model, dataloader_cvtest, criterion=None, device=device)

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

        self.bias = torch.nn.Parameter(data=torch.zeros(numout), requires_grad=True)
        self.w = torch.nn.Parameter(data=torch.randn((dims, numout)),
                                    requires_grad=True)  # random init shape must be (dims,numout), requires_grad to True

    def forward(self, x):
        # compute the prediction over batched input x

        # print(x.size()) # (batchsize,dims)
        # print(self.w.size())

        v = x.view((-1, 28 * 28))  # flatten the image to (batchsize,dims), -1 allows to guess the number of elements
        y = self.bias + torch.mm(v, self.w)

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
    batchsize = 32
    maxnumepochs = 3

    # device=torch.device("cuda:0")
    device = torch.device("mps") # optimize for apple silicon (swap for "cpu" otherwise)

    datatransforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    ds = {
        'trainval': datasets.FashionMNIST('./data', train=True, download=True, transform=datatransforms),
        'test': datasets.FashionMNIST('./data', train=False, download=True, transform=datatransforms)
    }
    numcl = 10
    indims = 784

    dataloaders = {

        'train': torch.utils.data.DataLoader(ds['trainval'], batch_size=batchsize, shuffle=False,
                                             sampler=torch.utils.data.sampler.SubsetRandomSampler(np.arange(50000))),

        'val': torch.utils.data.DataLoader(ds['trainval'], batch_size=batchsize, shuffle=False,
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               np.arange(50000, 60000))),

        'test': torch.utils.data.DataLoader(ds['test'], batch_size=batchsize, shuffle=False)
    }

    # model
    model = onelinear(indims, numcl).to(device)

    # loss
    loss = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')

    lrates = [0.01, 0.001]

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

    accuracy, _ = evaluate(model=model, dataloader=dataloaders['test'], criterion=None, device=device)

    print('accuracy val', bestmeasure.item(), 'accuracy test', accuracy.item())


if __name__ == '__main__':
    run()



