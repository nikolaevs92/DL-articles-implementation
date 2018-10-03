from torch import nn
from functools import reduce


class InOneDemensionReshape(nn.Module):
    '''
    Reshape tensor to one demenshion tensor
    '''
    def forward(self, x):
        return x.reshape((x.shape[0], reduce(lambda d1, d2: d1 * d2, x.shape[1:])))


def alexnet(n_classes):
    '''
    AlexNet implementation:
    https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11,11), stride=4, padding=2, groups=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),
        nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5,5), stride=1, padding=2, groups=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),
        nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3,3), stride=1, padding=1, groups=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3,3), stride=1, padding=1, groups=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3,3), stride=1, padding=1, groups=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),
        InOneDemensionReshape(),
        nn.Dropout(),
        nn.Linear(256*6*6, 4096),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(4096, n_classes)
    )

def vggA(n_classes):
    '''
    VGG16 implementation:
    https://arxiv.org/pdf/1409.1556
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0),
        
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0),
        
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0),
        
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0),
        
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0),
        
        InOneDemensionReshape(),
        
        nn.Dropout(),
        nn.Linear(512*7*7, 4096),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(4096, 1024),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(1024, n_classes)
    )

def vggA_LRN(n_classes):
    '''
    VGG16 implementation:
    https://arxiv.org/pdf/1409.1556
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.LocalResponseNorm(),
        nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),
        
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),
        
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),
        
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),
        
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),
        
        InOneDemensionReshape(),
        
        nn.Dropout(),
        nn.Linear(512*7*7, 4096),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(4096, 1024),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(1024, n_classes)
    )


def vggB(n_classes):
    '''
    VGG16 implementation:
    https://arxiv.org/pdf/1409.1556
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),
        
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),
        
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),
        
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),
        
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),
        
        InOneDemensionReshape(),
        
        nn.Dropout(),
        nn.Linear(512*7*7, 4096),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(4096, 1024),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(1024, n_classes)
    )


def vggC(n_classes):
    '''
    VGG16 implementation:
    https://arxiv.org/pdf/1409.1556
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),
        
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),
        
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1,1), stride=1, padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),
        
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1,1), stride=1, padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),
        
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1,1), stride=1, padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),
        
        InOneDemensionReshape(),
        
        nn.Dropout(),
        nn.Linear(512*7*7, 4096),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(4096, 1024),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(1024, n_classes)
    )


def vggD(n_classes):
    '''
    VGG16 implementation:
    https://arxiv.org/pdf/1409.1556
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),
        
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),
        
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),
        
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),
        
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),
        
        InOneDemensionReshape(),
        
        nn.Dropout(),
        nn.Linear(512*7*7, 4096),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(4096, 1024),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(1024, n_classes)
    )


def vggE(n_classes):
    '''
    VGG16 implementation:
    https://arxiv.org/pdf/1409.1556
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),
        
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),
        
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),
        
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),
        
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),
        
        InOneDemensionReshape(),
        
        nn.Dropout(),
        nn.Linear(512*7*7, 4096),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(4096, 1024),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(1024, n_classes)
    )
