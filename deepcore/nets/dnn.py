import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision
from .nets_utils import EmbeddingRecorder

vmodel=torchvision.models.resnet18(weights="IMAGENET1K_V1")
# vmodel=torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1")

class DNNs(nn.Module):
    def __init__(self,  record_embedding: bool = False):
        super().__init__()
        self.encoder= vmodel
        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.decoder= nn.Linear(1000, 3)
        self.projection = nn.Linear(1000, 128)
        
    def forward(self, x):
        features = self.encoder(x)
        features - self.embedding_recorder(features)
        projection = self.projection(features)
        projection = torch.nn.functional.normalize(projection, dim=1)
        logits = self.decoder(features)
        return logits,projection
    

    def get_last_layer(self):
        return self.decoder

class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet18', head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()

        self.encoder = ViTs()
        if head == 'linear':
            self.head = nn.Linear(1000, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(1000, 1000),
                nn.ReLU(inplace=True),
                nn.Linear(1000, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat


class SupCEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='resnet18', num_classes=3):
        super(SupCEResNet, self).__init__()
        self.encoder = ViTs()
        self.fc = nn.Linear(1000, num_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='resnet18', num_classes=3):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(128, num_classes)

    def forward(self, features):
        return self.fc(features)



def DNN():
    return DNNs()

def Classifier():
    return LinearClassifier()