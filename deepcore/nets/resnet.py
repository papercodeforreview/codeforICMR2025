import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision
from .nets_utils import EmbeddingRecorder

# vmodel=torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1")

vmodel=torchvision.models.resnet18(weights="IMAGENET1K_V1")

class RseNet18(nn.Module):
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



def Res18():
    return RseNet18()
