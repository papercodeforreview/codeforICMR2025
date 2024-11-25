from .. import nets
from .earlytrain import EarlyTrain
from .coresetmethod import CoresetMethod
import torch, time
import numpy as np
from ..nets.nets_utils import MyDataParallel
import torch.nn.functional as F
from datetime import datetime
import os
import math
from copy import deepcopy


class Jsd(CoresetMethod):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, balance=True, replace=False, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed)
        self.dst_train = deepcopy(dst_train)
        self.balance = balance
        self.replace = replace
        self.n_train = len(dst_train)
        self.coreset_size = round(self.n_train * fraction)
        self.balance = balance
        self.torchvision_pretrain = True if self.args.dataset == "ImageNet" else False
        self.specific_model = None
        self.feature_extractor = self.args.model
        self.num_classes = 3

    def select(self, **kwargs):
        self.train_indx = np.arange(self.n_train)
        if self.args.dataset == "ImageNet":
            model = nets.__dict__[self.feature_extractor](
                self.args.channel, self.num_classes, pretrained=self.torchvision_pretrain,
                im_size=(224, 224) if self.torchvision_pretrain else self.args.im_size).to(self.args.device)
        else:
            model = nets.__dict__[self.feature_extractor]().to(self.args.device)
            # checkpoint = torch.load(self.args.resume, map_location=self.args.device)
            # # Loading model state_dict
            # model.load_state_dict(checkpoint["state_dict"],strict=True)

        model = nets.nets_utils.MyDataParallel(model).cuda()
        model.eval()
        batch_loader = torch.utils.data.DataLoader(
            self.dst_train, batch_size=self.args.selection_batch, num_workers=self.args.workers,
            shuffle=False) # shuffle should be False to make sure index are the same!
        sample_num = self.n_train
        features = torch.zeros([self.n_train, self.num_classes])
        with torch.no_grad():
            for i, (input, target) in enumerate(batch_loader):
                if i % self.args.print_freq == 0:
                    print('| Current Sample [%3d/%3d]' % (i * self.args.selection_batch, sample_num))
                feature = model(input.to(self.args.device))[0].cpu()
                features[i * self.args.selection_batch:min((i + 1) * self.args.selection_batch, sample_num)] = feature

        prots = np.zeros((self.num_classes, features.shape[-1]))
        for c in range(self.num_classes):
            c_indx = self.train_indx[self.dst_train.targets == c]
            prots[c] = np.mean(features[c_indx].detach().numpy().squeeze(), axis=0, keepdims=True)

        prots_for_each_example = np.zeros(shape=(features.shape[0], prots.shape[-1]))
        for c in range(self.num_classes):
            c_indx = self.train_indx[self.dst_train.targets == c]
            prots_for_each_example[c_indx, :] = prots[c]
        # distance = np.linalg.norm(features - prots_for_each_example, axis=1)
        features = F.softmax(features, dim=1)
        prots_for_each_example = torch.from_numpy(prots_for_each_example).float()  
        prots_for_each_example = F.softmax(prots_for_each_example, dim=1)

       
        distance_tensor =0.5* F.kl_div(features.log(), prots_for_each_example, reduction='none').sum(dim=1)+0.5* F.kl_div(prots_for_each_example.log(), features, reduction='none').sum(dim=1)
        distance = distance_tensor.cpu().numpy()
     

        if not self.balance:
            raise NotImplementedError("Jsd Coreset only support class-balanced selection")
        else:
            top_examples = np.array([], dtype=np.int64)
            for c in range(self.num_classes):
                c_indx = self.train_indx[self.dst_train.targets == c]
                budget = round(self.fraction * len(c_indx))
                top_examples = np.append(top_examples, c_indx[np.argsort(distance[c_indx])[::-1][:budget]])

        return {"indices": top_examples, "scores": distance}



   

    