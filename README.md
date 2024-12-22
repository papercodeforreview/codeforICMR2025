# Core Inter-Category Contrastive Learning for Enhancing Robustness of Caries Classification

This is the code for *Core Inter-Category Contrastive Learning for Enhancing Robustness of Caries Classification* submitted to ICME 2025. 

## Core Data Selection

The specific implementation details are in the deepcore/methods/jsd.py

## SupervisedContrastiveLoss

The specific implementation details are in the main_cl.py, with the loss function being:

```
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, projections, labels):
        device = projections.device
        batch_size = projections.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        similarity_matrix = torch.matmul(projections, projections.T) / self.temperature

        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        exp_logits = torch.exp(logits) * (1 - torch.eye(batch_size).to(device))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -mean_log_prob_pos.mean()
        return loss
```



## RUN

### 1. Requirements

```
pip install -r requirements.txt
```

### 2. Ensure that the dataset folder location is correct in deepcore/datasets/teeth.py

### 3. Getting Started

 For ResNet18 experiments :

```
python main_cl.py --fraction 0.01 --dataset Teeth --model Res18 --selection Jsd --num_exp 50 --epochs 200 --min_lr 0  --lr 0.01 --weight_decay 5e-4 --batch-size 256 --scheduler LambdaLR  --data_update_epochs 50 --log ./logs/new.log 
```

