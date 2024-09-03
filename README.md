# Less is More: Robust Caries RGB Images Learning with Core Data Selection
This is the code for _Less is More: Robust Caries RGB Images Learning with Core Data Selection_ submitted to ICASSP 2025.

## Abstract
Caries detection based on RGB images is widely applied in caries recognition due to its convenience and low cost. However, the blurriness of captured data and label errors in manual annotation impact the model's ability to the degree of caries prevalence. To address this problem, we propose Robust Contrastive Learning with Core Data Selection (CoreRCL) to improve the predictive performance of caries classification models. Instead of fine-tuning the model backbone network structure, CoreRCL focuses on improving robustness to label errors in caries classification from a novel perspective by identifying core data that is highly relevant to the caries category. Specifically, CoreRCL utilizes Jensen-Shannon Divergence to calculate the average mutual information between data samples and caries category cluster centers, selecting core data to mitigate the impact of label errors from low-quality data on model performance. Furthermore, we design inter-category contrastive learning, which enhances the model’s ability to distinguish between different caries categories by increasing the feature representation between samples of different categories. Extensive experiments on caries RGB image datasets demonstrate that CoreRCL significantly outperforms other core data selection methods in predictive performance. More excitingly, CoreRCL achieves superior predictive performance using only 50\% of the core data compared to state-of-the-art caries detection methods using the entire dataset.

## Requirements
numpy==1.22\
requests==2.25.1\
scipy==1.5.3\
torch==1.10.1\
torchvision==0.11.2
