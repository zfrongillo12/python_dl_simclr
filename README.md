# EN.705.643 | Python DL SimCLR Project

### Authors: Zoe Frongillo, Scott Hansen

### Objective / Hypothesis
This project will investigate how contrastive self-supervised pretraining (through the SimCLR framework) affects transfer learning performance for medical X-Ray classification.

The SimCLR framework will be used to pretrain backbone models (ResNet-50 and ViT-Base) on a large unlabeled dataset of chest X-ray data (Stanford's CheXpert X-Ray dataset). These pretrained models will then be transferred and fine-tuned on a different, labeled X-Ray dataset (the Stanford CheXpert dataset) for chest pathology classification. We will compare three pretraining strategies (training from scratch, supervised ImageNet pretraining, and SimCLR self-supervised pretraining) to determine which of these methods yields higher classification performance when adapting to a new labeled dataset.