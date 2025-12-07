# EN.705.643 | Python DL - MoCo Contrastive Learning & Transfer Learning for Chest X-Rays

### Authors: Zoe Frongillo, Scott Hansen

## Objective / Hypothesis
This project will investigate how contrastive self-supervised pre-training (through the MoCo framework)can effectively learn feature representations for medical images, in the form of Chest X-rays.  We will use transfer learning on different architectures to evaluate medical X-Ray classification.

The MoCo-v2 framework will be used to pretrain backbone models (ResNet-50 and ViT-Base) on a large unlabeled dataset of chest X-ray data (Stanford's CheXpert X-Ray dataset). These pretrained models will then be transferred and fine-tuned on a different, labeled X-Ray dataset (the Stanford CheXpert dataset, NIH, Indiana University) for chest pathology classification. We will compare three pretraining strategies (training from scratch, supervised ImageNet pretraining, and SimCLR self-supervised pretraining) to determine which of these methods yields higher classification performance when adapting to a new labeled dataset.

## Novel Improvement Ideas for MoCo
### Feature 1: Hybrid MoCo Architecture
For the ViT backbone architecture for the MoCo framework, replace the standard linear embeddings with a **ConvStem (CNN)** to process the patches of images.  Further, the standard ViT (PyTorch: `vit_b_16`) takes non-overlapping 166x16 patches, flattens each patch, and sends it through a linear layer.  This standard makes the ViT has no convolutional inductive bias, severely limiting the feature extraction before attention.  Thus, this architecture relies solely on attention to learn fine image information such as local structures and textures; necessitating tons of data to concretely learn.

**Why we think this is important:**
This is problematic in medical imaging because it leads to poor low-level feature extraction, is data inefficient (where quality labeled datasets are scarce), sensitive to noise, and does not retain a spatial hierarchy.

Our implementation of ConvStem uses three convolutions (3x3 conv with stride 2) to extract local features (e.g. edges, textures) to improve local inductive bias, learn stable representations early on, and is more sensitive for the use case of medical images because Chest X-Rays have more subtle textures, fine-grained details, and have spatially correlated structures.


### Feature 2: For finetuning and applying Transfer Learning (ViT based models) for classification applications on smaller, labeled data subsets
After obtaining the patch-level features, we have appended a small network (like a small convolutional layer or a small feedforward network) that could act as a patch-scoring head (assigning each patch a learnable scalar importance score). So, for each augmented view of the image, we would:
1. Rank all patch tokens (by their learned scores).
2. Select the “top-K” highest-scoring patch tokens (where K can be a hyperparameter that can be tuned).
3. Combine those selected patches (through averaging, concatenation, or another operation) to produce a final embedding for that view.
4. Feed this reduced embedding into the standard SimCLR contrastive loss technique.
5. The scoring head would learn the p

**Why we think this is important:**
* The top-K patch selection strategy could help focus the contrastive learning on the most informative anatomical regions (suppressing noise and improving cross-domain generalizability, which is a significant concern in the domain of medical imaging).
* The heat-map visualization of the selected important patches can provide a highly useful explainability component, which is especially important in medical-imaging contexts where clinicians need insight into why and how the model made its decision.