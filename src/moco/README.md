# MoCo

* To reduce SimCLR's dependence on large batch sizes, will use MoCo framework for the contrastive learning.
* **MoCo**: Self-supervised pretraining strategy to learn visual representation of images *without* labels.
  * Pre-text task:
    * Learn to differentiate positive pairs (two augmented views of the same image)
    * Versus negative pairs (the other images) 

## About the Architecture

### 2 Encoders:
Purpose: Extract feature representations form the images
* Map the two augmented views of the same image to similar, high level semantic features

1. `Encoder_q`: Process the "query" image (augmented view 1)
   1. Primary network for training; weights updated during backprop
2. `Encoder_k`: Process the "key" image (augmented view 2)
   1. Weights are not backproped; it is updated with momentum
   2. Slow moving target encoder to stabilize training

### Projection Heads
The output feature vectors are sent to `mlp_q` and `mlp_k` as projection heads
* Maps encoder features (and key features) to the contrastive space
* Projection allows the encoder to keep general features, while the MLP pushes contrastive-specific structure into its own space
* After training, for the classification downstream tasks, the MLPs are no longer needed (only use the encoder)

### MoCo Queue

* Required so the projection head can compare the query projections against the many negative keys without the dependency on large GPU batch size
  * Serves as a memory bank of past projected features; for stability
  * Requirement: SimCLR showed that you need 4k–8k negative pairs per step for good results
    * Need many, else the model will collapse or learn poor representations

### Resources
* Momentum Contrast for Unsupervised Visual Representation Learning (MoCo V1): https://arxiv.org/pdf/1911.05722
* Improved Baselines with Momentum Contrastive Learning (MoCo V2): https://arxiv.org/pdf/2003.04297
* Examples in Pytorch:
  * https://www.analyticsvidhya.com/blog/2020/08/moco-v2-in-pytorch/ 
  * https://github.com/facebookresearch/moco/blob/main/main_moco.py
  * https://github.com/facebookresearch/moco/blob/main/moco/builder.py

## DataLoader Expectations
* Expects the folder to be formatted as such:

```
├── root_dataset
   ├── train
   │   ├── image.img
   │   ├── image.img
   │   └── image.img
   ├── val
   │   ├── image.img
   │   ├── image.img
   ├── test
   │   ├── image.img
   │   ├── image.img
   └── 
```

* Description for the dataset is a CSV 
  * 1 CSV per data partition - train.csv, val.csv, test.csv
  * Column `Path` - file path to the data sample (relative directory should start from **./train;** e.g. should NOT include ./train for train)
    * Please see for example CSVs at this directory: `python_dl_simclr\Dataset_Processing\NIH_Chest_XR_Pneumonia\dataset_splits`
  * Column per dataset label
    * e.g. `Pneumonia` with the possible values 0,1

## MoCo Testing Strategy

### Linear Evaluation

* Consistent with MoCo v2 paper (and other SSL papers)

**Overview:**
1. Freeze the pretrained backbone (no gradient updates)
2. Add a single linear layer for classification.
3. Train only the linear layer on your labeled train set.
4. Measure accuracy on your test set.

**Objective:**
* Tells us how good the representation is *without* fine-tuning
* Comparable to other SSL literature

#### MoCo Pretraining
* No labels, no classes
* The dataset is all un-labeled images, where there are positive pairs (im_q, im_k), and all other images act as negatives via the queue

#### Linear Evaluation using MLP
* Need to send in data that has the labels needed for the downstream image classification task

---
## About the Files

### Baseline ResNet-50 & VIT-16
* train_moco.py
* test_moco.py

Classification dataset (shared)
* (Copied in src/finetune; as needed by Google Collab)
* classification_dataset.py

### ViT Hybrid
* train_moco_vit_hybrid.py
* test_moco_vit_hybrid.py
