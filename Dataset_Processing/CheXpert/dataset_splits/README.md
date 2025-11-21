# CheXpert Dataset Splits

## Moco

### Backbone Pretraining
File paths and Labels:
* `1_final_project_updated_names_train_moco.csv`
* `1_final_project_updated_names_test_moco.csv`

Images:
* Corresponds to dataset: `CheXpert_reduced_dataset_split_v3.zip`
  * Contrastive Representation learning; no labels are used
  * Frontal and Lateral images used; should be good for representation learning

### Testing Evaluation 
File paths and Labels:
* `2_final_project_updated_names_train_linear.csv`
* `2_final_project_updated_names_test_linear.csv`

Images:
* Corresponds to dataset: `CheXpert_reduced_dataset_split_v3.zip`
  * Uses a **subset** of the data available for general representation learning
  * `Pneumonia` label is used (0, 1) for classification

## Transfer Learning

### ResNet50
**Pneumonia Classification (2 Classes)** 
File paths and Labels:
* `0_final_project_updated_names_train_transfer_binary.csv`
* `0_final_project_updated_names_val_transfer_binary.csv`
* `0_final_project_updated_names_test_transfer_binary.csv`

Images:
* Corresponds to dataset: `CheXpert_reduced_dataset_split_transfer_binary.zip`

**Multi-class (14 classes)** - Corresponds to dataset:
File paths and Labels:
* `3_final_project_updated_names_train_transfer.csv`
* `3_final_project_updated_names_val_transfer.csv`
* `3_final_project_updated_names_test_transfer.csv`
  
Images:
* `CheXpert_reduced_dataset_split_multiclass.zip`
  * `Label`: (14 classes) label is used for classification
  * Only data rows with 1 label were used to construct this dataset
  * Frontal orientation images only

```python
{'No Finding': 0,
 'Enlarged Cardiomediastinum': 1,
 'Cardiomegaly': 2,
 'Lung Opacity': 3,
 'Lung Lesion': 4,
 'Edema': 5,
 'Consolidation': 6,
 'Pneumonia': 7,
 'Atelectasis': 8,
 'Pneumothorax': 9,
 'Pleural Effusion': 10,
 'Pleural Other': 11,
 'Fracture': 12,
 'Support Devices': 13}
```