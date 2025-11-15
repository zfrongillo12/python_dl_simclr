# Readme: Data Processing

### 1. Download dataset locally from: https://www.kaggle.com/datasets/mimsadiislam/chexpert?resource=download

```
mkdir CheXpert_Processing
unzip CheXpert-v1.0-small.zip
```

Which will have 2 metadata files that organize the provided datasets:
* train.csv
* valid.csv

### 2. Run this Jupyter notebook to see basic information about the provided datasets, and to create the train/validation/test splits (as CSV files):
* `1_Chexpert_basic_data_info.ipynb`

Will produce 3 files to represent the dataset splits:
* project_train.csv
* project_val.csv
* project_test.csv

### 3. Run this Jupyter notebook in the same relative directory as your unzipped data (e.g. `./CheXpert_Processing`) to create the datasets
* `2_Create_dataset_directories.ipynb`

**Note:** the file names are often duplicate (e.g. `view1_frontal.jpg)`, across the paths (due to dependency on unique patient ID/s study in paths instead)
* Contains: New file name processing
  * e.g.) From: `CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg`
  * To: `patient00001_study1_view1_frontal.jpg`

Example output:
```
=== Processing train ===
Copying 14032 images...
Copying train images: 100%|██████████| 14032/14032 [00:21<00:00, 656.73it/s]
[INFO] Saved updated CSV to final_project_updated_names_train.csv

=== Processing val ===
Copying 2005 images...
Copying val images: 100%|██████████| 2005/2005 [00:02<00:00, 744.95it/s]
[INFO] Saved updated CSV to final_project_updated_names_val.csv

=== Processing test ===
Copying 4010 images...
Copying test images: 100%|██████████| 4010/4010 [00:05<00:00, 748.80it/s]
[INFO] Saved updated CSV to final_project_updated_names_test.csv

Done! Images copied into train/val/test directories.
```

Will produce 3 files with the updated file names:
* final_project_updated_names_train.csv
* final_project_updated_names_val.csv
* final_project_updated_names_test.csv