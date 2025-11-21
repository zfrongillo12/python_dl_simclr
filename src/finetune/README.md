# Fine Tune
Subdirectory contains code required for fine tuning the pre-trained backbone on a smaller dataset.

Main execution function: `src\finetune_resnet.py`

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