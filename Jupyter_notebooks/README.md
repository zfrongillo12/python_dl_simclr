# ReadMe for Jupyter Notebooks

## Directory Explanation
### Baseline

For the MoCo baseline. ResNet and ViT

### Update

For the novel implementations / modifications to the Hybrid ViT architecture

### generate_report_visuals

Generate report visuals based on input training data

### template

Base transfer learning setup/code.

## Instructions for Setup (Copied from root README)
You will need to modify the first few lines of the notebooks to direct to the local file dependencies defined below.

1. Specify your ROOT where you're storing files (note: `/content/drive/MyDrive/` is the same base mount directory on all Google Collaboratory notebooks):
   * e.g.) `FP_ROOT="/content/drive/MyDrive/Colab Notebooks/7_Py_DL/FP/"`
2. Create `./Data` at `$FP_ROOT`. 
   * Copy the zipped image dataset (e.g. zipped and organized into train/val/test subdirectories) and the CSV labels located in `./Dataset_Processing/CheXpert/dataset_splits`
   * `$FP_ROOT/Data/`: flattened structure, copy the zipped data files and the csv labels from `./Dataset_processing` (or your own custom)
3. Create `./src` at `$FP_ROOT`. Copy/paste the files and directory structure from Github: 
   * `$FP_ROOT/src/`: copy/paste the directory structure from Github:  `./src/` 
4. Create `./artifacts` at `$FP_ROOT`.
   * `$FP_ROOT/artifacts/`: outputs will be written here 

When executing on Collab, will need to agree to allow the application to review your files.
