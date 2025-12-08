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

* Ran our pre-training and transfer learning using Google Collab (L4 and A100 High RAM GPUs)

To execute our experiments, we used Jupyter notebooks that can be found in `./Jupyter_notebooks`

**Collab File Structure Setup:**

You will need to modify the first few lines of the notebooks to direct to the local file dependencies defined below.

1. Specify your ROOT where you're storing files (note: `/content/drive/MyDrive/` is the same base mount directory on all Google Collaboratory notebooks):
   * e.g.) `FP_ROOT="/content/drive/MyDrive/Colab Notebooks/7_Py_DL/FP/"`
2. Create `./Data` at `$FP_ROOT`. 
   * Copy the zipped image dataset (e.g. zipped and organized into train/val/test subdirectories) and the CSV labels located in `./Dataset_Processing/CheXpert/dataset_splits`
   * `$FP_ROOT/Data/`: flattened structure; should contain zipped image data and the CSV labels 
3. Create `./src` at `$FP_ROOT`. Copy/paste the files and directory structure from Github: 
   * `$FP_ROOT/src/`: copy/paste the directory structure from Github:  `./src/` 
4. Create `./artifacts` at `$FP_ROOT`.
   * `$FP_ROOT/artifacts/`: outputs will be written here 

When executing on Collab, will need to agree to allow the application to review your files.
