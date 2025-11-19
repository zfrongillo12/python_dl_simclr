# Setting a constant for the Resnet Image Size (224 is expected for ResNet50)
resnet_image_size = 224

# Hyperparameters for MoCo Pretraining
queue_size = 65536
momentum_update_rate = 0.999
softmax_temperature = 0.07

# Optimizer settings (I think the Moco paper uses SGD with momentum)
learning_rate_moco = 0.03
weight_decay_moco = 1e-4
momentum_sgd = 0.9

# Training constants
epochs = 150
batch_size = 256

# Device setting
device = "cuda"

# CSV paths (PATHS IN MY DRIVE)
train_csv_path = "/content/drive/MyDrive/Data_Reduced/final_project_updated_names_train.csv"
val_csv_path   = "/content/drive/MyDrive/Data_Reduced/final_project_updated_names_val.csv"
test_csv_path  = "/content/drive/MyDrive/Data_Reduced/final_project_updated_names_test.csv"