# Deep Geometric neural network using rigid and non-rigid transformations for human action recognition

**Prerequisites**

install [Pytorch 1.7.1](https://pytorch.org/get-started/locally/) (If there are any versions presenting difference in performance please feel free to report the event to us) with torchvision that is compatible with your hardware:

Example for cuda 11.0:
```bash
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
```
install the packages for the code generation, interpolation and the custom geometric layer
```bash
pip install torchgeometry scipy tqdm
```

**KShapeNet pretraining**
Please follow the step 1 & 2 in Readme.html to prepare the NTU dataset for KShapeNet pretraining.

The model pretrained using single-person actions (94 out of 120 in total) can be downloaded from: [Google drive](https://drive.google.com/file/d/11MJgkg3byfDYgphAtcjrNss6QLBeHSjK/view?usp=drive_link). To use it, download the checkpoint and put it to *models/checkponts/*.

**Model finetuning**
1. Run data preprocessing to cut the 3D skeleton sequences into fixed-length subsequences:
```bash
python preprocess.py # modify hyperparameters in the script
```
2.[Option 1] Run multi-fold cross-validation (without test data) based on *split_name_file*, which contains per-fold train/valid splits:
```bash
python model.py
```
[Option 2] Run multi-fold cross-validation with train/valid/test splits based on *split_name_file*, which contains per-fold train/valid/test splits:
```bash
python model_wtest.py
```

⚠️ *Remark: Command-line argument parsing for customized configuration has not been implemented in this version. To customize settings, please edit the script file directly in its **Arguments** section.*

