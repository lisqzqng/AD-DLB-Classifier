# KShapeNet: Deep Geometric neural network using rigid and non-rigid transformations for human action recognition
 
**Prerequisites**

install [Pytorch 1.7.1](https://pytorch.org/get-started/locally/") (If there are any versions presenting difference in performance please feel free to report the event to us) with torchvision that is compatible with your hardware: "

Example for cuda 11.0:

	conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch 

install the packages for the code generation, interpolation and the custom geometric layer
	
	pip install torchgeometry scipy tqdm

	
**1- Dataset Download:**
Download the dataset (NTU60 and/or NTU120) from the [Roselabs](http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp) website and extract the files under the ./data/NTU-RGB+D(120) folder. 

The data folder tree should look like the following.

	├───NTU-RGB+D
	│   └───nturgb+d_skeletons /* .skeleton files for NTU60 under this folder
	├───NTU-RGB+D120
	│   └───nturgb+d_skeletons /* .skeleton files for NTU120 under this folder
	├───nturgb_d
	│   ├───xsub
	│   └───xview
	└───nturgb_d120
		├───xsetup
		└───xsub

**2- Data generation from .skeleton files**
	
Generate the main numpy files used for the data generation (This code goes to the creators of the [AS-GCN](https://github.com/limaosen0/AS-GCN) network). **Please make sure the data conforms to the previously presented tree**

For NTU60:

	cd data_gen 
	#The script can be ran without any parameters as it points to the already created folders/files in the repo:
	python ntu_gen_preprocess60.py --data_path [PATH to .Skeletonfiles] --ignore_sample_path [PATH to samples_with_missing_skeletons.txt] --out_folder [Output path]

For NTU120:
	
	cd data_gen
	#The script can be ran without any parameters as it points to the already created folders/files in the repo:
	python ntu_gen_preprocess60.py --data_path [PATH to .Skeletonfiles] --ignore_sample_path [PATH to samples_with_missing_skeletons.txt] --out_folder [Output path]

Result example for NTU60 xview:

	│   └───xview
	│           .gitkeep
	│           train_data_joint_pad.npy
	│           train_label.pkl
	│           val_data_joint_pad.npy
	│           val_label.pkl

**2- Data generation from *joint_pad.npy files, make sure you're still in the data_gen folder**

in both scripts: 

The **--num_frames** parameter sets the number of frames to read from default data and **--num_frames_interp** sets the number of frames to which it will interpolate the data.

The **--save_50x3** parameter specifies whether to save extra data with shape N_samples,N_frames,N_joints(50),N_dims(3), the default shape is N_samples,N_frames,150(50x3)

The **--save_one_body** parameter specifies whether to save each boby in the actions in a seperate .npy file.

For NTU60:

	python	generate_2bodies.py --prot [xsub,xview] --save_raw [True/False raw data save] --num_frames [n] --num_frames_interp [m] --save_one_body [True/False] --save_50x3 [True/False]

for NTU120:

	python	generate_2bodies.py --prot [xsub,xview] --save_raw [True/False raw data save] --num_frames [n] --num_frames_interp [m] --save_one_body [True/False] --save_50x3 [True/False]

Result example after generation NTU60 xview interpolated files:

    │   └───xview
    │           .gitkeep
    │           test_xview_interp100.npy
    │           train_data_joint_pad.npy
    │           train_label.pkl
    │           train_xview_interp100.npy
    │           val_data_joint_pad.npy
    │           val_label.pkl

**3- Model training and testing, (please make sure you're in the root file of the repository)**

There are two options for training the models: Both NTU60 and NTU120 scripts share the same parameters


**3.1: One run script**

	python onerunNet60.py --prot [xview,xsub] --num_epoch [EPOCHS] --layer_name [RigidTransform, RigidTransformInit, NonRigidTransform, NonRigidTransformInit] --cast [log_sref,log_0refseq,PT] --learning_rate [lr] --batch_size [size] --batch_size_test [size'] --save [True,False]
	
	python onerunNet120.py --prot [xsetup,xsub] --num_epoch [EPOCHS] --layer_name [RigidTransform, RigidTransformInit, NonRigidTransform, NonRigidTransformInit] --cast [log_sref,log_0refseq,PT] --learning_rate [lr] --batch_size [size] --batch_size_test [size'] --save [True,False]
	
**3.2: Multiple runs script: This script is created to increase the chances of saving the best model for each layer/cast modality**

This is mainly due to the large initialization space of the rotation matrices used as kernels in the custom layer. it will either save each and every model for each run or it will save the best model for every cast/layer modality. In our testing we opted for the experiment having 10 runs per layer modality from which we extract the best run.

	python network_avg_NTU60.py --prot [xview,xsub] --num_runs--num_epoch [EPOCHS] --cast [log_sref,log_0refseq,PT] --learning_rate [lr] --batch_size [size] --batch_size_test [size'] --save_max [True,False]

	python network_avg_NTU120.py --prot [xsetup,xsub] --num_runs--num_epoch [EPOCHS] --cast [log_sref,log_0refseq,PT] --learning_rate [lr] --batch_size [size] --batch_size_test [size'] --save_max [True,False]
	
**4- testing saved models**
	
Please make sure to specify the **--labels_path** if you specify the **--data_path** parameter else it will test the default generated data along the specified protocol

	python testNetwork.py --checkpoint [path to .pth saved model] --prot [xsub,xview,xsetup] --use_cuda [True,False] --data_path [path .npy test file] --labels_path[path .pkl labels file]
	
**5- Sample results**	

<div class="center">

<div id="table:transformationLayer">

| Dataset               | NTU-RGB+D |          | NTU-RGB+D120 |          |
|:----------------------|:----------|:---------|:-------------|:---------|
| Protocol              | CS        | CV       | CS           | Cset     |
| Rigid Matrix based    | 97.0      | 97.1     | 90.2         | 85.9     |
| Rigid Angle based     | 96.9      | 96.3     | 89.1         | 84.9     |
| NonRigid Matrix based | 96.8      | 96.9     | 90.6         | 84.3     |
| NonRigid Angle based  | **97.0**  | **98.5** | **90.6**     | **86.7** |

Table: Comparison of different variants of the transformation layer (%
accuracy).

</div>

</div>