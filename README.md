# TGD
Leveraging Topological Guidance for Improved Knowledge Distillation, GRaM workshop @ ICML

This repository is of our proposed method, TGD, where we use topological features in knowledge distillation (KD) to train and evaluate a light weight model for Cifar10 and CINIC10 dataset.

# Data

The CIFAR-10 data can be downloaded at [link](https://www.dropbox.com/scl/fo/uhlraru1x1azyuhe0x2q8/AFs1w7yLsqE6NhCJBzAC5Z4?rlkey=hu2d76xzvolw3ym6kegks6eqe&dl=0). You would have to unzip the cifar10.zip file inorder to access the data. The data follows the following file format:

[Data]
```
cifar10
├──test # Contains the Org testing images
├──test_PIc # Contains the Column wise PI testing images
├──test_PIr # Contains the Row wise PI testing images
├──train # Contains the Org training images
├──train_PIc # Contains the Column wise PI training images
├──train_PIr # Contains the Row wise PI training images
├──labels.txt # File containing the list of labels
├──test.csv 
└──train_csv 
```

If you want to generate the PI images for some specific data, you can use the following GenPI_image.ipynb file.  

## PI Image Generarion

The GenPI_Images.ipynb File was used to generate the PI images. To generate the PI image we follow the following steps:
1) Normalize the images in range [0,1]
2) The code mentioned above will generate the row wise PI images, in order to generate the columns wise PI images you would need to uncomment the following line:
```python=
x_data.T
```
The following statement is mentioned in the gen_PI_image function.

Note: You would need to specify the data directory. Please verify the directory before excuting the GenPI_Images.ipynb file. 


# Model Training 

To train the model you will need to run the train.py script. This script will accept the following arguments:

1) epochs: It is used to define the number of epochs. Default value is 200.
2) dataset: It is used to define the dataset we want to train. It could be can be either cifar10 or cifar100 or cinic10. Default value is 'cifar10'
3) batch_size: It is used to define the batch size. Default value is 128
4) alpha: It is define the alpha parameter. It is one of the hyperparam that's used while calculating the loss. Default value is 0.95
5) learning_rate: It is used to define the initial learning rate. Default value is 0.1
6) momentum: It is used to define the SGD momentum. Default value is 0.9
7) weight_decay: It is used to define the SGD weight decay (default: 1e-4).
8) teacher1: It is used to define the model architecture for 1st teacher model 
9) teacher2: It is used to define the model architecture for 2nd teacher model 
10) student: It is used to define the model architecture for student model.
11) teacher_checkpoint1: It's an optional argument. It is used to define the pretrained model checkpoint for 1st teacher model  
12) teacher_checkpoint2: It's an optional argument. It is used to define the pretrained model checkpoint for 2nd teacher model  
13) student_checkpoint: : It's an optional argument. It is used to define the pretrained model checkpoint for student model  
14) cuda: It is used to define the whether or not use cuda(train on GPU).
15) dataset_dir: It is used to define the dataset directory
16) trial: It is used to define the trial memo number
17) sbj: It is used to define the sbj number
18) seed: It is used to define the seed for given experiment
19) save_weight: By default it is set to 0. If we wanna save the train weight of our model we should set it as 1.

Command to train the model:

'''
python3 main_train.py --epochs 200 --alpha 0.99 --teacher1 wrn163 --teacher2 wrn163 --teacher_checkpoint1 ./models/wrn163_Teacher1.pth.tar --teacher_checkpoint2 ./models/wrn163_Teacher2.pth.tar --student wrn161 --cuda 1 --dataset cifar10 --batch-size 128 --trial T_wrn163_163_S_wrn161_TGD --seed 1234 --save_weight 0 --student_checkpoint ./models/wrn161_Student.pth.tar
'''


# Model Evaluation

To train the model you will need to run the main_eval.py script. 

Note: The main_eval.py script also contains most of the arguments similar to the train.py script as mentioned above. The main_eval.py script only contains the 'dataset', 'batch_size', 'student', 'student_checkpoint', 'cuda', 'dataset_dir', 'trial' and 'seed' arguments.

Command to evaluate the model:

'''
python3 main_eval.py --student wrn161 --batch-size 1 --cuda 1 --dataset cifar10 --trial eval_161 --seed 1234 --save_weight 0 --student_checkpoint ./models/T163_S161_TGD.pth.tar
'''

