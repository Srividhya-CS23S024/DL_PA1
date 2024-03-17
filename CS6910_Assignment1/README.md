## Overview ##

This Folder consists of all my experiments and observations corresponding to programming assignment 1.

## Key Packages used ##
1. wandb
2. numpy
3. pandas
4. matplotlib
5. keras (for downloading datasets)

The details of the experiments and the observations are highlighted in the respective notebooks and the report (wandb) in this Folder.

1. The notebook for running experiments on Fashion Mnist dataset can be accessed from - PA1_CS23S024_FashionMNIST.ipynb.
2. The notebook for running experiments on Mnist dataset can be accessed from - PA1_CS23S024_FashionMNIST.ipynb.
3. The script to run the training with the desired hyperparameter configuration can be done using python script - Train.py

## Wandb report link ## 
https://wandb.ai/cs23s024/CS6910_DL_assignment1/reports/CS6910-Assignment-1--Vmlldzo3MDY3Njky

## How to use Train.py? ##
1. The default hyperparameters are set to the setting with the highest accuracy for both the datasets.
2. To overwrite these parameters, use respective arguments and set the desired values. Use !python Train.py -h - to know the order of the positional arguments.
3. Use the API key from Wandb to connect to wandb project for running and logging results.
4. If due to some issue, this python script could not be used, we can use the notebook -PA1_CS23S024_FashionMNIST.ipynb; change the parameters passed to do_GD() function to run experiments on Fashion Mnist dataset with the desired hyperparameter values. Similarly, we can alter parameters passed to do_GD() function to run experiments on Mnist dataset.
5. Alternatively, we can also change the parameters in the sweep_config and parameters_dict to run for the desired configurations.

## Parameters ##
* X_train, Y_train input and output datasets. X_train to be converted to row vector; Y_train - column vector.
* optimizer-'nadam','sgd', 'momentum', 'nesterov', 'rmsprop', 'adam' for selecting optimizer
* activation-activation function ('sigmoid'/'relu'/'tanh')
* hl_size-size of each hidden layer in a list
* input_size,output_size-size of input and output vectors
* n_epoch-number of epochs
* lr-learning rate
* reg-weight decay term
* w_init-initialization algorithm-'xavier'/'random'
* loss_type-'cross_entropy','squared_error'
* minibatch_size-size of each minibatch
* logging-set to True to log loss and accuracy to wandb


## References ##
- https://docs.wandb.ai/quickstart
- http://www.cse.iitm.ac.in/~miteshk/CS6910.html
- https://github.com/srinivasrdhkrshnn/CS6910_Assignment_1/blob/main/Assignment_1.ipynb
- https://github.com/sowmyamanojna/CS6910-Deep-Learning-Assignment-1/blob/main/Question-4.py
