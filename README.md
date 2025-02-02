# kerastfML


# Directory Structure / Architecture

	The MVC for Machine Learning: Data-Model-Learn (DML)
		<https://hackernoon.com/the-mvc-for-machine-learning-data-model-learner-dml-8127d793f930>

		Components

			Data: data or more accurately data manager that takes charges of managing the dataset

				1) organize data;
				2) format data;
				3) store data in a folder/database;
				4) generate input; 
				5) provide label names for the dataset;

			Model: the machine learning model

			Learn: class that runs learning tasks on the model given data. It is tied to one or more models to perform dynamic operations.
				It encapsulates tasks like
                
                1) train models
				2) making prediction
				3) extracting features
				4) serializing and deserializing trained model to/from a file.

				It connects model and data, kinda like the C in MVC.

		Interactions

			Data <-> Model: model takes data in and form computation logic.

			Data <-> Learn: data is fed into learner directly for running tasks.

			Model <-> Learn: the model is used internally by the learner. During training, the learner takes the data, computes loss and optimization rules based on the model. Learner then updates model parameters. During testing/evaluation, learner passes the data to model to get prediction.


		Directory structure

			data: The folder which contains modules/classes related to Data component
			model: The folder which contains modules/classes related to Model component
			learn: The folder which contains modules/classes related to Learn component


# step 1
# Moving code files into google colab

    move this folder into google colab or clone the following repo into google colab
    https://github.com/kavinduravishka/kerastfML 
    
# step 2
# Preparing directories and training data set before first run

	Before the kwasir-dataset archive is extracted, the directory tree will look like this

	.
	├── data
	│   ├── .....
	├── learn
	│   ├── .....
	├── model
	│   ├── .....
	├── main.py
	└── README.md

	Extract kwasir-dataset in to the same folder where the "main.py" file exists
	After doing it the directory tree should look like the following

	.
	├── data
	│   ├── .....
	├── kvasir-dataset
	│   ├── dyed-lifted-polyps
	│   ├── dyed-resection-margins
	│   ├── esophagitis
	│   ├── normal-cecum
	│   ├── normal-pylorus
	│   ├── normal-z-line
	│   ├── polyps
	│   └── ulcerative-colitis
	├── learn
	│   ├── .....
	├── model
	│   ├── .....
	├── main.py
	└── README.md

	If the data set is not placesd within the right directory with right directory structure, it will be caused for runtime errors


# step 3
# get into kerastfML folder in colab

# step 4
# run following code in colab

    import os

    modpath = os.getcwd() + '/'
    datadirectory = modpath+'kvasir-dataset/'
    batchdirectory = modpath+'databatch/'
    
    from learn.learn import learner
    from data.dirmanager import dirmanager
    from cmd import Cmd
    
    dmngr = dirmanager(datadirectory,batchdirectory)
    mlearner = learner(batchdirectory)
    

# step 5   
# use following commands to train and run

## create nesessary structures
        1st step - dmngr.createdirs() - create directories
        2nd step - dmngr.movedata(360,90,50) - move data into separate folders

## train and test
        mlearner.newModel() - ner CNN  model        
        mlearner.trainModel() - train available model
        mlearner.testModel() - test model after training
    
## save and load    
        mlearner.loadModel(<name>) - load saved model
        mlearnel.saveModel(<name>) - save trained model
        mlearner.listModel() - list available saved models