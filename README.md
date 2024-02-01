### Group FLOOD1: Spatio-temporal flood estimation using Deep Learning

This is the code repository created in order to provide the code for the project: Spatiotemporal flood modelling with Deep learning. The aim of this project is to selected models that are to predict the spatial-temporal evolution of floods over unseen topographies.
The project content consists of two main notebooks for the models namely CNN U-net style model and Graph neural network (GNN) model. In both cases, the Digital Elevation model and initial condition for water depth have been considered as inputs while predicting the water depths over a time series of 48 hours. 

U-net style model: CNNs are suited, because they work well with images or grid-based data. 

Graph neural network model (GNN): GNN are designed to work with Graph data. The graph consists of nodes that form the structure and edges that join the nodes. In the GNN model, the inputs DEM and WD (initial) have been added to the nodes. The model is made of an encoder-processor-decoder setup where MLPs are used in the encoder and decoder and the processor does the Graph Convolutions. 
The model .py file has been provided in the models folder. A separate notebook that was used for optimising the parameters to be used in the computation.

The repository has been divided into the following components:
1. Data - datasets to be used in the model runs
	a. raw_datasets [ids: 1-80 for training, 501-520 as testing dataset 1, 10001-10020 as Testing dataset 2 and 15001-15020: Testing dataset 3]
2. CNN model - where CNN model is stored. The folder consists of the following parts:

   a. Data - Data_processing_CNN.ipynb and map that contains the preprocessed data for the training set and the three test datasets (If the training dataset is used, first it has to be unzipped)

   b. Models - CNN_UNet.py and a map which contains the trained model with and without data augmentation

   c. Notebooks - CNN_model.ipynb and CNN_model_DataAugmentation.ipynb

   d. Results - gif animations of a simulation from dataset 1, generated with best_model.pth and dataset 2 and 3 generated with best_model_aug.pth
   
4. GNN_model - where GNN model is stored. The folder consists of the following parts:

   a. model - GNN_model.py and best_model_final.pth
   
   b. notebooks - GNN_notebook and GNN_optimizer.
   
   c. processing - GNN_preprocessing.py and GNN_postprocessing.py

   c. results - animations of a simulation from dataset 1, 2 and 3 generated with best_model_final.pth

Note that if you run the Notebooks you get a new, slightly different, best model than the model stored in the 'model' folder. The same for the animations in the 'results' folder. 
