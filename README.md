This is the code repository created in order to provide the code for the project: Spatiotemporal flood modelling with Deep learning. The aim of this project is to selected models should be able to predict the spatial-temporal evolution of floods over unseen topographies.
The project content consists of two main notebooks for the models namely U-net style model and Graph neural network model. In both cases, the Digital Elevation model and initial condition for water depth have been considered as inputs while predicting the water depths over a time series of 48 hours. 

U-net style model: CNNs are suited, because they work well with images or grid-based data. 

Graph neural network model (GNN): GNN are designed to work with Graph data. The graph consists of nodes that form the structure and edges that join the nodes. In the GNN model, the inputs DEM and WD (initial) have been added to the nodes. The model is made of an encoder-processor-decoder setup where MLPs are used in the encoder and decoder and the processor does the Graph Convolutions. 
The model .py file has been provided in the models folder. A separate notebook that was used for optimising the parameters to be used in the computation.

The repository has been divided into the following folder:
1. Data - datasets to be used in the model runs
	a. raw_datasets [ids: 1-80 for training, 501-520 as testing dataset 1, 10001-10020 as Testing dataset 2 and 15001-15020: Testing dataset 3]
2. Models - where all the models are stored. py files contain the model + supporting functions
3. Notebooks (python notebook to be used to run the models for training and testing)
	a. CNN 
	b. GNN - GNN_model.py and GNN_preprocessing.py

Note that if you run the Notebooks you get a new, slightly different, best model than the model stored in the 'Models' folder.
