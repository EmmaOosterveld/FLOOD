import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
import random
import networkx as nx
import torchvision.transforms.v2.functional as trans
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.nn import GCNConv, TAGConv, ChebConv
from tqdm import tqdm
from torch_geometric.data import Data

def get_coords(pos):
    '''
    Returns array of dimensions (n_nodes, 2) containing x and y coordinates of each node
    ------
    pos: dict
        keys: (x,y) index of every node
        values: spatial x and y positions of each node
    '''
    return np.array([xy for xy in pos.values()])

def convert_to_pyg(graph, pos, DEM, WD):
    '''
    Converts a graph or mesh into a PyTorch Geometric Data type
    Then, add position, DEM, and water variables to data object
    - graph: graph grid
    - pos: position mapping for the layout
    - DEM: Digital elevation model
    - WD: water depth
    - VX: velocity in x-direction
    - VY: velocity in y-direction
        
    '''
    DEM = DEM.reshape(-1)

    edge_index = torch.LongTensor(list(graph.edges)).t().contiguous()
    row, col = edge_index

    data = Data()

    delta_DEM = torch.FloatTensor(DEM[col]-DEM[row])
    coords = torch.FloatTensor(get_coords(pos))
    edge_relative_distance = coords[col] - coords[row]
    edge_distance = torch.norm(edge_relative_distance, dim=1)
    edge_slope = delta_DEM/edge_distance

    data.edge_index = edge_index
    data.edge_distance = edge_distance
    data.edge_slope = edge_slope
    data.edge_relative_distance = edge_relative_distance
    data.num_nodes = graph.number_of_nodes()
    data.pos = torch.tensor(list(pos.values()))
    data.DEM = torch.FloatTensor(DEM)
    data.DEM = data.DEM.reshape(data.DEM.shape[0], 1) #reshape to get size [4096,1]
    data.WD = torch.FloatTensor(WD.T)[:,2:]
    start_flood = data.WD[:, 0] # WD0
    start_flood = start_flood.reshape(start_flood.shape[0], 1)
    start_flood1 = data.WD[:, 1] # WD1
    start_flood1 = start_flood1.reshape(start_flood1.shape[0], 1)
    data.x = torch.stack([data.DEM, start_flood, start_flood1], dim=1)[:, : , 0] # input parameters
    data.y = data.WD #output parameters

    return data
        
def Augment_data(water_depth, dem):
  """
  Applies horizontal and vertical flip to the original DEM and water depth
  Input arguments:
  water_depth = Water depth over time for the considered simulation, dtype = array
  dem = DEM for the considered simulation, dtype = array

  Return:
  Water depth and DEM - rotated horizontally and vertically as torch.Tensors
  """
  rotated_dem_horizontal = trans.horizontal_flip(torch.Tensor(dem))
  # Rotate the water depth sequence
  rotated_water_depth_horizontal = trans.horizontal_flip(torch.Tensor(water_depth))

  rotated_dem_vertical = trans.horizontal_flip(torch.Tensor(dem))
  # Rotate the water depth sequence
  rotated_water_depth_vertical = trans.horizontal_flip(torch.Tensor(water_depth))

  return rotated_water_depth_horizontal, rotated_dem_horizontal, rotated_water_depth_vertical, rotated_dem_vertical


def load_data(start_sim, n_sim, folder_path, augmentation=False, augmentation_per=1):
    """
    Loads the DEM and WD dataset.
    Input arguments:
    start_sim = number of the start simulation, dtype=int
    n_sim = number of simulations, dtype=int
    folder_path = folder location of the data, dtype=str

    Return:
    Two list of arrays. One for DEM data and one for WD data.
    """
    augment_list = []
    if augmentation:
      all_simulations = list(range(start_sim, start_sim + n_sim))
      # Specify the percentage of simulations to augment
      num_sim_to_augment = int(augmentation_per * n_sim)

      # Randomly select a subset of simulations to augment
      augment_list = random.sample(all_simulations, num_sim_to_augment)
      augment_list = all_simulations[:num_sim_to_augment+1]

    DEMS = [] # storing DEM
    WDS = [] # storing WD
    for i in tqdm(range(start_sim,start_sim+n_sim)):
        DEM = np.loadtxt(f"{folder_path}/DEM/DEM_{i}.txt")[:, 2]
        DEMS.append(DEM)
        WD = np.loadtxt(f"{folder_path}/WD/WD_{i}.txt")
        WDS.append(WD)

        if i in augment_list:
          augment_WD_h, augment_DEM_h, augment_WD_v, augment_DEM_v  = Augment_data(WD, DEM)
          DEMS.append(np.array(augment_DEM_h))
          WDS.append(np.array(augment_WD_h.squeeze(0)))
          DEMS.append(np.array(augment_DEM_v))
          WDS.append(np.array(augment_WD_v.squeeze(0)))
    return DEMS, WDS
    
def create_dataset(G, pos, DEM, WD):
  """
  Creates a dataset with graphs.
  Input arguments:
    G = graph, dtype=networkx.classes.digraph.DiGraph
    pos = positions, dtype=dict
    DEM = DEM data, dtype=array
    WD = water depth data, dtype=array

  Return:
  A list with graphs
  """
  dataset = []
  for i in range(len(WD)):
    grid_i = convert_to_pyg(G, pos, DEM[i], WD[i]) # assigning values to the graph nodes
    dataset.append(grid_i)
  return dataset

def normalize_WD(scaler, WD_data, scaled_DEM, G, pos, train=False):
  """
  Normalizing the water depth with the given scaler. Also creating the graph dataset.
  Input arguments:
    scaler = scaler to be used for normalizing, dtype =
    WD_data = water depth dataset, dtype = list
    scaled_DEM = normalized DEM data, dtype =

  Keyword arguments:
    train = specify if training dataset or not. default = False, dtype = boolean

  Return:
    list of graphs and, if train=True, the scaler
  """
  original_shape = np.array(WD_data).shape # store the original shape to get same output shape
  reshaped_WD = np.array(WD_data).reshape(-1, original_shape[-1]) # from 3D to 2D
  if train: # fit only on training data
    scaler.fit(reshaped_WD)
  scaled_WD = scaler.transform(reshaped_WD)
  final_WD = scaled_WD.reshape(original_shape)
  dataset = create_dataset(G, pos, scaled_DEM, final_WD)

  if train:
    return scaler, dataset
  return dataset
  
class CustomMinMaxScaler_interpolation:
    """
    Provides a minmax scaler. For test datasets which have a different size than the train dataset on which the scaler was fitted, interpolation is applied.

    Input arguments:
    features_range = lower and upper bound scaler, default=(0,1), dtype=tuple

    Methods:
    fit: Fits the scaler to a dataset. Takes the dataset as input.
    transform: Transforms the data to the fitted scaler.
    fit_transform: Does both the fitting and transforming on the data.
    inverse_transform: Inverses the done transformation, i.e. denormalizes the data
    """
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_val = None
        self.max_val = None

    def fit(self, X):
        self.min_val = np.nanmin(X, axis=0)
        self.max_val = np.nanmax(X, axis=0)
        # If there is no variation, i.e. min = max, this will give a Division by Zero error.
        # To prevent this 0.1 is added to the maximum. 
        # Since the numerator will be zero anyway (X - self.new_min = 0, since there is no variation in X)
        # The number added to the maximum is irrelevant, as the outcome will always be zero.
        if self.min_val == self.max_val: 
           self.max_val += 0.1

    def transform(self, X):
        # Check if the scaler has been fitted
        if self.min_val is None or self.max_val is None:
            raise ValueError("Scaler has not been fitted. Call fit() before transform().")

        if X.shape[-1] > self.min_val.shape[-1]: # in case the testdata grid is larger than the traindata grid
          original_size = len(self.min_val)
          new_size = X.shape[-1]
          step_size = original_size / new_size

          # Create an array with indices corresponding to the new size
          indices = np.arange(0, original_size, step_size)

          # Interpolate values using numpy's interpolation function
          self.new_min = np.interp(indices, np.arange(original_size), self.min_val)
          self.new_max = np.interp(indices, np.arange(original_size), self.max_val)

          if self.new_min == self.new_max: 
            self.new_max += 0.1

          # Scale the features to the specified range
          scaled_X = (X - self.new_min) / (self.new_max - self.new_min)

          # Clip values to the specified feature_range
          scaled_X = scaled_X.clip(self.feature_range[0], self.feature_range[1])

          return scaled_X
        # In case the test dataset has the same grid size
        # Scale the features to the specified range
        scaled_X = (X - self.min_val) / (self.max_val - self.min_val)

        # Clip values to the specified feature_range
        scaled_X = scaled_X.clip(self.feature_range[0], self.feature_range[1])

        return scaled_X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        if X.shape[-1] > self.min_val.shape[-1]: # in case the testdata grid is larger than the traindata grid
          X = X * (self.new_max - self.new_min) + self.new_min
          return X
        # In case the test dataset has the same grid size
        X = X * (self.max_val - self.min_val) + self.min_val
        return X