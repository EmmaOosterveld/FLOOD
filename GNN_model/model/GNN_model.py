from torch import nn
from torch_geometric.nn import GCNConv, TAGConv, ChebConv

def make_mlp(in_features, out_features, hidden_features, dropout=0, n_layers_MLP=2):
    """
    Builds a MLP with linear layers and ReLU activation.
    Input parameters:
    in_features = number of features in the input channel, dtype=int
    out_features = number of features in the output channel, dtype=int
    hidden_features = number of features in the hidden channel(s), dtype=int
    Keyword arguments:
    dropout = dropout rate, default=0, dtype=float
    n_layers_MLP = number of layers in the MLP, default=2, dtype=int
    
    Return:
    A MLP consisting of the given amount of layers in the form of a torch.nn.Sequential
    """
    
    layers = []
    if n_layers_MLP == 1:
        layers.append(nn.Linear(in_features, out_features))
    else:
        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.BatchNorm1d(hidden_features))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_features, hidden_features))

        for i in range(n_layers_MLP - 2):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.BatchNorm1d(hidden_features))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_features, hidden_features))

        layers.append(nn.Linear(hidden_features, out_features))
    layers.append(nn.Dropout(dropout))
    layers.append(nn.BatchNorm1d(out_features))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(out_features, out_features))

    mlp = nn.Sequential(*layers)
    return mlp

class GNN_Conv(nn.Module):
    """
    Applies a chosen GNN convolution together with batch normalization and ReLU activation.
    Input arguments:
    in_features = number of features in the input channel, dtype=int
    out_features = number of features in the output channel, dtype=int
    Keyword arguments:
    K = number of hops in graph convolution, default = 20, dtype=int
    convolution_type = type of graph convolution to apply. Choose from ['ChebConv', 'TAGConv', 'GCNConv'].
    Default='ChebConv', dtype=str
    
    Return:
    A GNN convolution class
    """
    def __init__(self, in_features, out_features, K=20, convolution_type='ChebConv'):
        super().__init__()
        if convolution_type == 'ChebConv':
            self.conv = ChebConv(in_features, out_features, K=K)
        elif convolution_type == 'TAGConv':
            self.conv = TAGConv(in_features, out_features, K=K)
        elif convolution_type == 'GCNConv':
            self.conv = GCNConv(in_features, out_features, K=K)
        else:
            print('Please choose a valid convolution type: ChebConv, TAGConv or GCNConv')
        self.batch = nn.BatchNorm1d(out_features)
        self.activation = nn.ReLU()
    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.batch(x)
        x = self.activation(x)
        return x

class encoder_decoder_GNN(nn.Module):
    """
    Applies a encoder decoder architecture with MLPs. The processor consists of Graph Neural Network convolution(s).
    Input arguments:
    in_features = number of features in the input channel, dtype=int
    out_features = number of features in the output channel, dtype=int
    hidden_features = number of features in the hidden channel(s), dtype=int
    Keyword arguments:
    n_layers_MLP = number of layers in the encoder/decoder MLP, default=2, dtype=int
    n_layers_GNN = number of layers in the processor, default=2, dtype=int
    K = number of hops in graph convolution, default = 20, dtype=int
    convolution_type = type of graph convolution to apply. Choose from ['ChebConv', 'TAGConv', 'GCNConv'].
    Default='ChebConv', dtype=str
    
    Return:
    A GNN convolution class
    """
    def __init__(self, in_features, out_features, hidden_features, n_layers_MLP=2, n_layers_GNN=2, K=20, convolution_type='ChebConv'):
        super().__init__()
        self.encoder = make_mlp(in_features, hidden_features, hidden_features, n_layers_MLP=n_layers_MLP)
        self.layers = nn.ModuleList()
        for i in range(n_layers_GNN):
            self.layers.append(GNN_Conv(hidden_features, hidden_features, K=K, convolution_type=convolution_type))
        self.decoder = make_mlp(hidden_features, out_features, hidden_features, n_layers_MLP=n_layers_MLP)

    def forward(self, data):
        """
        data is the PyG object that contains (among the rest):
            - x: node feature matrix (shape: [num_nodes, in_features])
            - edge_index: graph connectivity
        """
        x, edge_index = data.x, data.edge_index
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x, edge_index)
        x = self.decoder(x)
        return x
    
def build_network(hidden_features, n_layers_MLP, n_layers_GNN, K, convolution_type):
    """
    Builds encoder_decoder_GNN with the DEM, waterdepth at timestep 0 and waterstep at time step 1 as input of the network.
    The output of the network are the waterdepths at the 95 following time steps.
    Input arguments:
    hidden_features = number of features in the hidden channel(s), dtype=int
    n_layers_MLP = number of layers in the encoder/decoder MLP, dtype=int
    n_layers_GNN = number of layers in the processor, dtype=int
    K = number of hops in graph convolution, dtype=int
    convolution_type = type of graph convolution to apply. Choose from ['ChebConv', 'TAGConv', 'GCNConv'].
    
    Return:
    An encoder_decoder_GNN network
    """
    network = encoder_decoder_GNN(3, 95, hidden_features,
                  n_layers_MLP=n_layers_MLP,
                  n_layers_GNN=n_layers_GNN, 
                  K=K,
                  convolution_type=convolution_type)

    return network