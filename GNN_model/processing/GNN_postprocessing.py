import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def output_animation(d_dem, d_output, d_data, id, dataset, size, pos):
  """
  Generates an animation of the outputs per timestep. 

  Input arguments:
  d_dem = denormalized input of the DEM, dtype = torch.float64
  d_output = denormalized output from the model, dtype = torch.float64
  d_data = denormalized water depths, dtype = torch.float64
  id = simulation number, dtype = int
  dataset = test dataset number, dtype = int
  size = grid size, dtype = int
  pos = positions on the graph, dtype = dict

  Return:
  An animation with DEM, predicted waterdepths and actual waterdepths for the considered dataset[num_sim] over time
  """
  val = pos.values()
  x = []
  y = []
  for i in val:
    x.append(i[0])
    y.append(i[1])
  num_timesteps = d_output.shape[1]

  print(f'The results are shown for test dataset {dataset}; num_sim = {id}')
  # Create an empty figure
  fig, ax = plt.subplots(1, 3, figsize=(10, 4))

  # Create initial plot for DEM
  correct_dem = torch.flip(d_dem[:, 0].reshape(size, size), dims=[0])
  dem_plot = ax[0].imshow(correct_dem, cmap='terrain', extent=(min(x), max(x), min(y), max(y)))
  ax[0].set_title('DEM')

  # Create initial scatter plots
  WD_plot_predicted = ax[1].imshow(d_output[:,0].reshape(size, size), cmap='Blues', vmin= np.min(np.array(d_data)) , vmax=np.max(np.array(d_data)))
  ax[1].set_title('Predicted WaterDepth [m]')
  ax[1].set_xlim(min(x), max(x))
  ax[1].set_ylim(min(y), max(y))

  WD_plot_original = ax[2].imshow(d_data[:,0].reshape(size, size), cmap='Blues', vmin= np.min(np.array(d_data)) , vmax=np.max(np.array(d_data)))
  ax[2].set_title('Original Waterdepth [m]')
  ax[2].set_xlim(min(x), max(x))
  ax[2].set_ylim(min(y), max(y))

  # Add colorbars outside the animation loop
  cbar1 = fig.colorbar(dem_plot, ax=ax[0], fraction=0.045, pad = 0.1)
  cbar2 = fig.colorbar(WD_plot_predicted, ax=ax[1], fraction=0.045, pad = 0.1)
  cbar3 = fig.colorbar(WD_plot_original, ax=ax[2], fraction=0.045, pad = 0.1)

  plt.subplots_adjust(wspace=0.3, right=0.90)

  # Function to update the scatter plot for each timestep
  def update(timestep):
      ax[0].clear()  # Clear the current axis
      correct_dem = torch.flip(d_dem[:, 0].reshape(size, size), dims=[0])
      dem_plot = ax[0].imshow(correct_dem, cmap='terrain', extent=(min(x), max(x), min(y), max(y)))
      ax[0].set_title('DEM')

      ax[1].clear()  # Clear the current axis
      WD_plot_predicted = ax[1].imshow(d_output[:,timestep].reshape(size, size), cmap='Blues',
                                      vmin= np.min(np.array(d_data)) , vmax=np.max(np.array(d_data)))
      ax[1].set_title('Predicted WaterDepth [m]')
      ax[1].set_xlim(min(x), max(x))
      ax[1].set_ylim(min(y), max(y))

      ax[2].clear()  # Clear the current axis
      WD_plot_original = ax[2].imshow(d_data[:,timestep].reshape(size, size), cmap='Blues', vmin= np.min(np.array(d_data)) ,
                                      vmax=np.max(np.array(d_data)))
      ax[2].set_title('Original Waterdepth [m]')
      ax[2].set_xlim(min(x), max(x))
      ax[2].set_ylim(min(y), max(y))

      fig.suptitle(f'Predictions for dataset {dataset}[{id}] at Timestep - {timestep / 2 + 1} hours') # +1 as timestep 0 (0 hours) and timestep 1 (0.5 hours) are given as input

  # Create the animation
  animation = FuncAnimation(fig, update, frames=num_timesteps, interval=300, repeat=False)

  # Display the animation
  html_output = animation.to_jshtml()
  display(HTML(html_output))
  
  # Save the animation to an HTML file
  animation.save(f'animation_dataset{dataset}_simulation_{id}.gif', writer='pillow')
  
def compute_test_loss(test_dataset, model, WD_scaler, loss_function):
  """
  Computes the test loss
  
  Input arguments:
  test_dataset = list of graphs for test dataset, dtype = list
  model = trained model, dtype = class
  WD_scaler = fitted scaler to water depth, dtype = class
  loss_function = loss function, dtype = func
  
  Return:
  test_loss = average test loss for the considered dataset, dtype = float
  """
  test_loss = []
  for i in test_dataset:
    test_output = model(i).detach()
    output_size = test_output.shape[-1]
    denormalized_test_output = WD_scaler.inverse_transform(test_output.T).T
    denormalized_data = WD_scaler.inverse_transform(i.WD.T).T
    test_loss.append(loss_function(torch.Tensor(denormalized_test_output), torch.Tensor(denormalized_data[:, :output_size])))
  return np.array(test_loss).mean()