# esnchaos


Simply Python package to build and use Echo State Networks for the task of predicting
chaotic time series. 
----
### 🚧 Under construction 🚧: 
- Not sure to what extent this package will be developed as there are already nice 
    ESN packages, such as [reservoirpy](https://github.com/reservoirpy/reservoirpy) and 
    [PyRCN](https://github.com/TUD-STKS/PyRCN) and [rescomp](https://github.com/GLSRC/rescomp).
- This is more or less to be seen as a backup of my Master's thesis code.

### Installation (I used python 3.9): 
1. **Fork/Clone the repository to your local disk.** 
2. **Install the package in editable configuration in a virtual environment:** 
   - Create the virtual environment, using a _conda environment_ or _venv_
   - Install the package in editable mode by running the following in the terminal (from the upper level esnchaos directory):
    ```
    pip install -e .
    ```


### TODO (depending on whether this will be turned into a proper package): 
- Add documentation (Consistent docstrings and maybe even proper docs.)
- Add black, ruff, mypy and things like that. 
- Add proper examples. 

--- 

## Usage: 

### Simple  ESN to predict the Lorenz63 system: 
````python
import esnchaos as ec

# Get time series data by simulating the Lorenz63 system using RK4: 
data = ec.simulations.Lorenz63(sigma=10.0, rho=28.0, beta=8/3, dt=0.05).simulate(5000)  # -> data.shape = (5000, 3)

# Split data into train and test set: 
train = data[:3000, :]  # -> train.shape = (3000, 3)
test = data[3000:, :]  # -> test.shape = (2000, 3)

# Build standard Echo State Network: 
## initialize ESN object: 
esn_obj = ec.esn.ESN() 
## Build the esn_obj. Here a selection of possible settings is shown. Only the x_dim argument is strictly required. 
esn_obj.build( 
    # input and reservoir dimension:
    x_dim=3,    
    r_dim=500,
    
    # Leak factor: 
    leak_factor=0.0, 
    
    # Node bias: 
    node_bias_opt="random_bias", 
    node_bias_seed=1, 
    node_bias_scale=0.1, 
    
    # Train noise: 
    x_train_noise_scale=0.0, 
    x_train_noise_seed=None, 
    
    # Activation function: 
    act_fct_opt="tanh", 
    
    # Reservoir Network: 
    n_type_opt="erdos_renyi", 
    n_rad=0.1, 
    n_avg_deg=6.0, 
    
    # R to Rgen: 
    r_to_rgen_opt="linear_r",
    
    # Output fit / training: 
    reg_param=1e-8, 
    ridge_regression_opt="no_bias", 
    
    # Input scaler: 
    scale_input_bool=True, 
    
    # Input matrix: 
    w_in_opt="random_sparse", 
    w_in_scale=1.0, 
    w_in_seed=1,
)

# Train ESN (synchronize for 100 time steps): 
train_fit, train_true = esn_obj.train(use_for_train=train, sync_steps=100)  # -> train_fit.shape = train_true.shape = (2899, 3)

# Predict with ESN (synchronize for 100 time steps): 
test_pred, test_true = esn_obj.predict(use_for_pred=test, sync_steps=100)  # -> test_pred.shape = test_true.shape = (1900, 3)
````

Visualize the results using the plotting library of your choice.
Here matplotlib is used (some other examples use plotly). 
````python
# Plot train-fit and prediction (first 500 steps each and only first dimension of Lorenz63 system):
import matplotlib.pyplot as plt

## Plotting train-fit:
plt.figure()
plt.plot(train_fit[:500, 0], label="train_fit")
plt.plot(train_true[:500, 0], label="train_true")
plt.title("Training")
plt.legend()
plt.show()

## Plotting prediction:
plt.figure()
plt.plot(test_pred[:500, 0], label="test_pred")
plt.plot(test_true[:500, 0], label="test_true")
plt.title("Prediction")
plt.legend()
plt.show()
````

The generated plots look like: 

![readme_training.png](static%2Freadme%2Freadme_training.png) ![readme_pred.png](static%2Freadme%2Freadme_pred.png)

### More usage: 
Look at the files in the ``example`` folder. 

## Features and Modules:
- ``esnchaos.esn``: All the Echo State Network functionality. 
  - ``ESN``
  - ``ESNHybrid``: Hybrid ESN combining a knowledge-based predictor with an ESN. 
- ``esnchaos.data_preprocessing``: Pre-process the simulated data by scaling-and-shifting, 
adding noise, time-delay embedding, pca-transformation. 
- ``esnchaos.measures``: 
- ``esnchaos.simulations``: Simulate many chaotic dynamical systems. 
- ``esnchaos.sweep_experiments``: Conduct parameter sweeps.
  - The output of the sweep experiments is given as a pickled pandas dataframe. 
  You have to create your own visualization. 

## Resources and notes: 
- Most of the code is taken from my other repository, 
[EchoStateNetworkViewer](https://github.com/DuncDennis/EchoStateNetworkViewer), which is 
a [streamlit app](https://github.com/streamlit/streamlit) to visualize the ESN
behaviour. 
- The code in the EchoStateNetworkViewer is in turn influenced by the 
[rescomp](https://github.com/GLSRC/rescomp) package. 
- ⚠️ The author is aware, that this readme does not describe all the possibilities
of this code. 
This is to be seen as a "first commit". 
