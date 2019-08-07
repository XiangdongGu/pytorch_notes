#%%
import torch
import numpy as np
import csv

#%% [markdown]
# ### Tabular Data
# - *scatter_* to do one hot encoding, the arguments are: dimension along which the following arguments are specified, a column tensor for the indicies to scatter, a tensor containing the elements to scatter or a single scalar to scatter
# - unsqueeze add a singleton dimension without extra element added
# - torch.mean will calculate mean on different dimensions 0 for column, 1 for row

#%%
wine_path = 'data/winequality-white.csv'
wineq_numpy = np.loadtxt(wine_path, dtype = np.float32, delimiter = ";", skiprows = 1)
col_list = next(csv.reader(open(wine_path), delimiter=';'))
wineq = torch.from_numpy(wineq_numpy)

# one-hot encoding
target = wineq[:, -1].long()
target_onehot = torch.zeros(target.shape[0], 10)
target_onehot.scatter_(1, target.unsqueeze(1), 1.0)

data = wineq[:, :-1]
data_mean = torch.mean(data, dim = 0)
data_var = torch.var(data, dim = 0)
data_normalized = (data - data_mean) / torch.sqrt(data_var)

#%% [markdown]
# ### Time Series

# %%
