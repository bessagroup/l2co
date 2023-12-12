from f3dasm import ExperimentData
import xarray as xr
import numpy as np

experiment_data: ExperimentData = None
_, df_output = experiment_data.to_pandas()

df_output = df_output.loc[:, df_output.columns != 'random_samples']
df_output = df_output.apply(lambda x: experiment_data.project_dir / x)
files = df_output.values.tolist()

ds_raw = xr.open_mfdataset(files , concat_dim=['itemID', 'optimizer'], combine='nested')

ds_output = ds_raw.sel(output_dim='y', drop=True).output

# combine the itemID and realization coordinate
ds_output = ds_output.stack(problem=('itemID', 'realization'))

ds_positive = ds_output - ds_output.min(dim=['optimizer', 'iterations']) + 1.0

rij = ds_positive / ds_positive.min(dim=['problem'])

f_values = np.logspace(0, 0.7, 300)