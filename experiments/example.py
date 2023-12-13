project_dir = '/home/martin/Documents/538734.hpc06.hpc'
experiment_data = ExperimentData.from_file(project_dir)

experiment_data = experiment_data.select(
    range(188)).select_with_status('finished')
experiment_data

iterations = 2029
n_realizations = 10

# features
features = get_labels(experiment_data, n_realizations=n_realizations)

# labels
performance = query_t(get_performance(experiment_data), t=iterations)
labels = performance.idxmin(dim='optimizer').expand_dims(
    dim={'realization': np.arange(n_realizations)}
).stack(problem=['itemID', 'realization'])


clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

ds_data = load_all_data(experiment_data)
ds_stacked = ds_data.stack(problem=['itemID', 'realization'])

# Predict for training data
predictions = clf.predict(features)
da_predictions = xr.DataArray(predictions, dims='problem', coords={
                              'problem': ds_stacked.problem}, name='tree')

performance_profile = create_performance_profile(
    ds_raw=ds_data, iterations=2029, strategies=[da_predictions])

performance_profile.load()

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xscale('log')
# ax.set_yscale('log')
for optimizer in performance_profile.data_vars:
    if optimizer not in ['CMAES', 'LBFGSB', 'Adam', 'PSO']:
        # change line to dashed
        linestyle = '--'
    else:
        linestyle = '-'
    ax.plot(performance_profile.f,
            performance_profile[optimizer], label=optimizer, linestyle=linestyle)

ax.legend()
