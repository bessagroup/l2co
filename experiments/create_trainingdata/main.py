import logging
from pathlib import Path
from shutil import copyfile
from time import sleep

import f3dasm
import hydra
import numpy as np
from config import Config
from hydra.core.config_store import ConfigStore
from run_opt import execute

from l2co.filehandlers import CombineFeaturesFileHandler


def initial_script(config: Config):

    if config.design.functions == 'all':
        function_names = [f.__name__ for f in f3dasm.functions.FUNCTIONS]

    # Create design of experiments
    function_name = f3dasm.CategoricalParameter(name='function_name', categories=function_names)
    dimensionality = f3dasm.CategoricalParameter(
        name='dimensionality', categories=list(config.design.dimensionality_list))
    noise = f3dasm.CategoricalParameter(name='noise', categories=list(config.design.noise_list))
    seed = f3dasm.DiscreteParameter(name='seed', lower_bound=0, upper_bound=int(1e6))

    status = f3dasm.ConstantParameter(name='status', value='')

    design = f3dasm.DesignSpace(input_space=[function_name, dimensionality, noise, seed], output_space=[status])

    sampler_name = f3dasm.sampling.find_sampler(config.design.sampler)
    sampler: f3dasm.Sampler = sampler_name(design=design, seed=config.design.seed)

    data = sampler.get_samples(numsamples=config.design.number_of_samples)

    # Make sure dimenionality and function comply!
    for index in range(data.get_number_of_datapoints()):
        i = data.get_inputdata_by_index(index)
        function_class: f3dasm.functions.PyBenchFunction = f3dasm.find_class(
            module=f3dasm.functions, query=i['function_name'])
        if not function_class.is_dim_compatible(int(i['dimensionality'])):
            while not function_class.is_dim_compatible(int(i['dimensionality'])):
                i['dimensionality'] = np.random.choice(list(config.design.dimensionality_list))

            data.set_inputdata_by_index(index, i)

    # Store the experimentdata to a csv file
    data.store(config.design.name)

    # Create the JobQueue object
    job_queue = f3dasm.experiment.JobQueue(filename=config.hpc.jobqueue_filename)
    job_queue.create_jobs_from_experimentdata(data)

    # Write the JobQueue object to a file
    job_queue.write_new_jobfile()

    # Initialize filehandler

    Path(f"./{config.files.dirname_recommender}").mkdir(parents=True, exist_ok=True)

    features_filehandler = CombineFeaturesFileHandler(dir_name='features', master_file='totalfeatures_functions')
    interactions_filehandler = CombineFeaturesFileHandler(
        dir_name=config.files.dirname_interactions, master_file='totalinteractions')

    # Continuous check on newly created datafiles

    if config.hpc.jobid == -1:
        return

    while True:
        features_filehandler.run()
        interactions_filehandler.run()

        if features_filehandler.does_master_file_exist():
            copyfile(f'./{config.files.dirname_features}/totalfeatures_functions.csv',
                     f'./{config.files.dirname_recommender}/totalfeatures_functions.csv')

        if interactions_filehandler.does_master_file_exist():
            copyfile(f'./{config.files.dirname_interactions}/totalinteractions.csv',
                     f'./{config.files.dirname_recommender}/totalinteractions.csv')

        if job_queue.is_all_finished():
            logging.info("All jobs are finished. Exiting filehandler loop!")
            break

        sleep(10)


@hydra.main(config_path=".", config_name="config")
def main(config: Config):
    """Main script to call

    Parameters
    ----------
    config
        Configuration parameters defined in config.yaml
    """
    # Execute the initial_script for the first job
    if config.hpc.jobid == 0:
        initial_script(config)

    elif config.hpc.jobid == -1:  # Sequential
        initial_script(config)
        job(config)

    else:
        sleep(3*config.hpc.jobid)  # Just to asynchronize the jobs
        job(config)


def job(config: Config):

    # Retrieve the queue
    job_queue = f3dasm.experiment.JobQueue(filename=config.hpc.jobqueue_filename)

    while True:
        try:
            jobnumber = job_queue.get()
        except f3dasm.experiment.NoOpenJobsError:
            break

        data = f3dasm.design.load_experimentdata(filename=config.design.name)
        args = data.get_inputdata_by_index(jobnumber)

        _ = execute(config, **args)

        job_queue.mark_finished(jobnumber)


cs = ConfigStore.instance()
cs.store(name="f3dasm_config", node=Config)

log = logging.getLogger(__name__)

if __name__ == "__main__":
    main()
