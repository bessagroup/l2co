"""create_optimization_dataset pipeline using f3dasm Pipeline API."""

#                                                                       Modules
# =============================================================================

# Standard
from dataclasses import replace
from functools import partial
from pathlib import Path

# Third-party
import bessaplots
import hydra
from f3dasm import (
    Block,
    CollectArrayResults,
    ExperimentData,
    Pipeline,
    SlurmResources,
    Step,
    SlurmCluster,
    datagenerator,
)
from f3dasm.design import Domain, grid
from hydra.utils import instantiate

# Local
from l2co import OptimizationStep, Task, create_combined_odata
import l2co
import numpy as np
from omegaconf import DictConfig, OmegaConf
from l2co import RunOptimization

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================
#
# =============================================================================


#  Step 1: Create ExperimentData with tasks, model, and experience placeholders
# =============================================================================


def _build_task_experimentdata(
    experimentdata_cfg: DictConfig,
    sampler_cfg: DictConfig,
    sampler_kwargs: DictConfig,
    data_generator_cfg: DictConfig,
    task_kwargs: DictConfig,
    project_dir: Path,
) -> ExperimentData:
    """Build a task ExperimentData from one factory and its sampled rows."""
    # `input_data` rows must reach ExperimentData as plain list-of-dicts;
    # an OmegaConf ListConfig is rejected by its _dict_factory. The
    # domain/grid path is left untouched.
    if "input_data" in experimentdata_cfg:
        experimentdata_cfg = OmegaConf.to_container(
            experimentdata_cfg, resolve=True
        )
    experiment_data = ExperimentData.from_yaml(experimentdata_cfg)
    experiment_data = experiment_data.set_project_dir(project_dir / "tasks")
    experiment_data.domain.add_output(
        name="task",
        to_disk=True,
        store_function=Task.save,
        load_function=Task.load,
    )

    sampler = Block.from_yaml(
        init_config=sampler_cfg,
        call_config=sampler_kwargs,
    )
    sampler.arm(experiment_data)
    experiment_data = sampler.call(data=experiment_data)

    task_fn = datagenerator(
        output_names="task",
        domain=experiment_data.domain,
    )(instantiate(data_generator_cfg, _partial_=True))

    return task_fn.call(data=experiment_data, **(task_kwargs or {}))


def create_experimentdata(
    project_dir: Path,
    config: DictConfig,
) -> None:
    """Set up tasks, optimizer schedules, and grid-sampled jobs."""
    # --- Sample tasks from config (single factory or composite `parts`) ---
    # A `parts` list mixes several task factories in one config: each part
    # carries its own data_generator + input_data and is built independently,
    # then the resulting tasks are row-concatenated. Without `parts` this is
    # the classic single-factory grid/explicit path.
    tasks_cfg = config.tasks
    if tasks_cfg.get("parts") is not None:
        built = [
            _build_task_experimentdata(
                OmegaConf.create({"input_data": part.input_data}),
                tasks_cfg.sampler,
                tasks_cfg.sampler_kwargs,
                part.data_generator,
                part.get("task_kwargs"),
                project_dir,
            )
            for part in tasks_cfg.parts
        ]
        experiment_data_tasks = built[0]
        for part_data in built[1:]:
            experiment_data_tasks = experiment_data_tasks + part_data
    else:
        experiment_data_tasks = _build_task_experimentdata(
            tasks_cfg.experimentdata,
            tasks_cfg.sampler,
            tasks_cfg.sampler_kwargs,
            tasks_cfg.data_generator,
            tasks_cfg.task_kwargs,
            project_dir,
        )

    experiment_data_tasks = experiment_data_tasks.set_project_dir(
        project_dir / "tasks"
    )
    experiment_data_tasks.store()

    # --- Create optimizer schedules ---
    steps = [
        {"optimizer": OptimizationStep.from_yaml(opt)}
        for opt in config.optimizers
    ]

    domain_step = Domain()
    domain_step.add_parameter(
        name="optimizer",
        to_disk=True,
        store_function=OptimizationStep.save,
        load_function=OptimizationStep.load,
    )

    experiment_data_schedules = ExperimentData(
        domain=domain_step,
        input_data=steps,
        project_dir=project_dir / "schedules",
    )
    experiment_data_schedules.store()

    # --- Grid sample tasks × optimizers ---
    domain_grid_sampling = Domain()

    _, df_task = experiment_data_tasks.to_pandas(keep_references=True)
    df_optimizer, _ = experiment_data_schedules.to_pandas(keep_references=True)

    ds_task = df_task["task"].apply(
        lambda row: replace(
            row,
            reference=(Path("../tasks/experiment_data") / row.reference),
        )
    )
    ds_optimizer = df_optimizer["optimizer"].apply(
        lambda row: replace(
            row,
            reference=(Path("../schedules/experiment_data") / row.reference),
        )
    )

    domain_grid_sampling.add_category(
        name="task",
        categories=[s.reference for s in ds_task.to_list()],
    )
    domain_grid_sampling.add_category(
        name="optimizer schedule",
        categories=[s.reference for s in ds_optimizer.to_list()],
    )

    experiment_data_combined = grid().call(
        data=ExperimentData(domain=domain_grid_sampling)
    )

    # --- Set up the combined domain with outputs ---
    domain_combined = Domain()
    domain_combined.add_parameter(
        name="task",
        to_disk=True,
        store_function=Task.save,
        load_function=Task.load,
    )
    domain_combined.add_parameter(
        name="optimizer schedule",
        to_disk=True,
        store_function=OptimizationStep.save,
        load_function=OptimizationStep.load,
    )
    domain_combined.add_output(
        name="loss history",
        to_disk=True,
    )

    experiment_data_combined.domain = domain_combined

    # Store to the run directory so subsequent steps can load it
    experiment_data_combined.store(project_dir)


#    Step 3: Post-processing: create combined experience dataset for PPO update
# =============================================================================


class CreateCombinedOdata(Block):
    """Create the combined xarray dataset from results."""

    def __init__(self, budget_steps: int) -> None:
        self.budget_steps: int = budget_steps

    def call(self, data: ExperimentData, **kwargs) -> ExperimentData:
        create_combined_odata(
            experiment_data=data,
            budget_steps=self.budget_steps,
        )
        return data


class DumpDatabank(Block):
    """Write the consolidated-zarr databank from finished runs (opt-in).

    Additive alternative to :class:`CreateCombinedOdata`: instead of one
    monolithic ``entire_dataset.odata`` it writes the consolidated-zarr
    store via :func:`l2co.databank.dump_databank` (one store per dataset,
    one chunk per ``(task, optimizer)``). The tasks / schedules
    ``ExperimentData`` are reloaded from the ``tasks`` / ``schedules``
    sub-directories created in :func:`create_experimentdata`.
    """

    def __init__(
        self, root: str, dataset_name: str, mode: str = "replace"
    ) -> None:
        self.root = Path(root)
        self.dataset_name = dataset_name
        self.mode = mode

    def call(self, data: ExperimentData, **kwargs) -> ExperimentData:
        project_dir = Path(data.project_dir)
        tasks_ed = ExperimentData.from_file(project_dir / "tasks")
        schedules_ed = ExperimentData.from_file(project_dir / "schedules")
        l2co.databank.dump_databank(
            experiment_data=data,
            tasks_experiment_data=tasks_ed,
            schedules_experiment_data=schedules_ed,
            root=self.root,
            dataset_name=self.dataset_name,
            mode=self.mode,
        )
        return data


# =============================================================================


def generate_report(project_dir: Path):
    d = l2co.DataLoader.from_file(project_dir / "entire_dataset.odata")
    fn_names = np.unique(d.task_labels.sel({"feature": "fn_name"})).tolist()
    fn_name_groups = [({"fn_name": [fn]}, fn) for fn in fn_names]

    save_dir = project_dir / "plots"
    save_dir.mkdir(exist_ok=True)

    _, paths_per_function, captions_per_function = (
        l2co.plotting.ertd_per_group(
            d=d,
            groups=fn_name_groups,
            save_dir=save_dir,
        )
    )

    dims = np.unique(d.task_labels.sel({"feature": "dimensionality"})).tolist()
    dim_groups = [({"dimensionality": [dim]}, f"{dim}D") for dim in dims]

    _, paths_per_dimension, captions_per_dimension = (
        l2co.plotting.ertd_per_group(
            d=d,
            groups=dim_groups,
            save_dir=save_dir,
        )
    )

    report = bessaplots.TypstReport(
        title="RTD of CEC2005 Functions",
        author="Martin van der Schelling",
        paper_size="letter",
    )

    report.add_paragraph("RTD of CEC2005 Functions")

    report.add_figures(
        paths=paths_per_function,
        subcaptions=captions_per_function,
        columns=4,
        caption="ERTD per function",
        rows_per_page=4,
    )

    report.add_paragraph("RTD of CEC2005 Functions per dimensionality")

    report.add_figures(
        paths=paths_per_dimension,
        subcaptions=captions_per_dimension,
        columns=4,
        caption="ERTD per dimensionality",
        rows_per_page=4,
    )

    report.save(project_dir / "report")


#                                                            Building pipeline
# =============================================================================


def build_pipeline(config: DictConfig) -> Pipeline:
    """Construct the create_optimization_dataset pipeline.

    Parameters
    ----------
    config : DictConfig
        Hydra config with tasks, optimizers, samplers, budgets,
        etc.

    Returns
    -------
    Pipeline
        Ready to execute with pipeline.run(mode=...).
    """
    # Check if the optimization is box-constrained
    if "bounded" in config.tasks:
        bounded = (
            config.tasks.bounded.lower_bound,
            config.tasks.bounded.upper_bound,
        )
    else:
        bounded = (None, None)

    # Calculate total number of experiments
    experiment_data_tasks = ExperimentData.from_yaml(
        config.tasks.experimentdata
    )

    sampler = Block.from_yaml(
        init_config=config.tasks.sampler,
        call_config=config.tasks.sampler_kwargs,
    )
    sampler.arm(experiment_data_tasks)
    experiment_data_tasks = sampler.call(data=experiment_data_tasks)

    n_tasks = len(experiment_data_tasks)
    n_optimizers = len(config.optimizers)
    total_experiments = int(n_tasks * n_optimizers)

    # Step 2: the DataGenerator that runs each optimization job
    run_optimization = RunOptimization(
        sampler=instantiate(config.samplers),
        budget_fn=instantiate(config.budgets),
        seed=int(config.seed),
        n_realizations=config.n_realizations,
        bounded=bounded,
        clear_jax_cache=config.clear_jax_cache,
        verbose=config.verbose,
        save_input_parameters=config.save_input_parameters,
    )

    # Step 3: collect JSON results, then create combined dataset
    post_process = CollectArrayResults(cleanup=True) >> CreateCombinedOdata(
        budget_steps=config.budget_steps
    )

    # Optionally also write the consolidated-zarr databank (opt-in).
    databank_cfg = config.get("databank", None)
    if databank_cfg is not None:
        post_process = post_process >> DumpDatabank(
            root=databank_cfg.databank_root,
            dataset_name=databank_cfg.dataset_name,
            mode=databank_cfg.get("mode", "replace"),
        )

    return Pipeline(
        name="create_optimization_dataset",
        steps=[
            Step(
                name="create",
                block=create_experimentdata,
                resources=SlurmResources(
                    time="00:30:00",
                    mem="2G",
                    cpus_per_task=1,
                ),
                kwargs={"config": config},
            ),
            Step(
                name="run",
                block=run_optimization,
                dependency="afterok",
                parallel=True,
                array_jobs=total_experiments,
                resources=SlurmResources(
                    time="00:30:00",
                    mem="8G",
                    cpus_per_task=1,
                ),
            ),
            Step(
                name="post",
                block=post_process,
                dependency="afterany",
                resources=SlurmResources(
                    time="02:00:00",
                    mem="16G",
                    cpus_per_task=4,
                ),
            ),
            Step(
                name="report",
                block=generate_report,
                dependency="afterany",
                resources=SlurmResources(
                    time="00:30:00",
                    mem="4G",
                    cpus_per_task=1,
                ),
            ),
        ],
    )


# =============================================================================


@hydra.main(config_path="../conf", config_name="config")
def main(config: DictConfig) -> None:
    pipeline = build_pipeline(config)

    cluster = (
        SlurmCluster.from_yaml(config.cluster)
        if config.cluster.enabled
        else None
    )
    rootdir = Path(config.rootdir) if config.rootdir else None

    pipeline.run(mode=config.mode, cluster=cluster, rootdir=rootdir)


if __name__ == "__main__":
    main()
