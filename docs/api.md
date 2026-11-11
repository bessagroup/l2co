# API Reference

## Models

::: l2co.AbstractModel

::: l2co.MLP

::: l2co.LSTM_Encoder

::: l2co.L2COModel

::: l2co.Stage0Model

::: l2co.Stage0Loader

## Feature Processing

::: l2co.FeatureProcessor

::: l2co.MixedEmbedding

## Optimizers

::: l2co.OptimizationStep

::: l2co.UpdateClass

::: l2co.optimizer_mapping

::: l2co.register_optimizer

::: l2co.init_run_state

::: l2co.optimizers.shade_update

## Switching between optimizers

The switching layer (ADR 0019): the `SubOpt` adapter a strategy wrapper dispatches across, the handshake each optimizer gets when control passes to it, and menu-dispatched evaluation. An optimizer's own contribution — its parts and transfer ports — comes from `l2co_optimizers`.

::: l2co.optimizers.SubOpt

::: l2co.optimizers.optstep_to_subopt

::: l2co.optimizers.grad_sub_optimizer

::: l2co.optimizers.pop_sub_optimizer

::: l2co.optimizers.HandshakePolicy

::: l2co.optimizers.handshake_policy_for

::: l2co.optimizers.population_best

::: l2co.optimizers.population_best_one

::: l2co.optimizers.menu_loss_and_grad

## States

::: l2co.RunState

::: l2co.BatchState

::: l2co.HistoryState

::: l2co.OptHistory

### Exporting a history

`HistoryState`, `BatchState` and `OptHistory` live in `l2co_optimizers` with the run loop (ADR 0017); the exports below need xarray and the ERT, so they stay in l2co.

::: l2co.history_to_xarray

::: l2co.history_to_xarray_realizations

::: l2co.history_to_dataloader

## Running Optimization

::: l2co.RolloutWrapper

::: l2co.RunOptimization

::: l2co.reset

::: l2co.run

::: l2co.evaluate

### L2CO strategy wrapper

The canonical execution path for the two-stage L2CO strategy
(`OptimizationStep(optimizer="l2co")`): the trained `L2COModel` is
exposed as a duck-typed evosax population strategy and bridged into
an `UpdateClass` through the optimizer registry, so it runs through
the same jitted rollout machinery as any plain optimizer. The stage-0
fire happens at construction; the stage-1 fire happens in-graph, once,
at the switch generation. See the design record in
`docs/adr/0002-l2co-strategy-canonical-path-via-updateclass-bridge.md`.

::: l2co.l2co_population_strategy

## Data Processing

::: l2co.create_combined_odata

::: l2co.DataLoader

::: l2co.validation_fn

::: l2co.preprocessing_factory

### Performance profiles

How each optimizer's cost to reach a target compares with the cheapest
optimizer's, task by task (ADR 0021): `DataLoader.expected_runtime`
builds the per-task cost table, `performance_profile` turns any subset
of it into ratio curves, and `plot_performance_profile` draws one.

::: l2co.data.performance_profile

::: l2co.plotting.plot_performance_profile

### Time-shared portfolio

A reference schedule built from stored runs with no new optimization:
K schedules interleaved in round robin over function evaluations,
without passing state between them (l2co#115). The result is a schedule
like any other, so ERT, ERTD and AUC read it unchanged.

::: l2co.data.time_shared_portfolio

## Databank

A lazy-loadable databank that stores a whole dataset in a single
consolidated-zarr store (one chunk per `(task, optimizer)`), with the raw
variable-length loss histories kept in a `loss_history_raw/` netCDF
sidecar and named task selections under `tasksets/` (see the design
reference in `src/l2co/_src/databank/README.md`). Exposed under the
`l2co.databank` namespace.

::: l2co.databank.dump_databank

::: l2co.databank.load_databank

::: l2co.databank.load_databank_dataset

::: l2co.databank.load_databank_performance

::: l2co.databank.load_loss_raw

::: l2co.databank.populated_pairs

::: l2co.databank.populated_iterations

::: l2co.databank.load_taskset

::: l2co.databank.verify_databank

::: l2co.databank.DatabankVerificationReport

::: l2co.databank.VerificationFinding

### Policy-artifact provenance

Learned optimizers (rl2co / l2co policies) enter the databank as regular
schedules carrying a provenance record that identifies the evaluated
*policy artifact* (one checkpoint of one training run). Training code
writes a manifest next to every checkpoint; evaluation loads it (a
checkpoint without one is not admissible) and attaches the assembled
record to the loss history. See the design record in
`docs/adr/0001-policy-artifact-provenance-manifest.md`.

::: l2co.databank.mint_run_id

::: l2co.databank.write_manifest

::: l2co.databank.bump_manifest

::: l2co.databank.load_manifest

::: l2co.databank.manifest_path

::: l2co.databank.weights_hash

::: l2co.databank.provenance_record

::: l2co.databank.record_weights_hash

::: l2co.databank.record_task_hashes

::: l2co.databank.policy_identity_hash

::: l2co.databank.load_provenance

::: l2co.databank.held_out_task_ids

::: l2co.databank.policy_artifacts_root

::: l2co.databank.MissingManifestError

::: l2co.databank.WeightsHashMismatchError

## Sampling

::: l2co.sampling

## Plotting

::: l2co.plotting

## Naming

::: l2co.naming

## Tree Utilities

::: l2co.tree_utils
