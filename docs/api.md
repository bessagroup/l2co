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

## States

::: l2co.RunState

::: l2co.BatchState

::: l2co.HistoryState

::: l2co.OptHistory

## Running Optimization

::: l2co.RolloutWrapper

::: l2co.RunOptimization

::: l2co.reset

::: l2co.run

::: l2co.evaluate

## Data Processing

::: l2co.create_combined_odata

::: l2co.DataLoader

::: l2co.validation_fn

::: l2co.preprocessing_factory

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

## Sampling

::: l2co.sampling

## Plotting

::: l2co.plotting

## Naming

::: l2co.naming

## Tree Utilities

::: l2co.tree_utils
