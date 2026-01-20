from l2co._src.update_class import UpdateClass

from .evosax_implementations import evosax_mapping
from .optax_implementations import optax_mapping

optimizers = evosax_mapping | optax_mapping


def optimizer_mapping(name: str) -> UpdateClass:
    """
    Maps optimizer name to corresponding UpdateClass or function.

    Parameters
    ----------
    name : str
        Name of the optimizer.

    Returns
    -------
    UpdateClass
        Corresponding UpdateClass or function for the optimizer.

    Raises
    ------
    ValueError
        If optimizer name is not recognized.
    """
    try:
        return optimizers[name.lower()]
    except KeyError:
        raise ValueError(f"Optimizer '{name}' not recognized.") from None


# import optax

# from l2co._src.update_class import UpdateClass


# def optimizer_mapping(name: str) -> UpdateClass:
#     """
#     Maps optimizer name to corresponding UpdateClass or function.

#     Parameters
#     ----------
#     name : str
#         Name of the optimizer.

#     Returns
#     -------
#     UpdateClass
#         Corresponding UpdateClass or function for the optimizer.

#     Raises
#     ------
#     ValueError
#         If optimizer name is not recognized.
#     """
#     match name:
#         case "adam":
#             return Partial(optax_update, optimizer=optax.adam, popsize=1)
#         case "sgd":
#             return Partial(optax_update, optimizer=optax.sgd, popsize=1)
#         case "sepcmaes":
#             return Partial(
#                 variable_popsize_optimizer,
#                 evosax_class=Sep_CMA_ES,
#             )
#         case "cmaes":
#             return Partial(
#                 variable_popsize_optimizer,
#                 evosax_class=CMA_ES,
#             )
#         case "pso":
#             return Partial(
#                 variable_popsize_optimizer,
#                 evosax_class=PSO,
#             )
#         case "lbfgsb":
#             return Partial(
#                 optax_update_extra_args,
#                 optimizer=optax.lbfgs,
#                 popsize=1,
#             )
#         case "snes":
#             return Partial(
#                 variable_popsize_optimizer,
#                 evosax_class=SNES,
#             )
#         case "simulatedannealing":
#             return Partial(
#                 variable_popsize_optimizer,
#                 evosax_class=SimulatedAnnealing,
#             )
#         case _:
#             raise ValueError(f"Optimizer '{name}' not recognized.")
