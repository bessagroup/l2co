import jax
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import PyTree

# =============================================================================


def tree_extend(base: PyTree, extension: PyTree, expand_axis=True,
                axis: int = 0):

    if expand_axis:
        def _tree_extend(base, extension):
            return jnp.concatenate(
                [base, jnp.expand_dims(extension, axis)], axis=axis
            )
    else:
        def _tree_extend(base, extension):
            return jnp.concatenate([base, extension], axis=axis)

    return jt.map(_tree_extend, base, extension)


def tree_trim(tree: PyTree, n: int, axis: int = 0):
    def _trim_leaf(leaf):
        return jnp.take(leaf, indices=jnp.arange(n, leaf.shape[axis]),
                        axis=axis)

    return jt.map(_trim_leaf, tree)


def tree_flatten_population_dim(params):
    def flatten_fn(x):
        # Get the shape of the parameter
        shape = x.shape
        # Assuming the shape is (batch_size, population_dim, *other_dims)
        batch_size, population_dim, *other_dims = shape
        # Reshape to (batch_size * population_dim, *other_dims)
        return x.reshape((batch_size * population_dim, *other_dims))

    # Apply flatten_fn to each element of the parameter tree
    return jt.map(flatten_fn, params)


def tree_to_dict_list(tree: PyTree, name: str):
    num_models = jax.tree_util.tree_leaves(
        tree)[0].shape[0]  # Get the batch size

    return [{name:
             jax.tree_util.tree_map(lambda param: param[i], tree)}
            for i in range(num_models)]


def dict_list_to_tree(dict_list, name: str) -> PyTree:
    return jt.map(lambda *params: jax.numpy.stack(params, axis=0),
                  *[d[name] for d in dict_list])
