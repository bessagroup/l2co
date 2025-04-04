# Third-party
import jax
import jax.tree as jt
from jaxtyping import PyTree

# =============================================================================


def tree_to_dict_list(tree: PyTree, name: str) -> list[dict[str, PyTree]]:
    """
    Convert a PyTree into a list of dictionaries.

    Parameters
    ----------
    tree : PyTree
        The PyTree to convert.
    name : str
        The key name to use in the dictionaries.

    Returns
    -------
    List[Dict[str, PyTree]]
        A list of dictionaries representing the PyTree.
    """
    num_models = jax.tree_util.tree_leaves(
        tree)[0].shape[0]  # Get the batch size

    return [{name:
             jax.tree_util.tree_map(lambda param: param[i], tree)}
            for i in range(num_models)]


def dict_list_to_tree(dict_list: list[dict[str, PyTree]], name: str) -> PyTree:
    """
    Convert a list of dictionaries into a PyTree.

    Parameters
    ----------
    dict_list : List[Dict[str, PyTree]]
        The list of dictionaries to convert.
    name : str
        The key name to extract from the dictionaries.

    Returns
    -------
    PyTree
        The resulting PyTree.
    """
    return jt.map(lambda *params: jax.numpy.stack(params, axis=0),
                  *[d[name] for d in dict_list])
