from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


class FeatureProcessor(eqx.Module):
    schema: dict[str, str] = eqx.field(static=True)
    means: dict[str, float] = eqx.field(static=True)
    stds: dict[str, float] = eqx.field(static=True)
    vocabs: dict[str, dict[object, int]] = eqx.field(static=True)

    def __init__(self, schema: dict[str, str], tags: Iterable[dict[str, Any]]):
        """
        Initialize feature processor with schema and task data.

        Parameters
        ----------
        schema : dict[str, str]
            Mapping of feature names to types ('continuous',
            'categorical', 'ordinal').
        tasks : Iterable[Task]
            Task objects containing feature data.
        """
        self.schema = schema

        self.means = {}
        self.stds = {}
        self.vocabs = {}

        if isinstance(tags, dict):
            tags = [tags]

        for name, ftype in schema.items():
            col = [d[name] for d in tags]
            if ftype == "continuous":
                arr = jnp.array(col, dtype=jnp.float32)
                self.means[name] = float(arr.mean())
                self.stds[name] = float(arr.std() + 1e-8)
            elif ftype in ("categorical", "ordinal"):
                uniqs = sorted(set(col))
                self.vocabs[name] = {v: i for i, v in enumerate(uniqs)}
            else:
                raise ValueError(f"Unknown feature type {ftype}")

    def transform(self, x: dict[str, object]) -> jax.Array:
        """
        Transform feature dictionary to normalized array.

        Parameters
        ----------
        x : dict[str, object]
            Dictionary of feature names to values.

        Returns
        -------
        jax.Array
            Normalized feature array of shape (num_features,).
        """
        feats = []
        for name, ftype in self.schema.items():
            val = x[name]
            if ftype == "continuous":
                normed = (float(val) - self.means[name]) / self.stds[name]
                feats.append(normed)
            else:  # categorical or ordinal
                feats.append(float(self.vocabs[name][val]))

        return jnp.array(feats, dtype=jnp.float32)

    def to_dict(self) -> dict:
        """
        Serialize feature processor configuration.

        Returns
        -------
        dict
            Configuration dictionary.
        """
        return {
            "schema": self.schema,
            "means": self.means,
            "stds": self.stds,
            "vocabs": self.vocabs,
        }

    @classmethod
    def reconstruct(cls, **kwargs) -> FeatureProcessor:
        """
        Reconstruct feature processor from configuration.

        Parameters
        ----------
        **kwargs
            Configuration dictionary.

        Returns
        -------
        FeatureProcessor
            Reconstructed feature processor instance.
        """
        return cls(**kwargs)


# =============================================================================


class MixedEmbedding(eqx.Module):
    """
    Mixed embedding layer for categorical and continuous features.

    Attributes
    ----------
    embedders : list[eqx.Module]
        List of embedding/linear layers per feature.
    final_proj : eqx.nn.Linear
        Final projection layer.
    """

    embedders: list[eqx.Module]
    final_proj: eqx.nn.Linear

    def __init__(self, embedders: list[eqx.Module], final_proj: eqx.nn.Linear):
        """
        Initialize mixed embedding layer.

        Parameters
        ----------
        embedders : list[eqx.Module]
            List of embedding/linear layers.
        final_proj : eqx.nn.Linear
            Final projection layer.
        """
        self.embedders = embedders
        self.final_proj = final_proj

    @classmethod
    def from_feature_processor(
        cls,
        feature_processor: FeatureProcessor,
        embed_dim: int,
        output_dim: int,
        *,
        key: jax.Array,
    ):
        """
        Create embedding from feature processor schema.

        Parameters
        ----------
        feature_processor : FeatureProcessor
            Feature processor with schema and vocabularies.
        embed_dim : int
            Dimension of embeddings.
        output_dim : int
            Output dimension after projection.
        key : jax.Array
            Random key for initialization.

        Returns
        -------
        MixedEmbedding
            New MixedEmbedding instance.
        """
        embedders = []
        keys = jax.random.split(key, len(feature_processor.schema) + 1)

        for (name, s), k in zip(
            feature_processor.schema.items(), keys[:-1], strict=True
        ):
            if s in ("categorical", "ordinal"):
                embedders.append(
                    eqx.nn.Embedding(
                        num_embeddings=len(feature_processor.vocabs[name]),
                        embedding_size=embed_dim,
                        key=k,
                    )
                )
            elif s == "continuous":
                embedders.append(eqx.nn.Linear(1, embed_dim, key=k))
            else:
                raise ValueError(f"Unknown feature type {s}")

        embedders = embedders
        final_proj = eqx.nn.Linear(
            embed_dim * len(feature_processor.schema),
            output_dim,
            key=keys[-1],
        )

        return cls(embedders=embedders, final_proj=final_proj)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Embed and project mixed features.

        Parameters
        ----------
        x : jax.Array
            Input feature array.

        Returns
        -------
        jax.Array
            Projected embedding vector.
        """
        outputs = []
        for xi, mo in zip(x, self.embedders, strict=True):
            if isinstance(mo, eqx.nn.Embedding):
                xi_input = xi.astype(jnp.int32)
            else:
                xi_input = xi[None]
            outputs.append(mo(xi_input))

        return self.final_proj(jnp.stack(outputs, axis=-1).flatten())

    def to_dict(
        self,
    ) -> dict:
        """
        Serialize embedding configuration.

        Returns
        -------
        dict
            Configuration with embedders and projection info.
        """
        embed_list = []
        for emb in self.embedders:
            if isinstance(emb, eqx.nn.Embedding):
                emb_dict = {
                    "_type": "embedding",
                    "num_embeddings": emb.num_embeddings,
                    "embedding_size": emb.embedding_size,
                }
            elif isinstance(emb, eqx.nn.Linear):
                emb_dict = {
                    "_type": "linear",
                    "in_features": 1,
                    "out_features": emb.out_features,
                }
            else:
                raise ValueError(f"Unknown embedder _type {type(emb)}")

            embed_list.append(emb_dict)

        final_dict = {
            "in_features": self.final_proj.in_features,
            "out_features": self.final_proj.out_features,
        }

        return {"embedders": embed_list, "final_proj": final_dict}

    @classmethod
    def reconstruct(
        cls, embedders, final_proj, key: jax.Array, **kwargs
    ) -> MixedEmbedding:
        """
        Reconstruct embedding from configuration.

        Parameters
        ----------
        embedders : list
            List of embedder configurations.
        final_proj : dict
            Final projection configuration.
        key : jax.Array
            Random key for initialization.
        **kwargs
            Additional arguments (unused).

        Returns
        -------
        MixedEmbedding
            Reconstructed embedding instance.
        """
        k1, k2 = jr.split(key)

        embedders_list = []
        for emb in embedders:
            k, k1 = jax.random.split(k1)
            if emb["_type"] == "embedding":
                embedders_list.append(
                    eqx.nn.Embedding(
                        num_embeddings=emb["num_embeddings"],
                        embedding_size=emb["embedding_size"],
                        key=k,
                    )
                )
            elif emb["_type"] == "linear":
                embedders_list.append(
                    eqx.nn.Linear(
                        in_features=1,
                        out_features=emb["out_features"],
                        key=k,
                    )
                )
            else:
                raise ValueError(f"Unknown embedder type {emb['_type']}")

        final_proj = eqx.nn.Linear(
            in_features=final_proj["in_features"],
            out_features=final_proj["out_features"],
            key=k2,
        )
        return cls(embedders=embedders_list, final_proj=final_proj, **kwargs)
