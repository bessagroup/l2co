"""Feature processing, normalization, and mixed embeddings for task inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

Schema = dict[str, str | list]


# =============================================================================


class FeatureProcessor(eqx.Module):
    """Encodes a raw feature dict into a flat JAX array.

    Continuous values are passed through as-is. Categorical and
    boolean values are mapped to integer indices via a per-feature
    vocabulary; booleans use the fixed mapping
    ``{"false": 0, "true": 1}``.

    Parameters
    ----------
    vocabs : dict[str, dict[object, int] | None]
        Per-feature vocabulary mapping. ``None`` marks a continuous
        feature; a dict gives the value→index lookup for boolean or
        categorical features. Insertion order of ``vocabs`` defines
        the feature ordering used by :meth:`transform`.
    """

    vocabs: dict[str, dict[object, int] | None] = eqx.field(static=True)

    @classmethod
    def init(cls, schema: Schema) -> FeatureProcessor:
        """Build a FeatureProcessor from a schema.

        Parameters
        ----------
        schema : Schema
            Mapping of feature names to type specs:
              - ``'continuous'`` : float, passed through as-is
              - ``'boolean'``    : bool, mapped to 0 or 1
              - iterable         : categorical, gives the full
                vocabulary in the desired index order

        Returns
        -------
        FeatureProcessor
        """
        vocabs: dict[str, dict[object, int] | None] = {}

        for name, spec in schema.items():
            if spec == "continuous":
                vocabs[name] = None
            elif spec == "boolean":
                vocabs[name] = {"false": 0, "true": 1}
            elif hasattr(spec, "__iter__"):
                categories = list(spec)
                vocabs[name] = {v: i for i, v in enumerate(categories)}
            else:
                raise ValueError(
                    f"Unknown feature spec for {name!r}: {spec!r}. "
                    f"Expected 'continuous', 'boolean', or an iterable "
                    f"of categories."
                )

        return cls(vocabs=vocabs)

    def transform(self, x: dict[str, Any]) -> jax.Array:
        """Encode a feature dict to a default-precision JAX array.

        Parameters
        ----------
        x : dict[str, Any]
            Raw feature dict. All keys in ``vocabs`` must be present.

        Returns
        -------
        jax.Array
            Shape (num_features,); dtype follows JAX's default float
            precision (``float32`` by default, ``float64`` when
            ``jax_enable_x64`` is set).
        """
        feats = []
        for name, vocab in self.vocabs.items():
            val = x[name]
            if vocab is None:
                feats.append(val)
            else:
                if isinstance(val, bool):
                    val = str(val).lower()
                feats.append(vocab[val])
        return jnp.array(feats, dtype=jnp.float_)

    def __call__(self, x: dict[str, Any]) -> jax.Array:
        return self.transform(x)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"vocabs": self.vocabs}

    @classmethod
    def reconstruct(cls, vocabs: dict) -> FeatureProcessor:
        return cls(vocabs=vocabs)

    def store(self, path: str) -> str:
        _path = Path(path)
        with open(_path.with_suffix(".eqx"), "wb") as f:
            f.write((json.dumps(self.to_dict()) + "\n").encode())
            eqx.tree_serialise_leaves(f, self)
        return str(_path)

    @classmethod
    def load(cls, path: str) -> FeatureProcessor:
        _path = Path(path)
        with open(_path.with_suffix(".eqx"), "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            fp = cls.reconstruct(**hyperparams)
            return eqx.tree_deserialise_leaves(f, fp)


class MixedEmbedding(eqx.Module):
    """Embeds a mixed-type feature array and projects to output_dim.

    Each feature gets its own layer:
      - continuous  → Linear(1, embed_dim)
      - boolean     → Embedding(2, embed_dim)
      - categorical → Embedding(vocab_size, embed_dim)

    Outputs are concatenated and passed through a final
    Linear(num_features * embed_dim, output_dim) projection.

    Parameters
    ----------
    embedders : list[eqx.Module]
        One layer per feature, in schema order.
    final_proj : eqx.nn.Linear
        Projection from concatenated embeddings to output_dim.
    embed_dim : int
        Embedding size of each individual layer.
    """

    embedders: list[eqx.Module]
    final_proj: eqx.nn.Linear
    embed_dim: int = eqx.field(static=True)

    @property
    def out_size(self) -> int:
        return self.final_proj.out_features

    @classmethod
    def from_feature_processor(
        cls,
        feature_processor: FeatureProcessor,
        embed_dim: int,
        output_dim: int,
        *,
        key: jax.Array,
    ) -> MixedEmbedding:
        """Build a MixedEmbedding from a FeatureProcessor.

        Parameters
        ----------
        feature_processor : FeatureProcessor
        embed_dim : int
            Size of each individual embedding / linear output.
        output_dim : int
            Dimensionality of the final projected output.
        key : jax.Array

        Returns
        -------
        MixedEmbedding
        """
        num_features = len(feature_processor.vocabs)
        keys = jax.random.split(key, num_features + 1)

        embedders: list[eqx.Module] = []
        for (_name, vocab), k in zip(
            feature_processor.vocabs.items(), keys[:-1], strict=True
        ):
            if vocab is None:
                embedders.append(eqx.nn.Linear(1, embed_dim, key=k))
            else:  # boolean or categorical
                embedders.append(
                    eqx.nn.Embedding(
                        num_embeddings=len(vocab),
                        embedding_size=embed_dim,
                        key=k,
                    )
                )

        final_proj = eqx.nn.Linear(
            embed_dim * num_features,
            output_dim,
            key=keys[-1],
        )
        return cls(
            embedders=embedders,
            final_proj=final_proj,
            embed_dim=embed_dim,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Embed features and project to output_dim.

        Parameters
        ----------
        x : jax.Array
            Default-precision float array of shape (num_features,) as
            produced by FeatureProcessor.transform.

        Returns
        -------
        jax.Array
            Shape (output_dim,).
        """
        outputs = []
        for xi, layer in zip(x, self.embedders, strict=True):
            if isinstance(layer, eqx.nn.Embedding):
                outputs.append(layer(xi.astype(jnp.int32)))
            else:
                outputs.append(layer(xi[None]))
        return self.final_proj(jnp.concatenate(outputs, axis=0))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        embed_list = []
        for layer in self.embedders:
            if isinstance(layer, eqx.nn.Embedding):
                embed_list.append(
                    {
                        "_type": "embedding",
                        "num_embeddings": layer.num_embeddings,
                        "embedding_size": layer.embedding_size,
                    }
                )
            else:
                embed_list.append({"_type": "linear"})
        return {
            "embedders": embed_list,
            "embed_dim": self.embed_dim,
            "output_dim": self.final_proj.out_features,
        }

    @classmethod
    def reconstruct(
        cls, embedders, embed_dim, output_dim, **kwargs
    ) -> MixedEmbedding:
        key = jr.key(0)
        k, *layer_keys = jax.random.split(key, len(embedders) + 1)

        embedders_list: list[eqx.Module] = []
        for emb, lk in zip(embedders, layer_keys, strict=True):
            if emb["_type"] == "embedding":
                embedders_list.append(
                    eqx.nn.Embedding(
                        num_embeddings=emb["num_embeddings"],
                        embedding_size=emb["embedding_size"],
                        key=lk,
                    )
                )
            else:
                embedders_list.append(eqx.nn.Linear(1, embed_dim, key=lk))

        proj = eqx.nn.Linear(
            in_features=embed_dim * len(embedders),
            out_features=output_dim,
            key=k,
        )
        return cls(
            embedders=embedders_list,
            final_proj=proj,
            embed_dim=embed_dim,
        )

    def store(self, path: str) -> str:
        _path = Path(path)
        with open(_path.with_suffix(".eqx"), "wb") as f:
            f.write((json.dumps(self.to_dict()) + "\n").encode())
            eqx.tree_serialise_leaves(f, self)
        return str(_path)

    @classmethod
    def load(cls, path: str) -> MixedEmbedding:
        _path = Path(path)
        with open(_path.with_suffix(".eqx"), "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            me = cls.reconstruct(**hyperparams)
            return eqx.tree_deserialise_leaves(f, me)
