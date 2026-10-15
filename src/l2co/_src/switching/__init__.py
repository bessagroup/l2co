"""The switching layer: what runs when control passes between optimizers.

* ``sub_optimizer`` -- the :class:`SubOpt` adapter a wrapper dispatches
  across with :func:`jax.lax.switch`, built from an optimizer's
  ``l2co_optimizers.optimizer_parts`` plus its handshake.
* ``handshake_policy`` -- which handshake each optimizer gets on a
  switch, and the canonical population layouts.
* ``menu_eval`` -- menu-dispatched loss evaluation that spends only the
  active optimizer's evaluation width (ADR 0011).
* ``switch`` -- the switch sequence itself, one implementation every
  consumer calls: whether a switch carries news, then reset, write and
  population in that order (ADR 0022).

What each optimizer exposes to this layer (its transfer ports, its
parts) lives in ``l2co_optimizers``; deciding when and how to switch
lives here (ADR 0019). Public through :mod:`l2co.optimizers`.
"""

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================
