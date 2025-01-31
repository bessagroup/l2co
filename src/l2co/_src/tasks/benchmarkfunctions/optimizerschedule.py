from __future__ import annotations

import json
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional

# =============================================================================


class OptimizationStep:
    def __init__(self, optimizer: str,
                 hyperparameters: Optional[Dict[str, Any]] = None,
                 x0_strategy: Optional[str] = None,
                 n_evals: Optional[int] = None):

        if hyperparameters is None:
            hyperparameters = {}

        self.optimizer = optimizer
        self.hyperparameters = dict(hyperparameters)
        self.x0_strategy = x0_strategy
        self.n_evals = n_evals

    def __repr__(self):
        return (
            f"Optimizer: {self.optimizer}, "
            f"Hyperparameters: {self.hyperparameters}, "
            f"x0_strategy: {self.x0_strategy}, "
            f"n_evals: {self.n_evals}, "
        )

    def __str__(self):
        return (
            f"Optimizer: {self.optimizer}\n"
            f"  Hyperparameters: {self.hyperparameters}\n"
            f"  x0_strategy: {self.x0_strategy}\n"
            f"  n_evals: {self.n_evals}"
        )

    def __eq__(self, other):
        if not isinstance(other, OptimizationStep):
            return False
        return (
            self.optimizer == other.optimizer
            and self.x0_strategy == other.x0_strategy
            and self.n_evals == other.n_evals
            and self.hyperparameters == other.hyperparameters
        )

    @classmethod
    def from_yaml(cls, config: Dict[str, Any]) -> OptimizationStep:
        if 'hyperparameters' not in config:
            return cls(optimizer=config['optimizer'])
        else:
            return cls(optimizer=config['optimizer'],
                       hyperparameters=config['hyperparameters'])

    def to_dict(self) -> Dict[str, Any]:
        return {
            'optimizer': self.optimizer,
            'hyperparameters': self.hyperparameters,
            'x0_strategy': self.x0_strategy,
            'n_evals': self.n_evals
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationStep':
        return cls(
            optimizer=data['optimizer'],
            x0_strategy=data.get('x0_strategy'),
            n_evals=data.get('n_evals'),
            hyperparameters=data.get('hyperparameters'),
        )


class OptimizationSchedule:
    def __init__(self, steps: 'OptimizationStep | list[OptimizationStep]',
                 n_evals: 'int | list[int]'):
        if isinstance(steps, OptimizationStep):
            steps = [steps]

        if isinstance(n_evals, int):
            n_evals = [n_evals]

        self.steps = deepcopy(steps)
        self.tag = {'opt_name': '+'.join([step.optimizer for step in steps])}

        for i, (step, single_n_evals) in enumerate(zip(self.steps, n_evals)):
            if i == 0:
                step.x0_strategy = 'new'
            else:
                step.x0_strategy = get_handshake(
                    self.steps[i-1].optimizer, step.optimizer)
            step.n_evals = single_n_evals

    @property
    def name(self) -> str:
        return self.tag['opt_name']

    def __iter__(self):
        return iter(self.steps)

    def __repr__(self):
        return (
            f"OptimizationSchedule(tag={self.name}, "
            f"steps={self.steps})"
        )

    def __str__(self):
        step_details = "\n".join(
            [f"Step {i+1}:\n{step}" for i, step in enumerate(self.steps)])
        return f"Optimization Schedule ({self.name}):\n{step_details}"

    def __eq__(self, other):
        if not isinstance(other, OptimizationSchedule):
            return False
        return (self.tag == other.tag and self.steps == other.steps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'tag': self.tag,
            'steps': [step.to_dict() for step in self.steps]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationSchedule':
        steps = [OptimizationStep.from_dict(
            step_data) for step_data in data['steps']]
        return cls(steps=steps, n_evals=[step_data['n_evals']
                                         for step_data in data['steps']])

    def save_to_json(self, filepath: Path | str):
        with open(Path(filepath).with_suffix('.json'), 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @staticmethod
    def save(object: OptimizationSchedule, path: str) -> str:
        object.save_to_json(path)
        return Path(path).with_suffix('.json')

    @classmethod
    def load(cls, filepath: str) -> 'OptimizationSchedule':
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


def get_handshake(optimizer_1: str, optimizer_2: str) -> str:
    """
    Return the handshake method to use between two optimizers

    Parameters
    ----------
    optimizer_1 : str
        name of the first optimizer
    optimizer_2 : str
        name of the optimizer to switch from

    Returns
    -------
    str
        method to determine the starting population of the second optimizer

    Notes
    -----
    The handshake method is either 'best', 'last', 'new' or 'random'
    """
    if optimizer_1.lower() == optimizer_2.lower():
        return 'last'

    if optimizer_2.lower() in ['lbfgsb', 'lbfgs', 'adam', 'randomsearch']:
        return 'best'

    if optimizer_1.lower() == 'randomsearch':
        return 'best'

    if optimizer_1.lower() == 'cmaes' and optimizer_2.lower() == 'pso':
        return 'last'

    if optimizer_1.lower() == 'pso' and optimizer_2.lower() == 'cmaes':
        return 'last'

    return 'new'


def get_all_combinations(steps: list[OptimizationStep],
                         total_evals: int, switch: float = 0.5
                         ) -> List[OptimizationSchedule]:

    step_combinations = []

    # Ensure switch is between 0 and 1
    switch = max(0, min(1, switch))

    # Include individual steps as OptimizationSchedule
    step_combinations.extend(
        [OptimizationSchedule(step, total_evals) for step in steps]
    )

    # Include all pairs of steps as OptimizationSchedule
    for step1, step2 in combinations(steps, 2):
        handshake = get_handshake(step1.optimizer, step2.optimizer)

        # Corrected n_evals split
        n_evals1 = int(total_evals * switch)
        n_evals2 = total_evals - n_evals1

        # Create a copy of step2 with updated x0_strategy
        step2 = OptimizationStep(optimizer=step2.optimizer,
                                 hyperparameters=step2.hyperparameters)
        step2.x0_strategy = handshake

        # Append the new schedule
        step_combinations.append(OptimizationSchedule(
            [step1, step2], [n_evals1, n_evals2]))

    return step_combinations
