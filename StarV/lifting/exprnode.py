"""
Expression node definition for lifting DAGs.
Author: Zhuoyang Zhou
Date: 02/17/2026
"""

from dataclasses import dataclass, field
from typing import Any


SUPPORTED_OPS = {
    'var',
    'const',
    'add',
    'sub',
    'mul',
    'pow_even',
    'pow_odd',
    'sin',
    'cos',
    'neg',
}


@dataclass
class ExprNode:
    """Single node in an expression graph."""

    node_id: int | str
    op: str
    inputs: list[int | str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    out_dim: int | None = None

    def __post_init__(self) -> None:
        if self.op not in SUPPORTED_OPS:
            raise ValueError(f"Unsupported op '{self.op}'. Supported ops: {sorted(SUPPORTED_OPS)}")

        if self.op in ('var', 'const') and len(self.inputs) != 0:
            raise ValueError(f"Node '{self.node_id}' with op '{self.op}' must have empty inputs")

        if self.op == 'var':
            name = self.params.get('name', None)
            if not isinstance(name, str) or len(name) == 0:
                raise ValueError(f"Var node '{self.node_id}' requires params['name'] as non-empty string")

        if self.op == 'const' and 'value' not in self.params:
            raise ValueError(f"Const node '{self.node_id}' requires params['value']")

    def is_leaf(self) -> bool:
        return self.op in ('var', 'const')
