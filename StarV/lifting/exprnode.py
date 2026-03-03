"""
Expression node definition for lifting DAGs.
Author: Zhuoyang Zhou
Date: 02/17/2026 Updated: 2026/02/26
"""

SUPPORTED_OPS = {
    "var",
    "const",
    "add",
    "sub",
    "mul",
    "powEven",
    "powOdd",
    "sin",
    "cos",
    "neg",
}


class ExprNode:
    """
    A single node in an expression DAG for lifting.

    Fields:
      - id: unique node identifier (int/str)
      - op: operation name, must be in SUPPORTED_OPS
      - inputs: list of upstream node ids (DAG edges)
      - params: extra parameters (e.g., {"name": "x"} or {"value": 1.0} or {"n": 2})
      - outIndex: the index of this node's output observable in the lifted Star (filled by executor)
    """

    def __init__(self, id, op, inputs=None, params=None, outIndex=None):
        self.id = id
        self.op = op
        self.inputs = list(inputs) if inputs is not None else []
        self.params = dict(params) if params is not None else {}
        self.outIndex = outIndex
        self.validate()

    def validate(self):
        # op check
        if self.op not in SUPPORTED_OPS:
            raise ValueError(
                "Unsupported op '{}'. Supported ops: {}".format(
                    self.op, sorted(SUPPORTED_OPS)
                )
            )

        # leaf nodes must have empty inputs
        if self.op in ("var", "const") and len(self.inputs) != 0:
            raise ValueError("Node '{}' with op '{}' must have empty inputs".format(self.id, self.op))

        # arity checks (keeps DAG clean and makes executor simpler)
        if self.op in ("neg", "sin", "cos", "powEven", "powOdd"):
            if len(self.inputs) != 1:
                raise ValueError("Node '{}' with op '{}' requires exactly 1 input".format(self.id, self.op))

        if self.op in ("add", "sub", "mul"):
            if len(self.inputs) != 2:
                raise ValueError("Node '{}' with op '{}' requires exactly 2 inputs".format(self.id, self.op))

        # var node requires a name
        if self.op == "var":
            name = self.params.get("name", None)
            if not isinstance(name, str) or len(name) == 0:
                raise ValueError("Var node '{}' requires params['name'] as a non-empty string".format(self.id))

        # const node requires a value
        if self.op == "const":
            if "value" not in self.params:
                raise ValueError("Const node '{}' requires params['value']".format(self.id))

        # pow nodes require exponent n
        if self.op in ("powEven", "powOdd"):
            if "n" not in self.params:
                raise ValueError("Node '{}' with op '{}' requires params['n']".format(self.id, self.op))
            n = self.params.get("n", None)
            if not isinstance(n, int) or n <= 0:
                raise ValueError("Node '{}' requires params['n'] as a positive integer".format(self.id))

            if self.op == "powEven" and (n % 2 != 0):
                raise ValueError("Node '{}' is powEven but n={} is not even".format(self.id, n))
            if self.op == "powOdd" and (n % 2 != 1):
                raise ValueError("Node '{}' is powOdd but n={} is not odd".format(self.id, n))

    def isLeaf(self):
        return self.op in ("var", "const")

    def __repr__(self):
        return "ExprNode(id={}, op={}, inputs={}, params={}, outIndex={})".format(
            self.id, self.op, self.inputs, self.params, self.outIndex
        )