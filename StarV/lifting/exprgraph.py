"""
DAG expression graph for lifting workflows.
Author: Zhuoyang Zhou
Date: 02/17/2026
"""

from collections import deque

from StarV.lifting.exprnode import ExprNode


class ExpressionGraph:
    """A minimal directed acyclic expression graph."""

    def __init__(self) -> None:
        self.nodes: dict[int | str, ExprNode] = {}
        self.output_node_ids: list[int | str] = []

    def add_node(
        self,
        node_id: int | str,
        op: str,
        inputs: list[int | str] | None = None,
        params: dict | None = None,
    ) -> ExprNode:
        if node_id in self.nodes:
            raise ValueError(f"Duplicate node id '{node_id}'")

        node = ExprNode(
            node_id=node_id,
            op=op,
            inputs=[] if inputs is None else list(inputs),
            params={} if params is None else dict(params),
        )
        self.nodes[node_id] = node
        return node

    def add_var(self, node_id: int | str, name: str) -> ExprNode:
        return self.add_node(node_id=node_id, op='var', inputs=[], params={'name': name})

    def add_const(self, node_id: int | str, value: float) -> ExprNode:
        return self.add_node(node_id=node_id, op='const', inputs=[], params={'value': float(value)})

    def set_outputs(self, output_node_ids: list[int | str]) -> None:
        for node_id in output_node_ids:
            if node_id not in self.nodes:
                raise ValueError(f"Unknown output node id '{node_id}'")
        self.output_node_ids = list(output_node_ids)

    def topological_sort(self) -> list[int | str]:
        if len(self.nodes) == 0:
            return []

        indegree: dict[int | str, int] = {nid: 0 for nid in self.nodes}
        succ: dict[int | str, list[int | str]] = {nid: [] for nid in self.nodes}

        for nid, node in self.nodes.items():
            for src in node.inputs:
                if src not in self.nodes:
                    raise ValueError(f"Node '{nid}' depends on unknown input node '{src}'")
                indegree[nid] += 1
                succ[src].append(nid)

        q = deque([nid for nid, deg in indegree.items() if deg == 0])
        order: list[int | str] = []

        while q:
            cur = q.popleft()
            order.append(cur)
            for nxt in succ[cur]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    q.append(nxt)

        if len(order) != len(self.nodes):
            raise ValueError('Expression graph is not a DAG (cycle detected)')

        return order

    def validate(self) -> None:
        self.topological_sort()

        if len(self.output_node_ids) == 0:
            raise ValueError('Expression graph has no outputs; call set_outputs([...])')

        for nid in self.output_node_ids:
            if nid not in self.nodes:
                raise ValueError(f"Unknown output node id '{nid}'")
