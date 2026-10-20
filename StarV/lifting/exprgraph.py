"""
Expression graph utilities for lifting DAGs.
Author: Zhuoyang Zhou
Date: 02/17/2026 Updated: 2026/02/27
"""
from StarV.lifting.exprnode import ExprNode


class ExprGraph:
    """
    A lightweight DAG wrapper around a list of ExprNode objects.

    Responsibilities:
      1) Build id -> node index
      2) Validate references (inputs exist)
      3) Build dependency structure (inDegree + successors)
      4) Provide topological order for execution
    """

    def __init__(self, nodes):
        self.nodes = list(nodes) if nodes is not None else []
        self.idToNode = {}
        self.inDegree = {}
        self.successors = {}
        self.buildIndex()
        self.buildDeps()

    def buildIndex(self):
        """Build id -> node map, and check duplicate ids."""
        self.idToNode = {}
        for node in self.nodes:
            if node is None:
                raise ValueError("ExprGraph: found None in node list")

            nodeId = node.id
            if nodeId in self.idToNode:
                raise ValueError("ExprGraph: duplicate node id '{}'".format(nodeId))

            self.idToNode[nodeId] = node

    def buildDeps(self):
        """
        Build:
          - inDegree[nodeId] = number of inputs
          - successors[inputId] = list of nodeIds that depend on inputId
        Also validates that every input reference exists.
        """
        # initialize tables
        self.inDegree = {}
        self.successors = {}
        for node in self.nodes:
            self.inDegree[node.id] = 0
            self.successors[node.id] = []

        # fill edges
        for node in self.nodes:
            nodeId = node.id
            inputs = node.inputs if node.inputs is not None else []

            # validate references
            for inId in inputs:
                if inId not in self.idToNode:
                    raise ValueError("ExprGraph: node '{}' references missing input id '{}'".format(nodeId, inId))

                # edge: inId -> nodeId
                self.successors[inId].append(nodeId)
                self.inDegree[nodeId] += 1

    def getNode(self, nodeId):
        """Return node by id."""
        if nodeId not in self.idToNode:
            raise KeyError("ExprGraph: unknown node id '{}'".format(nodeId))
        return self.idToNode[nodeId]

    def getRoots(self):
        """Return all root nodes (inDegree == 0)."""
        roots = []
        for node in self.nodes:
            if self.inDegree.get(node.id, 0) == 0:
                roots.append(node)
        return roots

    def topoSort(self):
        """
        Return nodes in a valid topological order (Kahn's algorithm).
        Raises an error if a cycle exists.
        """
        # copy degrees so we don't destroy the stored ones
        deg = {}
        for k, v in self.inDegree.items():
            deg[k] = v

        # init queue with inDegree==0
        queue = []
        for node in self.nodes:
            if deg[node.id] == 0:
                queue.append(node.id)

        orderIds = []
        head = 0

        # standard BFS-like process
        while head < len(queue):
            curId = queue[head]
            head += 1

            orderIds.append(curId)

            for nxtId in self.successors.get(curId, []):
                deg[nxtId] -= 1
                if deg[nxtId] == 0:
                    queue.append(nxtId)

        # cycle detection
        if len(orderIds) != len(self.nodes):
            # Find nodes still with deg > 0 for debugging
            stuck = []
            for node in self.nodes:
                if deg.get(node.id, 0) > 0:
                    stuck.append(node.id)

            raise ValueError(
                "ExprGraph: cycle detected or graph not fully connected. "
                "Unresolved nodes: {}".format(stuck)
            )

        # convert ids to nodes
        orderedNodes = []
        for nodeId in orderIds:
            orderedNodes.append(self.idToNode[nodeId])

        return orderedNodes

    def summary(self):
        """Return a readable text summary for debugging."""
        lines = []
        lines.append("ExprGraph summary:")
        lines.append("  numNodes: {}".format(len(self.nodes)))
        lines.append("  roots: {}".format([n.id for n in self.getRoots()]))

        lines.append("  inDegree:")
        for node in self.nodes:
            lines.append("    {}: {}".format(node.id, self.inDegree.get(node.id, 0)))

        lines.append("  successors:")
        for node in self.nodes:
            lines.append("    {} -> {}".format(node.id, self.successors.get(node.id, [])))

        return "\n".join(lines)