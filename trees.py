# SYSTEM IMPORTS
from collections.abc import Iterable, Sequence
from typing import Type, Tuple
import re
import sys


# PYTHON PROJECT IMPORTS


class RootDeletedException(Exception):
    pass


class Node(object):
    def __init__(self: Type["Node"],
                 label: str,
                 children: Sequence[Type["Node"]]):
        self.label: str = label
        self.children: Sequence[Type["Node"]] = children

        for (i, child) in enumerate(self.children):
            if child.parent is not None:
                child.detach()
            child.parent = self
            child.order = i

        self.parent: Type["Node"] = None
        self.order: int = 0

    def __str__(self: Type["Node"]) -> str:
        return self.label

    def _subtree_str(self: Type["Node"]) -> str:
        if len(self.children) != 0:
            return "(%s %s)" % (self.label, " ".join(child._subtree_str() for child in self.children))
        else:
            s = '%s' % self.label
            return s

    def insert_child(self: Type["Node"],
                     i: int,
                     child: Type["Node"]
                     ) -> None:
        if child.parent is not None:
            child.detach()

        child.parent = self
        self.children[i:i] = [child]

        for j in range(i,len(self.children)):
            self.children[j].order = j

    def append_child(self, child):
        if child.parent is not None:
            child.detach()
        child.parent = self
        self.children.append(child)
        child.order = len(self.children)-1

    def delete_child(self: Type["Node"],
                     i: int
                     ) -> None:
        self.children[i].parent = None
        self.children[i].order = 0
        self.children[i:i+1] = []

        for j in range(i,len(self.children)):
            self.children[j].order = j

    def detach(self: Type["Node"]) -> None:
        if self.parent is None:
            raise RootDeleteException
        self.parent.delete_child(self.order)

    def delete_clean(self: Type["Node"]) -> None:
        "Cleans up childless ancestors"
        parent = self.parent
        self.detach()
        if len(parent.children) == 0:
            parent.delete_clean()

    def bottomup(self: Type["Node"]) -> Iterable[Type["Node"]]:
        for child in self.children:
            for node in child.bottomup():
                yield node
        yield self

    def leaves(self: Type["Node"]) -> Iterable[Type["Node"]]:
        if len(self.children) == 0:
            yield self
        else:
            for child in self.children:
                for leaf in child.leaves():
                    yield leaf

class Tree(object):
    def __init__(self: Type["Tree"],
                 root: Node):
        self.root: Node = root

    def __str__(self: Type["Tree"]) -> str:
        if self.root is None: return ""
        return self.root._subtree_str()

    interior_node = re.compile(r"\s*\(([^\s)]*)")
    close_brace = re.compile(r"\s*\)")
    leaf_node = re.compile(r'\s*([^\s)]+)')

    @staticmethod
    def _scan_tree(s: str) -> Tuple[Node, int]:
        result = Tree.interior_node.match(s)
        if result != None:
            label: str = result.group(1)
            pos: int = result.end()
            children: Sequene[Node] = []
            (child, length) = Tree._scan_tree(s[pos:])
            while child != None:
                children.append(child)
                pos += length
                (child, length) = Tree._scan_tree(s[pos:])
            result = Tree.close_brace.match(s[pos:])
            if result != None:
                pos += result.end()
                return Node(label, children), pos
            else:
                return (None, 0)
        else:
            result = Tree.leaf_node.match(s)
            if result != None:
                pos = result.end()
                label = result.group(1)
                return (Node(label,[]), pos)
            else:
                return (None, 0)

    @staticmethod
    def from_str(s: str) -> Type["Tree"]:
        s = s.strip()
        (tree, n) = Tree._scan_tree(s)
        return Tree(tree)

    def bottomup(self: Type["Tree"]) -> Iterable[Node]:
        """ Traverse the nodes of the tree bottom-up. """
        return self.root.bottomup()

    def leaves(self: Type["Tree"]) -> Iterable[Node]:
        """ Traverse the leaf nodes of the tree. """
        return self.root.leaves()

    def remove_empty(self: Type["Tree"]) -> None:
        """ Remove empty nodes. """
        nodes: Sequence[Node] = list(self.bottomup())
        for node in nodes:
            if node.label == '-NONE-':
                try:
                    node.delete_clean()
                except RootDeletedException:
                    self.root = None

    def remove_unit(self: Type["Tree"]) -> None:
        """ Remove unary nodes by fusing them with their parents. """
        nodes: Sequence[Node] = list(self.bottomup())
        for node in nodes:
            if len(node.children) == 1:
                child: Node = node.children[0]
                if len(child.children) > 0:
                    node.label = "%s_%s" % (node.label, child.label)
                    child.detach()
                    for grandchild in list(child.children):
                        node.append_child(grandchild)

    def restore_unit(self: Type["Tree"]) -> None:
        """ Restore the unary nodes that were removed by remove_unit(). """
        if self.root is None: return

        def visit(node: Node) -> Node:
            children: Sequence[Node] = [visit(child) for child in node.children]
            labels: Sequene[str] = node.label.split('_')
            node: Node = Node(labels[-1], children)
            for label in reversed(labels[:-1]):
                node = Node(label, [node])
            return node

        self.root = visit(self.root)

    def binarize_right(self: Type["Tree"]) -> None:
        """ Binarize into a right-branching structure. """
        nodes: Sequence[Node] = list(self.bottomup())
        for node in nodes:
            if len(node.children) > 2:
                # create a right-branching structure
                children: Sequence[Node] = list(node.children)
                children.reverse()

                vlabel: str = node.label+"*"
                prev: Node = children[0]
                for child in children[1:-1]:
                    prev = Node(vlabel, [child, prev])

                node.append_child(prev)

    def binarize_left(self: Type["Tree"]) -> None:
        """ Binarize into a left-branching structure. """
        nodes: Sequence[Node] = list(self.bottomup())
        for node in nodes:
            if len(node.children) > 2:
                vlabel: str = node.label+"*"
                children: Sequence[Node] = list(node.children)

                prev: Node = children[0]
                for child in children[1:-1]:
                    prev = Node(vlabel, [prev, child])

                node.insert_child(0, prev)

    def binarize(self: Type["Tree"]) -> None:
        """ Binarize into a left-branching or right-branching structure
        using linguistically motivated heuristics. Currently, the heuristic
        is extremely simple: SQ is right-branching, everything else is left-branching. """
        nodes: Sequence[Node] = list(self.bottomup())
        for node in nodes:
            if len(node.children) > 2:
                if node.label in ['SQ']:
                    # create a right-branching structure
                    children: Sequence[Node] = list(node.children)
                    children.reverse()

                    vlabel: str = node.label+"*"
                    prev: Node = children[0]
                    for child in children[1:-1]:
                        prev = Node(vlabel, [child, prev])

                    node.append_child(prev)
                else:
                    # create a left-branching structure
                    vlabel: str = node.label+"*"
                    children: Sequence[Node] = list(node.children)

                    prev: Node = children[0]
                    for child in children[1:-1]:
                        prev = Node(vlabel, [prev, child])

                    node.insert_child(0, prev)

    def unbinarize(self: Type["Tree"]) -> Sequence[Node]:
        """ Undo binarization by removing any nodes ending with *. """
        if self.root is None: return

        def visit(node: Node) -> Sequence[Node]:
            children: Sequence[Node] = sum([visit(child) for child in node.children], [])
            if node.label.endswith('*'):
                return children
            else:
                return [Node(node.label, children)]

        roots: Sequence[Node] = visit(self.root)
        assert len(roots) == 1

        self.root = roots[0]

if __name__ == "__main__":
    for line in sys.stdin:
        t = Tree.from_str(line)
        print(t)
        
