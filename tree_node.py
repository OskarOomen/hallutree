"""
Tree node implementation for RAGAtomics system.
"""

from typing import List, Optional
from enum import Enum


class NodeType(Enum):
    """Enumeration of node types."""
    ROOT = "ROOT"
    EXTRACTIVE = "EXTRACTIVE"
    INFERENTIAL = "INFERENTIAL"
    EVIDENCE = "EVIDENCE"


class TreeNode:
    """Tree node with type, information, children, and parent reference."""
    
    def __init__(self, node_type: NodeType, info: str = "", parent: Optional['TreeNode'] = None):
        self.node_type = node_type
        self.info = info
        self.children: List[TreeNode] = []
        self.parent = parent
        self.supported: Optional[bool] = None
    
    def add_child(self, child: 'TreeNode') -> 'TreeNode':
        """Add a child node."""
        child.parent = self
        self.children.append(child)
        return child
    
    def remove_child(self, child: 'TreeNode') -> bool:
        """Remove a child node."""
        if child in self.children:
            child.parent = None
            self.children.remove(child)
            return True
        return False
    
    def get_root(self) -> 'TreeNode':
        """Get the root node of the tree."""
        current = self
        while current.parent:
            current = current.parent
        return current
    
    def get_descendants(self) -> List['TreeNode']:
        """Get all descendant nodes."""
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_descendants())
        return descendants
    
    def find_by_type(self, node_type: NodeType) -> List['TreeNode']:
        """Find all nodes of a specific type in the subtree."""
        results = []
        if self.node_type == node_type:
            results.append(self)
        for child in self.children:
            results.extend(child.find_by_type(node_type))
        return results
    
    def __str__(self) -> str:
        return f"TreeNode(type={self.node_type.value}, info='{self.info}', children={len(self.children)})"
    
    def __repr__(self) -> str:
        return self.__str__() 