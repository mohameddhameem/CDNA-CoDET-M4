"""
City of Agents - Code Analysis and Graph Generation Framework

A research framework for analyzing code structure, generating graph representations,
and visualizing software metrics as a "city of agents".
"""

__version__ = "0.1.0"
__author__ = "City of Agents Research Team"

# Core modules exports
from city_of_agents import ast2pyg
from city_of_agents import joern
from city_of_agents import codeclip

__all__ = [
    "ast2pyg",
    "joern", 
    "codeclip",
    "parsers",
    "builders",
    "utils",
    "simulation",
]
