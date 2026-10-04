"""tamias-router: local, metadata-only model/effort routing for Tamias.

This package currently provides configuration loading, validation, and a CLI.
There is no proxy yet: `tamias-router start` reports that and exits.

Rules that govern this package are in `router/AGENTS.md`.
"""
from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"