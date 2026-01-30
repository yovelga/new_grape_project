"""
Feature Selection Module for Hyperspectral Channel Reduction.

This module provides tools for reducing hyperspectral channels from 150 to an optimal
subset using SHAP-based ranking and RFECV-based active reduction.

Main Components:
- feature_selection_pipeline.py: Complete two-step feature selection pipeline
"""

from pathlib import Path

MODULE_DIR = Path(__file__).parent
