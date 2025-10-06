# SSUAP: Small-Sample Uncertainty Assessment Protocol

Implementation of sample-size adjusted confidence thresholds for multi-criteria environmental assessment under institutional sampling constraints (n<20).

## Reference

This repository contains code supporting:

Title: SSUAP: A Transferable Protocol for Small-Sample Uncertainty Assessment in Multi-Criteria Environmental Evaluation

Author: Fatima Zahra Kherazi
Affiliation: Jiangsu University 
Status:Manuscript under peer review  
Year: 2025

For citation details or preprint access, contact: fatima.kherazi@stmail.ujs.edu.cn

This page will be updated with publication details upon acceptance.

## Overview

Environmental management authorities often face small-sample constraints (n=5-15 sub-basins, protected areas, or administrative units) imposed by institutional boundaries. Classical statistical methods assuming large samples (n>30) fail in these contexts.

SSUAP provides:
- Sample-size adjusted thresholds: Formulas calculating confidence thresholds as functions of sample size (n=3 to n=20)
- Realistic ensemble scenarios: Six-scenario design spanning decision-relevant weight uncertainty without requiring probability distributions
- Confidence stratification: HIGH/MODERATE/LOW classifications enabling adaptive implementation protocols

## Installation

Requires Python 3.8+ with standard packages:
```bash
pip install numpy pandas
