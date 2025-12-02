# Genetic Algorithm Equation Solver

Pure Python implementation of a Real-Coded Genetic Algorithm for solving nonlinear equations and continuous optimization problems.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue)](https://numpy.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-green)](https://matplotlib.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

## Features

- Real-coded chromosomes (no binary encoding)
- Simulated Binary Crossover (SBX) + Polynomial Mutation
- Tournament selection with elitism
- Adaptive parameters & restart strategy
- Built-in support for single/multi-variable equations and benchmark functions
- Automatic convergence plotting and statistics

## Performance Highlights (from Final Report)

| Problem                  | Variables | Best Fitness       | Generations | Success Rate |
|--------------------------|-----------|--------------------|-------------|--------------|
| Circle: x² + y² = 4      | 2         | 3.1e-12            | ~120        | 100%         |
| Nonlinear System (3 eq)  | 3         | < 1e-11            | ~180        | 98%          |
| Sphere Function          | 10        | 1.8e-9             | 350         | 100%         |
| Rastrigin Function       | 5         | 0.994 (global ≈ 0) | 720         | 92%          |

## Project Documents

- [Proposal.pdf](Proposal.pdf) – Problem definition, literature review & methodology  
- [Rep.pdf](Rep.pdf) – Complete results, analysis, comparisons & conclusions (28 pages)

