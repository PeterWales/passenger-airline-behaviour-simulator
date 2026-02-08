# Passenger and Airline Behavior Simulator (PABSim)

This repository (`airline-schedule-predictor`) contains the code developed as part of an undergraduate master’s final-year project in Aeronautical Engineering. **PABSim** is a prototype simulation tool for projecting the long-term evolution of passenger airline networks under high-level economic and technological scenarios. The tool integrates an econometric passenger demand model with an agent-based airline behaviour model to generate annualised projections of variables indicating the state of the airline network.

---

## Overview

PABSim is designed to project how passenger airline networks may evolve in response to changes in factors including:
- Aircraft performance characteristics
- Fuel prices
- Regional economic indicators

The simulator produces projections of:
- Airline fleet composition
- Network structure and flight frequencies
- Fares and passenger demand

---

## Academic Context

This code was developed as part of the final-year project for:

- **Degree**: MEng Aeronautical Engineering with a Year in Industry  
- **Institution**: Imperial College London
- **Project type**: Final Year Project  
- **Thesis title**: Projecting Future Airline Networks  
- **Submission date**: 2nd June 2025  

---

## Repository Contents

- Core simulation code (Python)
- Example CSV input file
- Post-processing and plotting scripts for analysing simulation outputs

---

## Requirements

- **Language**: Python  
- **Dependencies**: Listed in `requirements.txt`

No specific hardware is required, but realistic simulations are computationally intensive and were primarily run on a high-performance computing (HPC) cluster.

---

## Running Simulations

### Execution model

Simulations are typically run by executing the main simulation entry point via a Slurm `sbatch` script. Runtime configuration is handled through an input CSV file.

### Input data availability

The simulator depends on large, proprietary or restricted datasets which cannot be redistributed openly.

As a result:
- This repository cannot be run out-of-the-box
- Users wishing to run simulations must either:
  - Reconstruct equivalent datasets independently, or
  - Obtain access to the original data through appropriate academic channels

---

## Project Status and Limitations

Status: Research prototype

Important notes:
- The codebase prioritises research functionality over software engineering quality
- Performance is poor for large scenarios
- The implementation is not optimised or refactored
- The project is not actively maintained

That said, the repository may still be useful as a basis for reimplementation or optimisation as part of further research.

---

## Citation

If you use or reference this work in academic research, please cite:

Peter Wales, Projecting Future Airline Networks,  
MEng Final Year Project Thesis, Imperial College London, 2025.

---
