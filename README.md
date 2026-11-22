# gabon-fiber-mule-dtn-sim
# Probabilistic Modeling of Hybrid Fiber–Mule Delivery Architecture (Gabon Case Study)
# Вероятностное моделирование информационного процесса доставки в гибридной архитектуре «волокно–мул»

Simulation and exact Pareto optimization scripts in Python (NumPy, Matplotlib) reproducing all tables and figures from the paper:
**R. S. Bangadi Mondjo, Yu. N. Filippovich** (Moscow Polytechnic University).

## Repository Structure / Структура репозитория
- `figures1,2,5,8_B.py` — Monte Carlo simulation of sequential & parallel modes, latency CDF, plateau analysis, fiber outage response, and tornado sensitivity.
- `figure3.py` — Decision space $(k, m)$ heatmap and Pareto front.
- `figure4.py` — Two-axis diversification and common-cause correlation $\rho$ analysis.
- `figure6.py` — Critical road blockage threshold $q^*(k)$ with bootstrap confidence intervals.
- `figure7.py` — Exact Pareto front optimization for the 5 Gabonese agricultural nodes: Tchibanga, Mayoumba, Makokou, Mandji, Okondja.

## Requirements / Требования
- Python 3.9+
- `numpy`, `matplotlib`
