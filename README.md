# TimeMachine QuantOptimizer 🚀

![TimeMachine QuantOptimizer Dashboard](assets/dashboard_main.jpg)

**TimeMachine QuantOptimizer** is a cutting-edge algorithmic backtesting and model optimization framework. Built to handle complex quantitative analysis, this environment provides researchers and traders with the tools to simulate, optimize, and validate quantitative trading models at high fidelity.

---

## 🌟 Key Features

*   **Advanced Micro-Tick Engine**: Simulates high-resolution market microstructure including next-bar open fill lag and deep liquidity profiles.
*   **Robust Strategy Library (StockSharp)**: Ships with 10 generic, highly-optimized StockSharp models for out-of-the-box benchmarking (MA Crossover, Breakouts, Mean Reversion, etc.).
*   **Monte Carlo Stress Testing**: Rigorous resampling to test strategy durability across thousands of simulated alternate realities, producing statistically significant confidence intervals.
*   **Parameter Optimization Engine**: Multidimensional hyperparameter searches utilizing walk-forward analysis and custom risk-adjusted scoring (Deflated Sharpe, Sortino, Calmar).
*   **Interactive Analytics Dashboard**: A full suite of visualizations, heatmaps, and performance summaries accessible locally.

---

## 📈 Parameter Optimization & Analytics

Optimize your systems over multi-year datasets, navigating parameter space efficiently to discover robust strategy configurations.

![Strategy Optimization Surface](assets/strategy_optimization.jpg)
*3D representation of parameter surface optimization mapping Sharpe Ratio across two moving average dimensions.*

## ⚙️ Architecture

The system is separated into distinct, modular components designed for high-throughput quantitative workloads:

*   **`core/`**: The heart of the simulation, featuring the `BacktestEngine`, event-driven execution models, and advanced order types.
*   **`analytics/`**: Includes Deflated Sharpe calculations, Hierarchical Risk Parity, factor premia analysis, and structural turbulence diagnostics.
*   **`monte_carlo/`**: Stress-testing utilities using bootstrapping and resampling techniques.
*   **`data/`**: Continuous futures data handling, session filtering, and validation.
*   **`pinescript/`**: Utilities to interface with, parse, and benchmark TradingView PineScript models natively.

## 🚀 Getting Started

### Prerequisites

*   Python 3.9+
*   Virtual Environment (recommended)

### Installation

Clone the repository and run the setup:

```bash
git clone https://github.com/simon-hawk/TimeMachine-QuantOptimizer.git
cd TimeMachine-QuantOptimizer
pip install -r requirements.txt
```

### Running Your First Backtest

Execute the CLI utility to run the pre-loaded generic strategies:

```bash
python -m mnq_backtester.cli --strategy ss_0001_0001_ma_crossover --data data/nq_f_10y.csv --evaluate
```

### Launching the Dashboard

Fire up the local interactive analytics web server:

```bash
python -m mnq_backtester.dashboard.server --port 8050
```

*Navigate to `http://localhost:8050` in your web browser to view your strategy analysis.*

## 🛡️ Security & Privacy Notice

This repository contains the generalized, open-source framework of the QuantOptimizer platform. Proprietary algorithms, private API keys, and internal account configurations have been explicitly excluded from this public distribution.

---

*Designed for high-performance quantitative research and systematic trading development.*
