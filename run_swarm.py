import os
import sys

# Ensure Python can import mnq_backtester properly
sys.path.insert(0, os.path.abspath("."))

from mnq_backtester.core.swarm_optimizer import AstParameterExtractor, AgenticSwarmOptimizer
from mnq_backtester.strategies.example_boswaves import BOSWavesGravityStrategy

def main():
    print("Agentic Swarm Backtest Optimizer Initiated...")
    
    # 1. Discover parameters from file
    strategy_path = "mnq_backtester/strategies/example_boswaves.py"
    print(f"Scanning {strategy_path} for tunable parameters...")
    base_params = AstParameterExtractor.extract_from_file(strategy_path)
    print(f"Discovered Base Parameters: {base_params}")
    
    # 2. Initialize Swarm
    swarm = AgenticSwarmOptimizer(
        strategy_class=BOSWavesGravityStrategy,
        base_params=base_params,
        iterations=5,
        population_size=10
    )
    
    # 3. Evolve (Self-Improving Loop)
    swarm.evolve()
    print("Swarm Optimization Phase 1 completed successfully.")

if __name__ == "__main__":
    main()
