# python -m scripts.sim_demo

from simulator.simulator import Simulator

if __name__ == "__main__":
    sim = Simulator(start_date="2025-09-15", end_date="2025-10-01")
    results = sim.run()
    summary_df = sim.report()
    print(summary_df)