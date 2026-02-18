import flwr as fl
from client import client_fn

# 1. Define the strategy (How to average the weights)
# We use FedAvg (Federated Averaging), the most common FL algorithm.
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,      # Sample 100% of available clients for training
    fraction_evaluate=1.0, # Sample 100% of available clients for evaluation
    min_fit_clients=3,     # Never start training unless all 3 hospitals are online
    min_evaluate_clients=3,
    min_available_clients=3,
)

# 2. Start the Simulation
print("Starting Heart Disease Federated Learning Simulation...")

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=3,
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)