import flwr as fl
from client import client_fn

def weighted_average(metrics):
    """Aggregates accuracy from all hospitals by weighting them by dataset size."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

# 1. Define the strategy with the new aggregation function
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
    evaluate_metrics_aggregation_fn=weighted_average,
)

print("Starting Heart Disease Federated Learning Simulation...")

# 2. Start the Simulation
fl.simulation.run_simulation(
    server_app=fl.server.ServerApp(
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy
    ),
    client_app=fl.client.ClientApp(client_fn=client_fn),
    num_supernodes=3,
    backend_config={"num_cpus": 1, "num_gpus": 0},
)