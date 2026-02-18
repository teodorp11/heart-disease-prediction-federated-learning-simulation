import flwr as fl
import numpy as np
from client import client_fn
from sklearn.linear_model import LogisticRegression
from utils import get_model_params

def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

# Initialize Global Weights
model = LogisticRegression()
model.coef_ = np.zeros((1, 14))
model.intercept_ = np.zeros((1,))
initial_params = fl.common.ndarrays_to_parameters(get_model_params(model))

class AestheticStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        print(f"\nFEDERATED LEARNING ROUND: {server_round}")
        
        # Extract and print details for each hospital
        for _, fit_res in results:
            h_id = fit_res.metrics["hospital_id"]
            acc = fit_res.metrics["accuracy"]
            l_loss = fit_res.metrics["loss"]
            num = fit_res.num_examples
            print(f"Hospital {h_id:0>2} | Local Acc: {acc:.2%} | Loc. Loss: {l_loss:.4f} | Records: {num}\n")
            
        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        if metrics:
            print(f"\nGLOBAL MODEL UPDATE")
            print(f"Combined Accuracy: {metrics['accuracy']:.2%}")
            print(f"Combined Loss: {loss:.4f}\n")
        return loss, metrics

strategy = AestheticStrategy(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_available_clients=3,
    evaluate_metrics_aggregation_fn=weighted_average,
    fit_metrics_aggregation_fn=weighted_average,
    initial_parameters=initial_params,
)

fl.simulation.run_simulation(
    server_app=fl.server.ServerApp(config=fl.server.ServerConfig(num_rounds=5), strategy=strategy),
    client_app=fl.client.ClientApp(client_fn=client_fn),
    num_supernodes=3,
    backend_config={"num_cpus": 1, "num_gpus": 0},
)