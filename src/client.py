import flwr as fl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from flwr.common import Context # New Import
from utils import load_data, get_model_params, set_model_params
import warnings

# Ignore convergence warnings for small initial rounds
warnings.filterwarnings("ignore")

class HeartDiseaseClient(fl.client.NumPyClient):
    def __init__(self, hospital_id):
        self.hospital_id = hospital_id
        
        # Load this specific hospital's data
        self.X_train, self.y_train = load_data(hospital_id)
        
        self.model = LogisticRegression(
            penalty='l2',
            max_iter=1, # Local training happens in "rounds"
            warm_start=True # Keeps learning from previous round's weights
        )
        
        # Initial fit to define classes
        self.model.fit(self.X_train, self.y_train)

    def get_parameters(self, config):
        return get_model_params(self.model)

    def fit(self, parameters, config):
        # Update local model with weights from the Global Server
        set_model_params(self.model, parameters)
        
        # Train for one epoch (iteration)
        self.model.fit(self.X_train, self.y_train)
        print(f"Hospital {self.hospital_id} finished training.")
        return get_model_params(self.model), len(self.X_train), {}

    def evaluate(self, parameters, config):
        set_model_params(self.model, parameters)
        
        # Calculate loss (how far off the model is)
        loss = log_loss(self.y_train, self.model.predict_proba(self.X_train))
        accuracy = self.model.score(self.X_train, self.y_train)
        return loss, len(self.X_train), {"accuracy": accuracy}

def client_fn(context: Context) -> fl.client.Client:
    hospital_id = int(context.node_config["partition-id"]) + 1
    return HeartDiseaseClient(hospital_id=hospital_id).to_client()