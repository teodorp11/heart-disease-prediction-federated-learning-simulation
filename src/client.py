import flwr as fl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from flwr.common import Context
from utils import load_data, get_model_params, set_model_params
import warnings

warnings.filterwarnings("ignore")

class HeartDiseaseClient(fl.client.NumPyClient):
    def __init__(self, hospital_id):
        self.hospital_id = hospital_id
        self.X_train, self.y_train = load_data(hospital_id)
        self.model = LogisticRegression(penalty='l2', max_iter=1, warm_start=True)
        self.model.fit(self.X_train, self.y_train)

    def fit(self, parameters, config):
        set_model_params(self.model, parameters)
        self.model.fit(self.X_train, self.y_train)
        
        # Calculate local metrics
        accuracy = self.model.score(self.X_train, self.y_train)
        loss = log_loss(self.y_train, self.model.predict_proba(self.X_train))
        
        # Sending the data back to the server
        metrics = {
            "accuracy": accuracy,
            "hospital_id": self.hospital_id,
            "loss": loss,
            "samples": len(self.X_train)
        }
        return get_model_params(self.model), len(self.X_train), metrics

    def evaluate(self, parameters, config):
        set_model_params(self.model, parameters)
        loss = log_loss(self.y_train, self.model.predict_proba(self.X_train))
        accuracy = self.model.score(self.X_train, self.y_train)
        return loss, len(self.X_train), {"accuracy": accuracy}

def client_fn(context: Context) -> fl.client.Client:
    hospital_id = int(context.node_config["partition-id"]) + 1
    return HeartDiseaseClient(hospital_id=hospital_id).to_client()