import numpy as np
import torch
from torch import nn, optim
from src.dataset import get_testloader, load_data, create_noniid_dataset_mnist
from src.model import get_model
from src.train import train, test_model


class FedAVG():
    def __init__(self, config) -> None:
        self.config = config
        self.global_model = get_model(config).to(config["device"])
        self.client_data = load_data(config)
        self.testloader = get_testloader(config)

    def client_train(self, client_id):
        data = self.client_data[client_id]
        model = get_model(self.config).to(self.config["device"])
        model.load_state_dict(self.global_model.state_dict())
        optimizer = getattr(optim, self.config["optimizer"])(model.parameters(), lr=self.config["lr"])
        loss_function = getattr(nn, self.config["loss_function"])()

        data_loader = torch.utils.data.DataLoader(data, batch_size=self.config["batch_size"], shuffle=True)

        model, local_train_loss, local_train_accuracy, _, _, _, _ = train(model, data_loader, None, optimizer, loss_function, self.config["device"], self.config["num_epochs"])

        return model.state_dict()
    
    def aggregate(self, weights, client_sample_sizes):
        total_samples = np.sum(client_sample_sizes)
        avg_weights = {}
        for key in weights[0].keys():
            avg_weights[key] = sum(
                weights[i][key] * (client_sample_sizes[i] / total_samples)
                for i in range(len(weights)))
        return avg_weights
    
    def run(self):
        for round in range(self.config["num_rounds"]):
            client_weights = []
            client_sample_sizes = []
            selected_clients = np.random.choice(len(self.client_data), self.config["num_client_selection"], replace=False)
            for client_id in selected_clients:
                client_weights.append(self.client_train(client_id))
                client_sample_sizes.append(len(self.client_data[client_id]))
            aggregated_weights = self.aggregate(client_weights, client_sample_sizes)
            self.global_model.load_state_dict(aggregated_weights)
            print(f"Round {round + 1} completed")

            test_loss, test_accuracy, test_f1 = test_model(self.global_model,self.testloader, getattr(nn, self.config["loss_function"])(), self.config["device"])
            print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")
            print("-"*50)
        return self.global_model