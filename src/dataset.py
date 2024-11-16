import torch    
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms



def load_data(config):
    if config["dataset"] == "MNIST":
        data = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]))
    elif config["dataset"] == "CIFAR10":
        data = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]))
    else:
        raise ValueError("Dataset not supported")
    
    if config["iid"]:
        client_data = random_split(data, [len(data) // config['num_clients'] for _ in range(config['num_clients'])])
    else:
        client_data = create_noniid_dataset_mnist(data, config['num_clients'], config['num_labels_per_client'])
    return client_data


def get_testloader(config):
    if config["dataset"] == "MNIST":
        data = datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]))
    elif config["dataset"] == "CIFAR10":
        data = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]))
    else:
        raise ValueError("Dataset not supported")
    return torch.utils.data.DataLoader(data, batch_size=config["batch_size"], shuffle=False)

def create_noniid_dataset_mnist(data, num_clients, num_labels_per_client):
    """
    Splits the dataset into non-IID subsets, where each client gets data for a fixed number of labels.
    
    Args:
        data: The dataset to be split.
        num_clients: The number of clients.
        num_labels_per_client: Number of unique labels per client.
    
    Returns:
        A list of Subset objects, one for each client.
    """
    # Group indices by labels
    label_indices = {i: [] for i in range(10)} 
    for idx, (_, label) in enumerate(data):
        label_indices[label].append(idx)
    
    # Shuffle indices within each label
    for label in label_indices:
        torch.manual_seed(42)  # For reproducibility
        torch.randperm(len(label_indices[label])).tolist()
    
    # Assign labels to clients
    clients = [[] for _ in range(num_clients)]
    label_list = list(label_indices.keys())
    
    # Divide labels across clients
    labels_per_client = len(label_list) // num_clients
    for client_id in range(num_clients):
        start_label = (client_id * labels_per_client) % len(label_list)
        client_labels = label_list[start_label:start_label + num_labels_per_client]
        
        # Add data for these labels to the client
        for label in client_labels:
            clients[client_id].extend(label_indices[label])
    
    # Create Subset objects for each client
    client_data = [Subset(data, client_indices) for client_indices in clients]
    return client_data


