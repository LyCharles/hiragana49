import torch
from torch import optim, nn
from data_processing import get_data_loaders
from model import SpinalVGG
from optuna_tuning import run_optuna


def train_with_best_params():
    params = run_optuna()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpinalVGG(num_classes=49).to(device)
    optimizer = getattr(optim, params['optimizer'])(model.parameters(), lr=params['lr'])
    train_loader, val_loader = get_data_loaders()

    # Training loop and validation (refer to your code)

    torch.save(model.state_dict(), "best_spinalvgg_model.pth")
    print("Model saved as best_spinalvgg_model.pth")
