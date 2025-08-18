import torch
import pandas as pd
from data_processing import preprocess_data
from dataset import MyDataset
from models import CardioPredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict():
    model = CardioPredictor().to(device)
    model.load_state_dict(torch.load('cardio_model.pth'))
    model.eval()

    data = pd.read_csv('your_test_data.csv')
    data = preprocess_data(data)
    dataset = MyDataset(data)
    predictions = []

    for i in range(len(dataset)):
        cat_data, num_data, text_data, _ = dataset[i]
        cat_data = cat_data.unsqueeze(0).to(device)
        num_data = num_data.unsqueeze(0).to(device)
        text_data = text_data.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(cat_data, num_data, text_data)
            predictions.append(output.item())

    return predictions