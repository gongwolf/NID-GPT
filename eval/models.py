import torch
from torch import nn
import torch.nn.functional as F

class MLP_Mult(nn.Module):
    def __init__(self, input_shape, d_layers, num_classes=11, dropout_rate=0.2):
        super(MLP_Mult, self).__init__()
        
        # Define each layer dynamically based on d_layers
        layers = []
        in_features = input_shape  # Start with input shape
        
        for i, units in enumerate(d_layers):
            layers.append(nn.Linear(in_features, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = units  # Set up input for the next layer
        
        # Final output layer
        layers.append(nn.Linear(in_features, num_classes))
        
        # Combine into sequential layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return F.log_softmax(self.model(x), dim=1)
