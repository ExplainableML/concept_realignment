import torch
import torch.nn as nn
from torchvision.models import resnet34

class BackboneWithExtraLayers(nn.Module):
    def __init__(self, backbone_name, output_dim, num_hidden_layers=0, pretrained=True):
        super(BackboneWithExtraLayers, self).__init__()
        
        self.backbone_name = backbone_name
        self.num_hidden_layers = num_hidden_layers
        self.pretrained = pretrained
        self.output_dim = output_dim

        print(self.backbone_name)
        print(self.num_hidden_layers)
        print(self.pretrained)
        print(self.output_dim)

        if self.backbone_name == 'resnet34':
            self.backbone = resnet34(pretrained=self.pretrained)
            if self.output_dim is not None:
                self.backbone.fc = torch.nn.Linear(512, self.output_dim)
        else:
            raise Exception(f"Backbone {self.backbone_name} not supported")

        if self.num_hidden_layers > 0 and self.output_dim is not None:
            self.extra_layers = nn.ModuleList()
            self.extra_layers.extend([nn.Linear(self.output_dim, self.output_dim) for _ in range(self.num_hidden_layers)])

            # Batch Normalization for each hidden layer
            self.batch_norms = nn.ModuleList([nn.BatchNorm1d(self.output_dim) for _ in range(self.num_hidden_layers)])

        print(f"C_EXTRACTOR IS BACKBONE + EXTRA LAYERS. BACKBONE = {self.backbone_name}. NUM_HIDDEN_LAYERS = {self.num_hidden_layers}. PRETRAINED = {self.pretrained}, OUTPUT_DIM = {self.output_dim}.")


    def forward(self, x):
        x = self.backbone(x)

        if self.num_hidden_layers > 0:
            for i in range(self.num_hidden_layers):
                x = self.batch_norms[i](x)
                x = torch.relu(x)
                x = self.extra_layers[i](x)

        return x

    # def to(self, device):
    #     self.backbone = self.backbone.to(device)

    #     if 
    #     for i in range(len(self.extra_layers)):
    #         self.extra_layers[i] = self.extra_layers[i].to(device)

    #     for i in range(len(self.batch_norms)):
    #         self.batch_norms[i] = self.batch_norms[i].to(device)
