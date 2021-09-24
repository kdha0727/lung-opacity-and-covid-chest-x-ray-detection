import torch.nn as nn


class Classifier(nn.Module):

    def __init__(self, backbone, num_classes, out_channels=None, dropout_rate=0.4):
        super().__init__()
        out_channels = out_channels or backbone.out_channels
        self.feature_extractor = backbone
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def get_classifier(backbone, num_classes=4):
    return Classifier(backbone, num_classes)


__all__ = ['Classifier', 'get_classifier']
