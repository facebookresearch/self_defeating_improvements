import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(512, num_classes)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), 2, 2))
        x = F.relu(F.max_pool2d(self.bn2(self.conv2(x)), 2, 2))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
    
    
class MLP(nn.Module):
    def __init__(self, in_features, num_classes, hidden_layers=[]):
        super(MLP, self).__init__()
        width = [in_features] + hidden_layers + [num_classes]
        self.layers = nn.ModuleList([nn.Linear(width[i], width[i + 1]) for i in range(len(width) - 2)])
        self.output_layer = nn.Linear(width[-2], width[-1])
        
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output_layer(x)