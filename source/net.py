import torch.nn as nn
import torch.nn.functional as F

class DenseNet(nn.Module):
    def __init__(self, n_input:int, n_output:int, size: list, softmax:bool, tempreture:float=1.0):
        super(DenseNet, self).__init__()
        assert len(size) > 0
        self._softmax = softmax
        self.input_layer = nn.Linear(n_input, size[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(size)-1):
            self.hidden_layers.append(nn.Linear(size[i], size[i+1]))
        self.output_layer = nn.Linear(size[-1], n_output)
        self._tempreture = tempreture

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        if self._softmax:
            x = F.softmax(x / self._tempreture, dim=-1)
        return x 
