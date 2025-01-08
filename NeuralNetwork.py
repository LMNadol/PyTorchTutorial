import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(
            self.layer_1.weight, nonlinearity="relu"
        )  # Weight initialisation
        self.layer_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))  # Activation function layer 1
        x = torch.nn.functional.sigmoid(self.layer_2(x))  # Activation function layer 2

        return x


# nn.Module = base class for all neural network modules built in PyTorch
# super = to overload
# nn.Linear = Applies an affine linear transformation to the incoming data: :math:`y = xA^T + b`.
# nn.init.kaiming_uniform_ = Fill the input `Tensor` with values using a Kaiming uniform distribution.
# torch.nn.functional.relu = Applies the rectified linear unit function element-wise.
# torch.nn.functional.sigmoid = Applies the element-wise function :math:`\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}` element-wise.
