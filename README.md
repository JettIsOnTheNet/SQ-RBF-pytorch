# SQ-RBF_pytorch
Scaled Quadratic Radial Basis (Activation) Function.

$f(x) = a * exp(-b * ||x - c||^2) + d$

```

class SQ_RBF(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(SQ_RBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_tensor):
        x = input_tensor.view(input_tensor.size(0), -1)
        x = torch.mm(x, self.weight.t())
        x = x + self.bias
        x = x.pow(2)
        x = -x
        x = torch.exp(x)
        return x

```

```

class SQ_RBF(torch.nn.Module):
    def __init__(self, a, b, c, d):
        super(SQ_RBF, self).__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        
    def forward(self, x):
        distance = torch.norm(x - self.c, dim=-1)
        return self.a * torch.exp(-self.b * distance**2) + self.d

```