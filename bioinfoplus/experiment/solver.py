import torch
import torch.nn.functional as F

Tanh = torch.nn.Tanh()
MaxPool = torch.nn.MaxPool2d(2, stride=2)
ReLU = torch.nn.ReLU()
MSELoss = torch.nn.MSELoss()
L1 = torch.nn.L1Loss()
Sigmoid = torch.nn.Sigmoid()
MatMul = torch.matmul
CrossEntropy = torch.nn.CrossEntropyLoss()


def Var(v):
    """pytorch梯度计算时作为**可变**的节点
    """
    return torch.autograd.Variable(v, requires_grad=True)


def Val(v):
    """pytorch梯度计算时作为**不可变**的节点
    """
    return torch.autograd.Variable(v, requires_grad=False)


def mlp(*layer_sizes):
    if len(layer_sizes) < 2:
        raise ValueError

    params = []
    biases = []

    for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
        param = Var(torch.rand(in_size, out_size))
        bias = Var(torch.rand(1, out_size))
        params.append(torch.nn.Parameter(param))
        biases.append(torch.nn.Parameter(bias))

    def application(inp):
        for param, bias in zip(params, biases):
            inp = F.relu(inp @ param) - bias
        return inp

    param_lst = []
    for param, bias in zip(params, biases):
        param_lst.append(param)
        param_lst.append(bias)

    return torch.nn.ParameterList(param_lst), application


def one_hot_without_undecided(target, species: int):
    ret = torch.FloatTensor(len(target), species)
    ret.zero_()
    ret.scatter_(1, target.view(-1, 1), 1)
    return torch.cat(list(ret))


def one_hot_with_undecided(target, species: int, undecided: dict):
    cells = []
    for idx, cell in enumerate(target):
        if cell is None:
            param = undecided.get(idx, None)
            if param is None:
                param = Var(torch.rand(species))
                undecided[idx] = param

            cells.append(param)

        else:
            cell_val = torch.zeros(species)
            cell_val[cell] = 1.0
            cells.append(Var(cell_val))

    return torch.cat(cells)


def from_one_hot(targets):
    return torch.argmax(targets, 1)


class Solver(torch.nn.Module):
    def __init__(self, n_input: int, primary_species: int,
                 secondary_species: int):
        super().__init__()

        self.secondary_species = secondary_species
        self.primary_species = primary_species
        width = (secondary_species + primary_species) * n_input
        self.constraint_params, self._forward = mlp(width, 2 * width + 1, 1)
        self.params = None

    def make_variables(self, primary_inputs, secondary_inputs):
        return torch.stack([
            torch.cat([
                one_hot_without_undecided(
                    torch.LongTensor(primary_input), self.primary_species),
                one_hot_without_undecided(
                    torch.LongTensor(secondary_input), self.secondary_species)
            ]) for primary_input, secondary_input in zip(
                primary_inputs, secondary_inputs)
        ], 0)

    def forward(self, x):
        return self._forward(x)

    def tuning_constraint(self,
                          primary_inputs,
                          secondary_inputs,
                          *,
                          epoch=100,
                          lr=0.01,
                          verbose=True):
        variables = Val(self.make_variables(primary_inputs, secondary_inputs))
        optimizer = torch.optim.Adam(self.constraint_params, lr=lr)
        for epoch in range(epoch):
            optimizer.zero_grad()

            target = self(variables)
            loss = torch.sum(torch.abs(target))
            if verbose:
                print(f'epoch: {epoch}, loss: {loss}')
            loss.backward()
            optimizer.step()

    def resolve_variable(self,
                         primary_input,
                         secondary_input,
                         *,
                         epoch=100,
                         lr=0.0001,
                         verbose=True):

        primary_input = list(primary_input)
        secondary_input = list(secondary_input)

        primary_params = {}
        secondary_params = {}

        for epoch in range(epoch):
            primary = one_hot_with_undecided(
                primary_input, self.primary_species, primary_params)
            secondary = one_hot_with_undecided(
                secondary_input, self.secondary_species, secondary_params)
            variables = torch.cat([primary, secondary])
            target = self(variables)
            loss = torch.abs(target)
            if verbose:
                print(f'epoch: {epoch}, loss: {loss}')
            params = (*primary_params.values(), *secondary_params.values())
            for param, gd in zip(params, torch.autograd.grad(loss, params)):
                param.data.sub_(lr, gd)

        for k, v in primary_params.items():
            primary_input[k] = torch.argmax(v).tolist()
        for k, v in secondary_params.items():
            secondary_input[k] = torch.argmax(v).tolist()

        return primary_input, secondary_input
