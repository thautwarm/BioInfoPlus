from itertools import product
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data

Tanh = torch.nn.Tanh()
MaxPool = torch.nn.MaxPool2d(2, stride=2)
ReLU = torch.nn.ReLU()
MSELoss = torch.nn.MSELoss()
L1 = torch.nn.L1Loss()
Sigmoid = torch.nn.Sigmoid()
MatMul = torch.matmul
CrossEntropy = torch.nn.CrossEntropyLoss()


def Var(v):
    return torch.autograd.Variable(v, requires_grad=True)


def Val(v):
    return torch.autograd.Variable(v, requires_grad=False)


def mlp(*layer_sizes):
    if len(layer_sizes) < 2:
        raise ValueError

    params = []
    biases = []

    for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
        param = Var(torch.rand(in_size, out_size)).cuda()
        bias = Var(torch.rand(1, out_size)).cuda()
        params.append(torch.nn.Parameter(param))
        biases.append(torch.nn.Parameter(bias))

    def application(inp):
        for param, bias in zip(params, biases):
            inp = F.dropout(F.relu(inp @ param - bias))
        return inp

    param_lst = []
    for param, bias in zip(params, biases):
        param_lst.append(param)
        param_lst.append(bias)

    return torch.nn.ParameterList(param_lst), application


class RegularNet(torch.nn.Module):
    def __init__(self, params, forward):
        super().__init__()
        self.params = params
        self._forward = forward

    def forward(self, x):
        return self._forward(x)

    def fit(self,
            X,
            y,
            opt_f,
            loss_f,
            *,
            batch_size=50,
            epoch=100,
            verbose=False):

        data_size = len(X)
        assert len(y) == data_size, ValueError

        optimizer: torch.optim.Optimizer = opt_f(self.params)

        X = Val(X)
        y = Val(y)

        subset = data.SubsetRandomSampler(range(data_size))
        sampling = data.BatchSampler(subset, batch_size, drop_last=False)

        for epoch in range(epoch):

            for tr_idx in sampling:

                optimizer.zero_grad()
                each_x = X[tr_idx]
                each_y = y[tr_idx]
                loss = loss_f(self(each_x), each_y)
                loss.backward()
                optimizer.step()

            if verbose:
                print('epoch {epoch}, loss: {loss}'.format(
                    epoch=epoch, loss=loss.data))

    def predict(self, X):
        return self(Val(X)).data


def one_hot_without_undecided(target, species: int):
    ret = torch.FloatTensor(len(target), species)
    ret.zero_()
    ret.scatter_(1, target.view(-1, 1), 1)
    return torch.cat(list(ret))


def one_hot_batch(targets, species: int):
    ret = torch.FloatTensor(len(targets), species)
    ret.zero_()
    ret.scatter_(1, targets.view(-1, 1), 1)
    return ret


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

    cells = torch.cat(cells)

    return cells


def get_penalty(param):
    norm = torch.norm(param)

    return 1 / (0.001 + norm) + norm - 2


def from_one_hot(targets):
    return torch.argmax(targets, 1)


def random_mutate(each):
    random.shuffle(each)
    return each


def random_noise(shape):
    return (torch.rand(shape).cuda() - 0.5) / 50.0


def generate_mutations(seqs, expand=3):
    for each in range(expand):
        mutated_seqs = [list(seq) for seq in seqs]
        for each in mutated_seqs:
            random_mutate(each)
        yield from mutated_seqs


def generate_prospective(seq, species):
    undecided = [i for i, each in enumerate(seq) if each is None]
    prospectives = product(*[range(species) for _ in range(len(undecided))])
    for prospective in prospectives:
        solu = list(seq)
        for i, loc in enumerate(undecided):
            solu[loc] = prospective[i]
        yield solu


class Solver(torch.nn.Module):
    def __init__(self, n_input: int, primary_species: int,
                 secondary_species: int):
        super().__init__()

        self.secondary_species = secondary_species
        self.primary_species = primary_species
        width = (secondary_species + primary_species) * n_input
        self.constraint_params, self._forward = mlp(width, width, 1)
        self.params = None

        inp_size = primary_species * n_input
        self.net = RegularNet(*mlp(inp_size, 2 * inp_size, 1)).cuda()
        self.cuda()

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

        variables = self.make_variables(primary_inputs, secondary_inputs)

        targets = torch.zeros(len(variables))

        correct = list(zip(primary_inputs, secondary_inputs))

        primary_inputs = list(generate_mutations(primary_inputs))
        secondary_inputs = list(generate_mutations(secondary_inputs))

        mutated = list(zip(primary_inputs, secondary_inputs))

        new_targets = torch.FloatTensor(
            [float(each not in correct)
             for each in mutated]) * torch.from_numpy(
                 np.random.randint(5, 10, len(mutated)).astype(np.float32))
        variables = Val(
            torch.cat([
                variables,
                self.make_variables(primary_inputs, secondary_inputs)
            ], 0).cuda())

        targets = Val(torch.cat([targets, new_targets]).cuda().view(-1, 1))

        optimizer = torch.optim.Adam(self.constraint_params, lr=lr)

        for epoch in range(epoch):
            optimizer.zero_grad()
            prediction = self(variables) - targets

            
            loss = torch.sum(
                torch.abs(prediction + random_noise(prediction.shape))**2)

            if verbose:
                print(f'epoch: {epoch}, loss: {loss}')

            loss.backward()

            optimizer.step()

    def see(self, primary_input, secondary_input):
        primary_input = list(primary_input)
        secondary_input = list(secondary_input)

        primary_params = {}
        secondary_params = {}

        primary = one_hot_with_undecided(primary_input, self.primary_species,
                                         primary_params)

        secondary = one_hot_with_undecided(
            secondary_input, self.secondary_species, secondary_params)

        variables = torch.cat([primary, secondary]).cuda()

        return self(variables).data

    def resolve_variable(self, primary_input, secondary_input):
        primary_inputs, secondary_inputs = zip(*product(
            generate_prospective(primary_input, self.primary_species),
            generate_prospective(secondary_input, self.secondary_species)))

        variables = self.make_variables(primary_inputs,
                                        secondary_inputs).cuda()
        results = self(variables)

        idx = torch.argmin(results).tolist()
        return primary_inputs[idx], secondary_inputs[idx]
