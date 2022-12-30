import torch as th

class Controller(th.nn.Module):

    '''
    Module that emits control signals for the given control problem.
    It uses an mlp for the control network.
    '''

    def __init__(self, input_size: int, output_size: int, network_size = [256, 256]):

        super(Controller, self).__init__()

        # build mlp network;

        num_layer = len(network_size)

        assert num_layer > 0, ""

        layer = []

        layer.append(th.nn.Linear(input_size, network_size[0]))
        layer.append(th.nn.Tanh())

        for i in range(num_layer - 1):

            layer.append(th.nn.Linear(network_size[i], network_size[i + 1]))
            layer.append(th.nn.Tanh())

        layer.append(th.nn.Linear(network_size[-1], output_size))

        self.network = th.nn.Sequential(*layer)

    def forward(self, obs: th.Tensor):

        return self.network(obs)