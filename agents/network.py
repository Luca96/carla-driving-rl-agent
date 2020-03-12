"""Provides various networks specifications (inputs, layers, ...) compatible with TensorForce"""


def _input(names: [str]):
    return dict(type='input', names=names)


def _output(name: str):
    return dict(type='output', name=name)


def _embedding(size: int, activation='tanh', **kwargs):
    return dict(type='embedding', size=size, activation=activation, **kwargs)


def _conv2d(size: int, window: (int, int), stride=1, activation='relu', **kwargs):
    return dict(type='conv2d', size=size, window=window, stride=stride, activation=activation, **kwargs)


def _dense(size: int, dropout=0.0, activation='relu', **kwargs):
    return dict(type='dense', size=size, dropout=dropout, activation=activation, **kwargs)


def _pooling(reduction, window=2, stride=2, **kwargs):
    return dict(type='pool2d', reduction=reduction, window=window, stride=stride, **kwargs)


def _instance_norm():
    return dict(type='instance_normalization')


def _flatten():
    return dict(type='flatten')


def _global_avg_pool():
    return dict(type='keras', name='GlobalAveragePooling2D')


def convolutional_network(in_names, out_name, initial_filters=64, blocks=(1, 4), dropout=0.0, flatten=False):
    """Returns a list of dictionaries representing the network specification.
        @:arg blocks: a tuple representing the number of dilated-convolutions with max-pooling,
                      and then conv2d with stride.
        @:arg flatten: if False, GlobalAveragePooling2D is used instead of a Flatten layer.
    """
    network = [_input(names=in_names)]

    for i in range(1, blocks[0] + 1):
        filters = initial_filters * i
        network.append(_conv2d(size=filters, window=(3, 3), dilation=2, dropout=dropout))
        network.append(_instance_norm())
        network.append(_pooling(reduction='max'))

    for i in range(1, blocks[1] + 1):
        filters = initial_filters * (i + blocks[0])
        network.append(_conv2d(size=filters, window=(3, 3), stride=2, dropout=dropout))
        network.append(_instance_norm())

    if flatten:
        network.append(_flatten())
    else:
        network.append(_global_avg_pool())

    network.append(_output(name=out_name))
    return network


def dense_network(in_names, out_name, units=64, blocks=4, dropout=0.0, activation='relu', normalize=True):
    network = [_input(names=in_names),
               _embedding(size=32, dropout=dropout, activation=activation)]  # TODO: consider max_norm parameter

    for i in range(blocks):
        network.append(_dense(units, dropout, activation))

        if normalize:
            network.append(_instance_norm())

    network.append(_output(name=out_name))
    return network


def recurrent_network(input_names, aggregation='concat', recurrent_units=128, layers=2):  # TODO: size parameter
    # network = [dict(type='input',
    #                 names=input_names,
    #                 aggregation_type=aggregation)]
    network= []

    for i in range(layers):
        network.append(dict(type='internal_gru',
                            size=recurrent_units))

    return network


def print_network(network_spec, tabs=''):
    for x in network_spec:
        if isinstance(x, list):
            print_network(x, tabs=tabs + '  ')
        else:
            print(tabs + str(x))


baseline = [
    # Image
    convolutional_network(in_names=['image'], out_name='image_out'),

    # Vehicle features
    dense_network(in_names=['vehicle_features'], out_name='vehicle_out'),

    # Road features
    dense_network(in_names=['road_features'], out_name='road_out'),

    # Previous Actions
    dense_network(in_names=['previous_actions'], out_name='actions_out'),

    # Concat inputs + add recurrence:
    recurrent_network(input_names=['image_out', 'vehicle_out', 'road_out', 'actions_out'])
]
