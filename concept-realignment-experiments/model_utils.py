import sys

sys.path.insert(0, '/home/bethge/bkr046/CBM-intervention-concept-correction')
sys.path.insert(0, '/home/bethge/bkr046/CBM-intervention-concept-correction/experiments')

from cem.train.training import train_independent_and_sequential_model, train_model
from experiment_utils import get_mnist_extractor_arch

def load_ind_seq_models(params):
    if params['config']['dataset'] == 'mnist_add':
        num_operands = params['config'].get('num_operands', 32)
        params['config']['c_extractor_arch'] = get_mnist_extractor_arch(
        input_shape=(
            params['config'].get('batch_size', 512),
            num_operands,
            28,
            28,
        ),
        num_operands=num_operands,
        )

    return train_independent_and_sequential_model(**params)


def load_intcem_model(params):
    if params['config']['dataset'] == 'mnist_add':
        num_operands = params['config'].get('num_operands', 32)
        params['config']['c_extractor_arch'] = get_mnist_extractor_arch(
        input_shape=(
            params['config'].get('batch_size', 512),
            num_operands,
            28,
            28,
        ),
        num_operands=num_operands,
        )

    return train_model(**params)