#!/usr/bin/env python
# This script loads trained models and runs them on train and test data, saves model predictions in pickle files

import sys
import os
from pathlib import Path

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Get the directory of the current file
current_file_dir = os.path.dirname(current_file_path)

# Add the path to the directory containing your script to sys.path
sys.path.insert(0, os.path.join(current_file_dir, '..'))
sys.path.insert(0, os.path.join(current_file_dir, '..', 'experiments'))
sys.path.insert(0, os.path.join(current_file_dir, '..', 'concept-realignment-experiments'))


# Load Configs of the model you want to train and store predictions of
config_file = os.path.join(current_file_dir, "..", "experiments", "configs", "cub.yaml")


import argparse
import copy
import joblib
import json
import logging
import numpy as np
import os
import sys
import torch
import yaml
from tqdm import tqdm
import pickle

from datetime import datetime
from pathlib import Path
from pytorch_lightning import seed_everything

from cem.data.synthetic_loaders import (
    get_synthetic_data_loader, get_synthetic_num_features
)
import cem.data.celeba_loader as celeba_data_module
import cem.data.chexpert_loader as chexpert_data_module
import cem.data.CUB200.cub_loader as cub_data_module
import cem.data.derm_loader as derm_data_module
import cem.data.mnist_add as mnist_data_module
import cem.data.awa2_loader as awa2_data_module
import cem.interventions.utils as intervention_utils
import cem.train.training as training
import cem.train.utils as utils

from run_experiments import _build_arg_parser
from experiment_utils import (
    evaluate_expressions, determine_rerun,
    generate_hyperatemer_configs, filter_results,
    print_table, get_mnist_extractor_arch
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = _build_arg_parser()
args = parser.parse_args("")

args.config = config_file

if args.project_name:
    # Lazy import to avoid importing unless necessary
    pass #import wandb
if args.debug:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

if args.config:
    with open(args.config, "r") as f:
        loaded_config = yaml.load(f, Loader=yaml.FullLoader)
else:
    loaded_config = {}
if "shared_params" not in loaded_config:
    loaded_config["shared_params"] = {}
if "runs" not in loaded_config:
    loaded_config["runs"] = []

if args.dataset is not None:
    loaded_config["dataset"] = args.dataset
if loaded_config.get("dataset", None) is None:
    raise ValueError(
        "A dataset must be provided either as part of the "
        "configuration file or as a command line argument."
    )
if loaded_config["dataset"] == "cub":
    data_module = cub_data_module
    args.project_name = args.project_name.format(ds_name="cub")
elif loaded_config["dataset"] == "derm":
    data_module = derm_data_module
    args.project_name = args.project_name.format(ds_name="derma")
elif loaded_config["dataset"] == "celeba":
    data_module = celeba_data_module
    args.project_name = args.project_name.format(ds_name="celeba")
elif loaded_config["dataset"] == "chexpert":
    data_module = chexpert_data_module
    args.project_name = args.project_name.format(ds_name="chexpert")
elif loaded_config["dataset"] == "awa2":
        data_module = awa2_data_module
        args.project_name = args.project_name.format(ds_name="awa2")
elif loaded_config["dataset"] in ["xor", "vector", "dot", "trig"]:
    data_module = get_synthetic_data_loader(loaded_config["dataset"])
    args.project_name = args.project_name.format(
        ds_name=loaded_config["dataset"]
    )
    input_features = get_synthetic_num_features(loaded_config["dataset"])
    def synth_c_extractor_arch(
        output_dim,
        pretrained=False,
    ):
        if output_dim is None:
            output_dim = 128
        return torch.nn.Sequential(*[
            torch.nn.Linear(input_features, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, output_dim),
        ])
    loaded_config["c_extractor_arch"] = synth_c_extractor_arch
elif loaded_config["dataset"] == "mnist_add":
    data_module = mnist_data_module
    args.project_name = args.project_name.format(ds_name=args.dataset)
    utils.extend_with_global_params(
        loaded_config,
        args.param or []
    )
    num_operands = loaded_config.get('num_operands', 32)
    loaded_config["c_extractor_arch"] = get_mnist_extractor_arch(
        input_shape=(
            loaded_config.get('batch_size', 512),
            num_operands,
            28,
            28,
        ),
        num_operands=num_operands,
    )
else:
    raise ValueError(f"Unsupported dataset {loaded_config['dataset']}!")

if args.output_dir is not None:
    loaded_config['results_dir'] = args.output_dir
if args.debug:
    print(json.dumps(loaded_config, sort_keys=True, indent=4))
logging.info(f"Results will be dumped in {loaded_config['results_dir']}")
logging.debug(
    f"And the dataset's root directory is {loaded_config.get('root_dir')}"
)
Path(loaded_config['results_dir']).mkdir(parents=True, exist_ok=True)
# Write down the actual command executed
# And the configuration file
now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%Y_%m_%d_%H_%M")
loaded_config["time_last_called"] = now.strftime("%Y/%m/%d at %H:%M:%S")
with open(
    os.path.join(loaded_config['results_dir'], f"command_{dt_string}.txt"),
    "w",
) as f:
    command_args = [
        arg if " " not in arg else f'"{arg}"' for arg in sys.argv
    ]
    f.write("python " + " ".join(command_args))

# Also save the current experiment configuration
with open(
    os.path.join(
        loaded_config['results_dir'],
        f"experiment_{dt_string}_config.yaml")
    ,
    "w"
) as f:
    yaml.dump(loaded_config, f)

# rerun=args.rerun
rerun=False
result_dir=(
    args.output_dir if args.output_dir
    else loaded_config['results_dir']
)
project_name=args.project_name
num_workers=args.num_workers
global_params=args.param
accelerator=(
    "gpu" if (not args.force_cpu) and (torch.cuda.is_available())
    else "cpu"
)
experiment_config=loaded_config
activation_freq=args.activation_freq
single_frequency_epochs=args.single_frequency_epochs

devices = 'auto'


# Load Dataset
seed_everything(42)
# parameters for data, model, and training
experiment_config = copy.deepcopy(experiment_config)
if 'shared_params' not in experiment_config:
    experiment_config['shared_params'] = {}
# Move all global things into the shared params
for key, vals in experiment_config.items():
    if key not in ['runs', 'shared_params']:
        experiment_config['shared_params'][key] = vals
experiment_config['shared_params']['num_workers'] = num_workers

utils.extend_with_global_params(
    experiment_config['shared_params'], global_params or []
)

experiment_config['shared_params']['num_workers'] = 1

train_dl, val_dl, test_dl, imbalance, (n_concepts, n_tasks, concept_map) = \
    data_module.generate_data(
        config=experiment_config['shared_params'],
        seed=42,
        output_dataset_vars=True,
        root_dir=experiment_config['shared_params'].get('root_dir', None),
    )

# For now, we assume that all concepts have the same
# aquisition cost
acquisition_costs = None
if concept_map is not None:
    intervened_groups = list(
        range(
            0,
            len(concept_map) + 1,
            experiment_config['shared_params'].get('intervention_freq', 1),
        )
    )
else:
    intervened_groups = list(
        range(
            0,
            n_concepts + 1,
            experiment_config['shared_params'].get('intervention_freq', 1),
        )
    )
experiment_config["shared_params"]["n_concepts"] = \
    experiment_config["shared_params"].get(
        "n_concepts",
        n_concepts,
    )
experiment_config["shared_params"]["n_tasks"] = \
    experiment_config["shared_params"].get(
        "n_tasks",
        n_tasks,
    )
experiment_config["shared_params"]["concept_map"] = \
    experiment_config["shared_params"].get(
        "concept_map",
        concept_map,
    )

sample = next(iter(train_dl))
real_sample = []
for x in sample:
    if isinstance(x, list):
        real_sample += x
    else:
        real_sample.append(x)
sample = real_sample
logging.info(
    f"Training sample shape is: {sample[0].shape} with "
    f"type {sample[0].type()}"
)
logging.info(
    f"Training label shape is: {sample[1].shape} with "
    f"type {sample[1].type()}"
)
logging.info(
    f"\tNumber of output classes: {n_tasks}"
)
logging.info(
    f"Training concept shape is: {sample[2].shape} with "
    f"type {sample[2].type()}"
)
logging.info(
    f"\tNumber of training concepts: {n_concepts}"
)

task_class_weights = None

if experiment_config['shared_params'].get('use_task_class_weights', False):
    logging.info(
        f"Computing task class weights in the training dataset with "
        f"size {len(train_dl)}..."
    )
    attribute_count = np.zeros((max(n_tasks, 2),))
    samples_seen = 0
    for i, data in enumerate(train_dl):
        if len(data) == 2:
            (_, (y, _)) = data
        else:
            (_, y, _) = data
        if n_tasks > 1:
            y = torch.nn.functional.one_hot(
                y,
                num_classes=n_tasks,
            ).cpu().detach().numpy()
        else:
            y = torch.cat(
                [torch.unsqueeze(1 - y, dim=-1), torch.unsqueeze(y, dim=-1)],
                dim=-1,
            ).cpu().detach().numpy()
        attribute_count += np.sum(y, axis=0)
        samples_seen += y.shape[0]
    print("Class distribution is:", attribute_count / samples_seen)
    if n_tasks > 1:
        task_class_weights = samples_seen / attribute_count - 1
    else:
        task_class_weights = np.array(
            [attribute_count[0]/attribute_count[1]]
        )


# Set log level in env variable as this will be necessary for
# subprocessing
os.environ['LOGLEVEL'] = os.environ.get(
    'LOGLEVEL',
    logging.getLevelName(logging.getLogger().getEffectiveLevel()),
)
loglevel = os.environ['LOGLEVEL']
logging.info(f'Setting log level to: "{loglevel}"')


# ## Function to make predictions using loaded model

# should use the loaded model to make predictions using train, val, test sets. 
# should return a dictionary of this form:
# predictions
#    - train
#       - c_pred
#       - y_pred and so on
#       - c_on_KL
#       - c_off_KL
#    - val
#    - test


def concept2embeddings(concepts, positive_embeddings, negative_embeddings):
    '''
    concepts: (batch_size, num_concepts)
    positive_embeddings, negative_embeddings: (batch_size, num_concepts, embedding_size)
    '''

    assert positive_embeddings.size() == negative_embeddings.size()
    assert concepts.size(0) == positive_embeddings.size(0)
    assert concepts.size(1) == positive_embeddings.size(1)

    # reshape concepts to match shape of embeddings
    batch_size, num_concepts, embedding_size = positive_embeddings.size()
    concepts = torch.unsqueeze(concepts, dim=2).expand(batch_size, num_concepts, embedding_size)
    embeddings = concepts * positive_embeddings + (1-concepts) * negative_embeddings

    # reshape embeddings
    embeddings = torch.reshape(embeddings, (batch_size, num_concepts * embedding_size))

    return embeddings


def compute_logits(concepts, c2y_model, embeddings):
    if embeddings is None:
        logits = torch.softmax(c2y_model(concepts), dim=1).detach().clone()
    else:
        positive_embeddings, negative_embeddings = embeddings['positive_embeddings'], embeddings['negative_embeddings']
        concept_embeddings = concept2embeddings(concepts, positive_embeddings, negative_embeddings)
        logits = torch.softmax(c2y_model(concept_embeddings), dim=1).detach().clone()

    return logits


kl_loss = torch.nn.KLDivLoss(reduction="none")


def c_on_and_off_KL_div(concepts, c2y_model, embeddings=None):  # embeddings is dict {'positive_embeddings':, 'negative_embeddings':}
    # first get original logits
    original_logits = compute_logits(concepts, c2y_model, embeddings)

    num_concepts = concepts.size(1)
    # now turn concepts on/ off one by one and compute logits and subsequently KL divergence
    all_concept_i_on_KL = []
    all_concept_i_off_KL = []
    for i in range(num_concepts):
        concepts_copy = concepts.detach().clone().to(device)
        
        # turn concept on
        concepts_copy[:, i] = 1
        concept_i_on_logits = compute_logits(concepts_copy, c2y_model, embeddings)
        concept_i_on_kl = torch.mean(kl_loss(original_logits, concept_i_on_logits), dim=1).reshape(-1, 1)
        all_concept_i_on_KL.append(concept_i_on_kl)

        # turn concept off
        concepts_copy[:, i] = 0
        concept_i_off_logits = compute_logits(concepts_copy, c2y_model, embeddings)
        concept_i_off_kl = torch.mean(kl_loss(original_logits, concept_i_off_logits), dim=1).reshape(-1, 1)
        all_concept_i_off_KL.append(concept_i_off_kl)

    # combine
    all_concept_i_on_KL = torch.hstack(all_concept_i_on_KL)
    all_concept_i_off_KL = torch.hstack(all_concept_i_off_KL)

    assert all_concept_i_on_KL.size() == concepts.size()

    return all_concept_i_on_KL, all_concept_i_off_KL


def make_predictions(model, dataloader, config, store_images=False):
    all_x, all_groundtruths_y, all_groundtruths_c, all_predictions_c, all_predictions_y  = [], [], [], [], []
    all_c_i_on_KL, all_c_i_off_KL = [], []

    if 'ConceptEmbeddingModel' in config['architecture']:
        all_positive_embeddings, all_negative_embeddings = [], []

    all_correct = []

    with torch.no_grad():
        model = model.to(device)

        for i, batch in tqdm(enumerate(dataloader)):
            if len(batch) == 2:
                x_train, [y_train, c_train] = batch[0], batch[1]
            else:
                x_train, y_train, c_train = batch
            
            if store_images:
                all_x.append(x_train)

            all_groundtruths_y.append(y_train)
            all_groundtruths_c.append(c_train)
            
            x_train, y_train, c_train = x_train.to(device), y_train.to(device), c_train.to(device)
            batch = [x_train, y_train, c_train]

            if 'ConceptBottleneckModel' in config['architecture']:
                outputs = model.predict_step(batch, batch_idx=i)
                c_sem, c_pred, y_logits = outputs

            else:
                outputs = model.predict_step(batch, batch_idx=i, output_embeddings=True)
                c_sem, c_pred, y_logits, pos_embeddings_batch, neg_embeddings_batch = outputs
                all_positive_embeddings.append(pos_embeddings_batch.detach().cpu())
                all_negative_embeddings.append(neg_embeddings_batch.detach().cpu())

            all_predictions_c.append(c_sem.detach().cpu())
            all_predictions_y.append(y_logits.detach().cpu())

            if y_logits.size(1) > 1:
                y_pred = torch.argmax(y_logits, dim=-1)
            else:
                y_pred = (torch.sigmoid(y_logits) >= 0.5).float().flatten()

            correct = (y_train == y_pred).numpy(force=True)
            all_correct.append(correct)

            # get KL divergence after turning concepts on and off
            if 'ConceptEmbeddingModel' not in config['architecture']:
                e = None
            else:
                e = {'positive_embeddings': pos_embeddings_batch.detach(), 'negative_embeddings': neg_embeddings_batch.detach()}

            concept_i_on_KL, concept_i_off_KL = c_on_and_off_KL_div(c_sem.detach(), model.c2y_model, embeddings=e)

            all_c_i_on_KL.append(concept_i_on_KL)
            all_c_i_off_KL.append(concept_i_off_KL)

    all_correct = np.hstack(all_correct)
    print("Accuracy (%) = ", np.mean(all_correct)*100)

    predictions_dict = {
        'groundtruths_y': torch.hstack(all_groundtruths_y),
        'groundtruth_c': torch.vstack(all_groundtruths_c),
        'predictions_y': torch.vstack(all_predictions_y),
        'predictions_c': torch.vstack(all_predictions_c),
        'concept_i_on_KL_div': torch.vstack(all_c_i_on_KL),
        'concept_i_off_KL_div': torch.vstack(all_c_i_off_KL),
    }

    if store_images:
        predictions_dict['x'] = all_x

    if 'ConceptEmbeddingModel' in config['architecture']:
        predictions_dict['positive_embeddings'] = torch.vstack(all_positive_embeddings)
        predictions_dict['negative_embeddings'] = torch.vstack(all_negative_embeddings)

    return predictions_dict


def compile_save_dict(model, train_dl, val_dl, test_dl, config, full_run_name, model_type, imbalance):
    print('MODEL TYPE = ', model_type)
    print("GOING OVER TRAINING DATA")
    predictions_dict_train = make_predictions(model, train_dl, config)
    print("GOING OVER VALIDATION DATA")
    predictions_dict_val = make_predictions(model, val_dl, config)
    print("GOING OVER TEST DATA")
    predictions_dict_test = make_predictions(model, test_dl, config)

    # if config['c_extractor_arch'] is not string, remove it from the dict
    if config['dataset'] == 'mnist_add':
        config['c_extractor_arch'] = None
    
    model_save_dict = {
        'model_type': model_type,
        'config': config,
        'predictions': {
            'train': predictions_dict_train,
            'val': predictions_dict_val,
            'test': predictions_dict_test,
        },
        'concept_map': concept_map,
        'imbalance': imbalance,
    }

    # add necessary information for loading the model later
    params = {
        'task_class_weights': task_class_weights,
        'n_concepts': n_concepts,
        'n_tasks': n_tasks,
        'config': config,
        'train_dl': None,
        'val_dl': None,
        'test_dl': None,
        'split': split,
        'result_dir': result_dir,
        'rerun': current_rerun,
        'project_name': project_name,
        'seed': (42 + split),
        'imbalance': imbalance,
        'single_frequency_epochs': single_frequency_epochs,
        'activation_freq': activation_freq,
    }

    if model_type == 'IntCEM':
        params['gradient_clip_val'] = run_config.get('gradient_clip_val', 0)
    
    model_save_dict['model_params'] = params

    save_path = os.path.join(config['results_dir'], 'predictions', full_run_name, f'{model_type}_split={split}.pickle')
    dirname = os.path.dirname(save_path)
    Path(dirname).mkdir(parents=True, exist_ok=True)

    with open(save_path, 'wb') as f:
        pickle.dump(model_save_dict, f)
    print("Model Predictions Dictionary Saved at ", save_path)


# Load Model
print('NUM TRIALS = ', experiment_config['shared_params']["trials"])

os.makedirs(result_dir, exist_ok=True)
results = {}
for split in range(
    experiment_config['shared_params'].get("start_split", 0),
    experiment_config['shared_params']["trials"],
):
    print('SPLIT = ', split)
    results[f'{split}'] = {}
    now = datetime.now()
    print(
        f"[TRIAL "
        f"{split + 1}/{experiment_config['shared_params']['trials']} "
        f"BEGINS AT {now.strftime('%d/%m/%Y at %H:%M:%S')}"
    )
    # And then over all runs in a given trial
    for current_config in experiment_config['runs']:
        # Construct the config for this particular trial
        trial_config = copy.deepcopy(experiment_config.get('shared_params', {}))

        trial_config.update(current_config)
        trial_config["concept_map"] = concept_map
        # Now time to iterate 5
        # over all hyperparameters that were given as part
        for run_config in generate_hyperatemer_configs(trial_config):
            now = datetime.now()
            run_config = copy.deepcopy(run_config)
            evaluate_expressions(run_config)
            run_config["extra_name"] = run_config.get("extra_name", "").format(
                **run_config
            )
            old_results = None
            full_run_name = (
                f"{run_config['architecture']}{run_config.get('extra_name', '')}"
            )
            current_results_path = os.path.join(
                result_dir,
                f'{full_run_name}_split_{split}_results.joblib'
            )
            current_rerun = determine_rerun(
                config=run_config,
                rerun=rerun,
                split=split,
                full_run_name=full_run_name,
            )
            if current_rerun:
                logging.warning(
                    f"We will rerun model {full_run_name}_split_{split} "
                    f"as requested by the config"
                )
            if (not current_rerun) and os.path.exists(current_results_path):
                with open(current_results_path, 'rb') as f:
                    old_results = joblib.load(f)

            if run_config["architecture"] in [
                "IndependentConceptBottleneckModel",
                "SequentialConceptBottleneckModel",
            ]:
                # Special case for now for sequential and independent CBMs
                config = copy.deepcopy(run_config)
                config["architecture"] = "ConceptBottleneckModel"
                config["sigmoidal_prob"] = True
                full_run_name = (
                    f"{config['architecture']}{config.get('extra_name', '')}"
                )
                seq_old_results = None
                seq_current_results_path = os.path.join(
                    result_dir,
                    f'Sequential{full_run_name}_split_{split}_results.joblib'
                )
                if os.path.exists(seq_current_results_path):
                    with open(seq_current_results_path, 'rb') as f:
                        seq_old_results = joblib.load(f)

                ind_old_results = None
                ind_current_results_path = os.path.join(
                    result_dir,
                    f'Sequential{full_run_name}_split_{split}_results.joblib'
                )
                if os.path.exists(ind_current_results_path):
                    with open(ind_current_results_path, 'rb') as f:
                        ind_old_results = joblib.load(f)
                ind_model, ind_test_results, seq_model, seq_test_results = \
                    training.train_independent_and_sequential_model(
                        task_class_weights=task_class_weights,
                        n_concepts=n_concepts,
                        n_tasks=n_tasks,
                        config=config,
                        train_dl=train_dl,
                        val_dl=val_dl,
                        test_dl=test_dl,
                        split=split,
                        result_dir=result_dir,
                        rerun=current_rerun,
                        project_name=project_name,
                        seed=(42 + split),
                        imbalance=imbalance,
                        ind_old_results=ind_old_results,
                        seq_old_results=seq_old_results,
                        single_frequency_epochs=single_frequency_epochs,
                        activation_freq=activation_freq,
                    )

                # THIS IS THE PART I HAVE ADDED
                # make predictions using the independent, sequential models, and store the results in a dictionary
                ind_model.eval()
                compile_save_dict(ind_model, train_dl, val_dl, test_dl, config, full_run_name, 'independent', imbalance)
                seq_model.eval()
                compile_save_dict(seq_model, train_dl, val_dl, test_dl, config, full_run_name, 'sequential', imbalance)

            else:
                config = run_config
                model, model_results = \
                    training.train_model(
                        task_class_weights=task_class_weights,
                        accelerator=accelerator,
                        devices=devices,
                        n_concepts=n_concepts,
                        n_tasks=n_tasks,
                        config=run_config,
                        train_dl=train_dl,
                        val_dl=val_dl,
                        test_dl=test_dl,
                        split=split,
                        result_dir=result_dir,
                        rerun=current_rerun,
                        project_name=project_name,
                        seed=(42 + split),
                        imbalance=imbalance,
                        old_results=old_results,
                        gradient_clip_val=run_config.get(
                            'gradient_clip_val',
                            0,
                        ),
                        single_frequency_epochs=single_frequency_epochs,
                        activation_freq=activation_freq,
                    )

                # make predictions using the model and store them in a dict
                if config['architecture'] == 'ConceptEmbeddingModel':
                    model_type = 'CEM'
                if config['architecture'] == 'IntAwareConceptEmbeddingModel':
                    model_type = 'IntCEM'
                elif config['architecture'] == 'ConceptBottleneckModel':
                    model_type = 'joint_CBM'

                model.eval()
                print('MODEL TYPE = ', model_type)
                compile_save_dict(model, train_dl, val_dl, test_dl, config, full_run_name, model_type, imbalance)




