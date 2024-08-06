#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
import os

# Add the path to the directory containing your script to sys.path
sys.path.insert(0, '/home/bethge/bkr046/cem')

#####

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
import cem.interventions.utils as intervention_utils
import cem.train.training as training
import cem.train.utils as utils

from experiment_utils import (
    evaluate_expressions, determine_rerun,
    generate_hyperatemer_configs, filter_results,
    print_table, get_mnist_extractor_arch
)


# In[3]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[4]:


# config

from run_experiments import _build_arg_parser

parser = _build_arg_parser()
args = parser.parse_args("")

args.config = 'configs/cub_config.yaml'

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


# In[5]:


rerun=args.rerun
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


# In[6]:


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

os.makedirs(result_dir, exist_ok=True)
results = {}
for split in range(
    experiment_config['shared_params'].get("start_split", 0),
    experiment_config['shared_params']["trials"],
):
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

                config["architecture"] = "IndependentConceptBottleneckModel"
                training.update_statistics(
                    results[f'{split}'],
                    config,
                    ind_model,
                    ind_test_results,
                )
                full_run_name = (
                    f"{config['architecture']}{config.get('extra_name', '')}"
                )
                results[f'{split}'].update(
                    intervention_utils.test_interventions(
                        task_class_weights=task_class_weights,
                        full_run_name=full_run_name,
                        train_dl=train_dl,
                        val_dl=val_dl,
                        test_dl=test_dl,
                        imbalance=imbalance,
                        config=config,
                        n_tasks=n_tasks,
                        n_concepts=n_concepts,
                        acquisition_costs=acquisition_costs,
                        result_dir=result_dir,
                        concept_map=concept_map,
                        intervened_groups=intervened_groups,
                        accelerator=accelerator,
                        devices=devices,
                        split=split,
                        rerun=current_rerun,
                        old_results=ind_old_results,
                        independent=True,
                        competence_levels=config.get(
                        'competence_levels',
                        [1],
                    ),
                    )
                )
                logging.debug(
                    f"\tResults for {full_run_name} in split {split}:"
                )
                for key, val in filter_results(
                    results[f'{split}'],
                    full_run_name,
                    cut=True,
                ).items():
                    logging.debug(f"\t\t{key} -> {val}")
                with open(ind_current_results_path, 'wb') as f:
                    joblib.dump(
                        filter_results(results[f'{split}'], full_run_name),
                        f,
                    )

                config["architecture"] = "SequentialConceptBottleneckModel"
                training.update_statistics(
                    results[f'{split}'],
                    config,
                    seq_model,
                    seq_test_results,
                )
                full_run_name = (
                    f"{config['architecture']}{config.get('extra_name', '')}"
                )
                results[f'{split}'].update(
                    intervention_utils.test_interventions(
                        task_class_weights=task_class_weights,
                        full_run_name=full_run_name,
                        train_dl=train_dl,
                        val_dl=val_dl,
                        test_dl=test_dl,
                        imbalance=imbalance,
                        config=config,
                        n_tasks=n_tasks,
                        n_concepts=n_concepts,
                        acquisition_costs=acquisition_costs,
                        result_dir=result_dir,
                        concept_map=concept_map,
                        intervened_groups=intervened_groups,
                        accelerator=accelerator,
                        devices=devices,
                        split=split,
                        rerun=current_rerun,
                        old_results=seq_old_results,
                        sequential=True,
                        competence_levels=config.get('competence_levels', [1]),
                    )
                )
                logging.debug(
                    f"\tResults for {full_run_name} in split {split}:"
                )
                for key, val in filter_results(
                    results[f'{split}'],
                    full_run_name,
                    cut=True,
                ).items():
                    logging.debug(f"\t\t{key} -> {val}")
                with open(seq_current_results_path, 'wb') as f:
                    joblib.dump(
                        filter_results(results[f'{split}'], full_run_name),
                        f,
                    )
                if experiment_config['shared_params'].get("start_split", 0) == 0:
                    attempt = 0
                    # We will try and dump things a few times in case there
                    # are other threads/processes currently modifying or
                    # writing this same file
                    while attempt < 5:
                        try:
                            with open(
                                os.path.join(result_dir, f'results.joblib'),
                                'wb',
                            ) as f:
                                joblib.dump(results, f)
                            break
                        except Exception as e:
                            print(e)
                            print(
                                "FAILED TO SERIALIZE RESULTS TO",
                                os.path.join(result_dir, f'results.joblib')
                            )
                            attempt += 1
                    if attempt == 5:
                        raise ValueError(
                            "Could not serialize " +
                            os.path.join(result_dir, f'results.joblib') +
                            " to disk"
                        )
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


# In[7]:


torch.save(model, '/mnt/qb/work/bethge/bkr046/CEM/models/CUB_CEM.pt')


# In[8]:


# # save CEM params to load it later
# import pickle

# cem_params = {
#     'task_class_weights': task_class_weights,
#     'accelerator': accelerator,
#     'devices': devices,
#     'n_concepts': n_concepts,
#     'n_tasks': n_tasks,
#     'config': run_config,
#     'train_dl': train_dl,
#     'val_dl': val_dl,
#     'test_dl': test_dl,
#     'split': split,
#     'result_dir': result_dir,
#     'rerun': current_rerun,
#     'project_name': project_name,
#     'seed': (42 + split),
#     'imbalance': imbalance,
#     'old_results': old_results,
#     'gradient_clip_val': run_config.get('gradient_clip_val', 0),
#     'single_frequency_epochs': single_frequency_epochs,
#     'activation_freq': activation_freq,
# }

# with open("../examples/CEM_CUB_params.pickle", 'wb') as f:
#     pickle.dump(cem_params, f)


# In[ ]:


config


# In[7]:


import copy
import joblib
import logging
import numpy as np
import os
import pytorch_lightning as pl
import time
import torch

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from scipy.special import expit
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import tensorflow as tf

import cem.metrics.niching as niching
import cem.metrics.oracle as oracle
import cem.train.utils as utils

from cem.metrics.cas import concept_alignment_score
from cem.models.construction import (
    construct_model,
    construct_sequential_models,
    load_trained_model,
)

def get_ind_seq_models(
n_concepts,
n_tasks,
config,
train_dl,
val_dl,
result_dir=None,
test_dl=None,
split=None,
imbalance=None,
task_class_weights=None,
rerun=False,
logger=False,
project_name='',
seed=None,
save_model=True,
activation_freq=0,
single_frequency_epochs=0,
accelerator="auto",
devices="auto",
ind_old_results=None,
seq_old_results=None,
enable_checkpointing=False,
):
    if seed is not None:
        seed_everything(seed)
    num_epochs = 0
    training_time = 0

    extr_name = config['c_extractor_arch']
    if not isinstance(extr_name, str):
        extr_name = "lambda"
    if split is not None:
        ind_full_run_name = (
            f"IndependentConceptBottleneckModel"
            f"{config.get('extra_name', '')}_{extr_name}_fold_{split + 1}"
        )
        seq_full_run_name = (
            f"SequentialConceptBottleneckModel"
            f"{config.get('extra_name', '')}_{extr_name}_fold_{split + 1}"
        )
    else:
        ind_full_run_name = (
            f"IndependentConceptBottleneckModel"
            f"{config.get('extra_name', '')}_{extr_name}"
        )
        seq_full_run_name = (
            f"SequentialConceptBottleneckModel"
            f"{config.get('extra_name', '')}_{extr_name}"
        )
    print(f"[Training {ind_full_run_name} and {seq_full_run_name}]")
    print("config:")
    for key, val in config.items():
        print(f"\t{key} -> {val}")

    # Create the two models we will manipulate
    # Else, let's construct the two models we will need for this
    _, ind_c2y_model = construct_sequential_models(
        n_concepts,
        n_tasks,
        config,
        imbalance=imbalance,
        task_class_weights=task_class_weights,
    )

    _, seq_c2y_model = construct_sequential_models(
        n_concepts,
        n_tasks,
        config,
        imbalance=imbalance,
        task_class_weights=task_class_weights,
    )

    # As well as the wrapper CBM model we will use for serialization
    # and testing
    # We will be a bit cheeky and use the model with the task loss
    # weight set to 0 for training with the same dataset
    model_config = copy.deepcopy(config)
    model_config['concept_loss_weight'] = 1
    model_config['task_loss_weight'] = 0
    model = construct_model(
        n_concepts,
        n_tasks,
        config=model_config,
        imbalance=imbalance,
        task_class_weights=task_class_weights,
    )
    print(
        "[Number of parameters in model",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        "]",
    )
    print(
        "[Number of non-trainable parameters in model",
        sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "]",
    )
    seq_model_saved_path = os.path.join(
        result_dir,
        f'{seq_full_run_name}.pt'
    )
    ind_model_saved_path = os.path.join(
        result_dir,
        f'{ind_full_run_name}.pt'
    )
    chpt_exists = (
        os.path.exists(ind_model_saved_path) and
        os.path.exists(seq_model_saved_path)
    )
    # Construct the datasets we will need for training if the model
    # has not been found
    if rerun or (not chpt_exists):
        x_train = []
        y_train = []
        c_train = []
        for elems in train_dl:
            if len(elems) == 2:
                (x, (y, c)) = elems
            else:
                (x, y, c) = elems
            x_train.append(x.cpu().detach())
            y_train.append(y.cpu().detach())
            c_train.append(c.cpu().detach())
        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        c_train = np.concatenate(c_train, axis=0)

        if test_dl:
            x_test = []
            y_test = []
            c_test = []
            for elems in test_dl:
                if len(elems) == 2:
                    (x, (y, c)) = elems
                else:
                    (x, y, c) = elems
                x_test.append(x.cpu().detach())
                y_test.append(y.cpu().detach())
                c_test.append(c.cpu().detach())
            x_test = np.concatenate(x_test, axis=0)
            y_test = np.concatenate(y_test, axis=0)
            c_test = np.concatenate(c_test, axis=0)
        if val_dl is not None:
            x_val = []
            y_val = []
            c_val = []
            for elems in val_dl:
                if len(elems) == 2:
                    (x, (y, c)) = elems
                else:
                    (x, y, c) = elems
                x_val.append(x.cpu().detach())
                y_val.append(y.cpu().detach())
                c_val.append(c.cpu().detach())
            x_val = np.concatenate(x_val, axis=0)
            y_val = np.concatenate(y_val, axis=0)
            c_val = np.concatenate(c_val, axis=0)
        else:
            c2y_val_dl = None


    if (project_name) and result_dir and (not chpt_exists):
        # Lazy import to avoid importing unless necessary
        import wandb
        enter_obj = wandb.init(
            project=project_name,
            name=ind_full_run_name,
            config=config,
            reinit=True
        )
    else:
        enter_obj = utils.EmptyEnter()
    with enter_obj as run:
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            # We will distribute half epochs in one model and half on the other
            max_epochs=config['max_epochs'],
            check_val_every_n_epoch=config.get("check_val_every_n_epoch", 5),
            callbacks=[
                EarlyStopping(
                    monitor=config["early_stopping_monitor"],
                    min_delta=config.get("early_stopping_delta", 0.00),
                    patience=config['patience'],
                    verbose=config.get("verbose", False),
                    mode=config["early_stopping_mode"],
                ),
            ],
            # Only use the wandb logger when it is a fresh run
            logger=(
                logger or
                (WandbLogger(
                    name=ind_full_run_name,
                    project=project_name,
                    save_dir=os.path.join(result_dir, "logs"),
                ) if project_name and (rerun or (not chpt_exists)) else False)
            ),
        )
        if activation_freq:
            raise ValueError(
                "Activation drop has not yet been tested for "
                "joint/sequential models!"
            )
        else:
            x2c_trainer = trainer
        if (not rerun) and chpt_exists:
            # Then we simply load the model and proceed
            print("\tFound cached model... loading it")
            ind_model = construct_model(
                n_concepts=n_concepts,
                n_tasks=n_tasks,
                config=config,
                imbalance=imbalance,
                task_class_weights=task_class_weights,
                x2c_model=model.x2c_model,
                c2y_model=ind_c2y_model,
            )
            ind_model.load_state_dict(torch.load(ind_model_saved_path))
            if os.path.exists(
                ind_model_saved_path.replace(".pt", "_training_times.npy")
            ):
                [ind_training_time, ind_num_epochs] = np.load(
                    ind_model_saved_path.replace(".pt", "_training_times.npy")
                )
            else:
                ind_training_time, ind_num_epochs = 0, 0

            seq_model = construct_model(
                n_concepts=n_concepts,
                n_tasks=n_tasks,
                config=config,
                imbalance=imbalance,
                task_class_weights=task_class_weights,
                x2c_model=model.x2c_model,
                c2y_model=seq_c2y_model,
            )
            seq_model.load_state_dict(torch.load(seq_model_saved_path))
            if os.path.exists(
                seq_model_saved_path.replace(".pt", "_training_times.npy")
            ):
                [seq_training_time, seq_num_epochs] = np.load(
                    seq_model_saved_path.replace(".pt", "_training_times.npy")
                )
            else:
                seq_training_time, seq_num_epochs = 0, 0
        else:
            # First train the input to concept model
            print("[Training input to concept model]")
            start_time = time.time()
            x2c_trainer.fit(model, train_dl, val_dl)
            training_time += time.time() - start_time
            num_epochs += x2c_trainer.current_epoch
            if val_dl is not None:
                print(
                    "Validation results for x2c model:",
                    x2c_trainer.test(model, val_dl),
                )

            # Time to construct intermediate dataset for independent model!
            print(
                "[Constructing dataset for independent concept to label model]"
            )
            ind_c2y_train_dl = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    torch.from_numpy(
                        c_train
                    ),
                    torch.from_numpy(y_train),
                ),
                shuffle=True,
                batch_size=config['batch_size'],
                num_workers=config.get('num_workers', 5),
            )
            if val_dl is not None:
                ind_c2y_val_dl = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(
                        torch.from_numpy(
                            c_val
                        ),
                        torch.from_numpy(y_val),
                    ),
                    batch_size=config['batch_size'],
                    num_workers=config.get('num_workers', 5),
                )
            else:
                ind_c2y_val_dl = None

            print(
                "[Constructing dataset for sequential concept to label model]"
            )
            train_batch_concepts = trainer.predict(
                model,
                torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(
                        torch.from_numpy(x_train),
                        torch.from_numpy(y_train),
                        torch.from_numpy(c_train),
                    ),
                    batch_size=1,
                    num_workers=config.get('num_workers', 5),
                ),
            )
            train_complete_concepts = np.concatenate(
                list(map(lambda x: x[1], train_batch_concepts)),
                axis=0,
            )
            seq_c2y_train_dl = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    torch.from_numpy(
                        train_complete_concepts
                    ),
                    torch.from_numpy(y_train),
                ),
                shuffle=True,
                batch_size=config['batch_size'],
                num_workers=config.get('num_workers', 5),
            )

            if val_dl is not None:
                val_batch_concepts = trainer.predict(
                    model,
                    torch.utils.data.DataLoader(
                        torch.utils.data.TensorDataset(
                            torch.from_numpy(x_val),
                            torch.from_numpy(y_val),
                            torch.from_numpy(c_val),
                        ),
                        batch_size=1,
                        num_workers=config.get('num_workers', 5),
                    ),
                )
                val_complete_concepts = np.concatenate(
                    list(map(lambda x: x[1], val_batch_concepts)),
                    axis=0,
                )
                seq_c2y_val_dl = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(
                        torch.from_numpy(
                            val_complete_concepts
                        ),
                        torch.from_numpy(y_val),
                    ),
                    batch_size=config['batch_size'],
                    num_workers=config.get('num_workers', 5),
                )
            else:
                seq_c2y_val_dl = None

            # Train the independent concept to label model
            print("[Training independent concept to label model]")
            ind_c2y_trainer = pl.Trainer(
                accelerator=accelerator,
                devices=devices,
                # We will distribute half epochs in one model and half on the
                # other
                max_epochs=config.get('c2y_max_epochs', 50),
                enable_checkpointing=enable_checkpointing,
                check_val_every_n_epoch=config.get(
                    "check_val_every_n_epoch",
                    5,
                ),
                callbacks=[
                    EarlyStopping(
                        monitor=config["early_stopping_monitor"],
                        min_delta=config.get("early_stopping_delta", 0.00),
                        patience=config['patience'],
                        verbose=config.get("verbose", False),
                        mode=config["early_stopping_mode"],
                    ),
                ],
                # Only use the wandb logger when it is a fresh run
                logger=(
                    logger or
                    (
                        WandbLogger(
                            name=ind_full_run_name,
                            project=project_name,
                            save_dir=os.path.join(result_dir, "logs"),
                        ) if project_name and (rerun or (not chpt_exists))
                        else False
                    )
                ),
            )
            start_time = time.time()
            ind_c2y_trainer.fit(
                ind_c2y_model,
                ind_c2y_train_dl,
                ind_c2y_val_dl,
            )
            ind_training_time = training_time + time.time() - start_time
            ind_num_epochs = num_epochs + ind_c2y_trainer.current_epoch
            if ind_c2y_val_dl is not None:
                print(
                    "Independent validation results for c2y model:",
                    ind_c2y_trainer.test(ind_c2y_model, ind_c2y_val_dl),
                )

            # Train the sequential concept to label model
            print("[Training sequential concept to label model]")
            seq_c2y_trainer = pl.Trainer(
                accelerator=accelerator,
                devices=devices,
                # We will distribute half epochs in one model and half on the
                # other
                max_epochs=config.get('c2y_max_epochs', 50),
                enable_checkpointing=enable_checkpointing,
                check_val_every_n_epoch=config.get(
                    "check_val_every_n_epoch",
                    5,
                ),
                callbacks=[
                    EarlyStopping(
                        monitor=config["early_stopping_monitor"],
                        min_delta=config.get("early_stopping_delta", 0.00),
                        patience=config['patience'],
                        verbose=config.get("verbose", False),
                        mode=config["early_stopping_mode"],
                    ),
                ],
                # Only use the wandb logger when it is a fresh run
                logger=(
                    logger or
                    (
                        WandbLogger(
                            name=seq_full_run_name,
                            project=project_name,
                            save_dir=os.path.join(result_dir, "logs"),
                        ) if project_name and (rerun or (not chpt_exists))
                        else False
                    )
                ),
            )
            start_time = time.time()
            seq_c2y_trainer.fit(
                seq_c2y_model,
                seq_c2y_train_dl,
                seq_c2y_val_dl,
            )
            seq_training_time = training_time + time.time() - start_time
            seq_num_epochs = num_epochs + seq_c2y_trainer.current_epoch
            if seq_c2y_val_dl is not None:
                print(
                    "Sequential validation results for c2y model:",
                    seq_c2y_trainer.test(seq_c2y_model, seq_c2y_val_dl),
                )

            # Dump the config file
            config_copy = copy.deepcopy(config)
            if "c_extractor_arch" in config_copy and (
                not isinstance(config_copy["c_extractor_arch"], str)
            ):
                del config_copy["c_extractor_arch"]
            joblib.dump(
                config_copy,
                os.path.join(
                    result_dir,
                    f'{ind_full_run_name}_experiment_config.joblib',
                ),
            )
            joblib.dump(
                config_copy,
                os.path.join(
                    result_dir,
                    f'{seq_full_run_name}_experiment_config.joblib',
                ),
            )

            # And serialize the end models
            ind_model = construct_model(
                n_concepts=n_concepts,
                n_tasks=n_tasks,
                config=config,
                imbalance=imbalance,
                task_class_weights=task_class_weights,
                x2c_model=model.x2c_model,
                c2y_model=ind_c2y_model,
            )
            if save_model:
                torch.save(
                    ind_model.state_dict(),
                    ind_model_saved_path,
                )
                np.save(
                    ind_model_saved_path.replace(".pt", "_training_times.npy"),
                    np.array([ind_training_time, ind_num_epochs]),
                )
            seq_model = construct_model(
                n_concepts=n_concepts,
                n_tasks=n_tasks,
                config=config,
                imbalance=imbalance,
                task_class_weights=task_class_weights,
                x2c_model=model.x2c_model,
                c2y_model=seq_c2y_model,
            )
            if save_model:
                torch.save(
                    seq_model.state_dict(),
                    seq_model_saved_path,
                )
                np.save(
                    seq_model_saved_path.replace(".pt", "_training_times.npy"),
                    np.array([seq_training_time, seq_num_epochs]),
                )

    return ind_model, seq_model, train_dl, val_dl, test_dl



# def get_joint_model(
#     n_concepts,
#     n_tasks,
#     config,
#     train_dl,
#     val_dl,
#     result_dir=None,
#     test_dl=None,
#     split=None,
#     imbalance=None,
#     task_class_weights=None,
#     rerun=False,
#     logger=False,
#     project_name='',
#     seed=None,
#     save_model=True,
#     activation_freq=0,
#     single_frequency_epochs=0,
#     gradient_clip_val=0,
#     old_results=None,
#     enable_checkpointing=False,
#     accelerator="auto",
#     devices="auto",
# ):
#     if seed is not None:
#         seed_everything(seed)

#     extr_name = config['c_extractor_arch']
#     if not isinstance(extr_name, str):
#         extr_name = "lambda"
#     key_full_run_name = (
#         f"{config['architecture']}{config.get('extra_name', '')}"
#     )
#     if split is not None:
#         full_run_name = (
#             f"{key_full_run_name}_{extr_name}_fold_{split + 1}"
#         )
#     else:
#         full_run_name = (
#             f"{key_full_run_name}_{extr_name}"
#         )
#     print(f"[Training {full_run_name}]")
#     print("config:")
#     for key, val in config.items():
#         print(f"\t{key} -> {val}")

#     # create model
#     model = construct_model(
#         n_concepts,
#         n_tasks,
#         config,
#         imbalance=imbalance,
#         task_class_weights=task_class_weights,
#     )
#     print(
#         "[Number of parameters in model",
#         sum(p.numel() for p in model.parameters() if p.requires_grad),
#         "]"
#     )
#     print(
#         "[Number of non-trainable parameters in model",
#         sum(p.numel() for p in model.parameters() if not p.requires_grad),
#         "]",
#     )
#     if config.get("model_pretrain_path"):
#         if os.path.exists(config.get("model_pretrain_path")):
#             # Then we simply load the model and proceed
#             print("\tFound pretrained model to load the initial weights from!")
#             model.load_state_dict(
#                 torch.load(config.get("model_pretrain_path")),
#                 strict=False,
#             )
            
#     return model, train_dl, val_dl, test_dl


# In[89]:


ind_model, seq_model, train_dl, val_dl, test_dl = \
    get_ind_seq_model(
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


# In[ ]:


# torch.save(seq_model, '/mnt/qb/work/bethge/bkr046/CEM/models/sequential_model.pt')


# In[ ]:


# def _inner_call(trainer, model):
#     [test_results] = trainer.test(model, test_dl)
#     output = [
#         test_results["test_c_accuracy"],
#         test_results["test_y_accuracy"],
#         test_results["test_c_auc"],
#         test_results["test_y_auc"],
#         test_results["test_c_f1"],
#         test_results["test_y_f1"],
#     ]
#     top_k_vals = []
#     for key, val in test_results.items():
#         if "test_y_top" in key:
#             top_k = int(key[len("test_y_top_"):-len("_accuracy")])
#             top_k_vals.append((top_k, val))
#     output += list(map(
#         lambda x: x[1],
#         sorted(top_k_vals, key=lambda x: x[0]),
#     ))
#     return output


# seq_trainer = pl.Trainer(
#     accelerator=accelerator,
#     devices=devices,
# )

# _inner_call(seq_trainer, seq_model)


# In[25]:


seq_trainer = pl.Trainer(
    accelerator=accelerator,
    devices=devices,
)


# In[ ]:


seq_trainer.test(seq_model, test_dl)


# In[ ]:


batch = next(iter(train_dl))
seq_model._run_step(batch, batch_idx=0)


# In[14]:


x, y, (c, competencies, prev_interventions) = seq_model._unpack_batch(batch)
outputs = seq_model._forward(
    x,
    intervention_idxs=None,
    c=c,
    y=y,
    train=False,
    competencies=competencies,
    prev_interventions=prev_interventions,
)

y_logits = outputs[-1]
y_pred = torch.argmax(y_logits, dim=-1)

print(y)
print(y_pred)


# In[15]:


batch[1]


# In[16]:


output = seq_model.predict_step(batch, batch_idx=0)
y_logits = output[-1]
y_pred = torch.argmax(y_logits, dim=-1)
y = batch[1]

print(y)
print(y_pred)


# In[17]:


predictions = seq_trainer.predict(seq_model, train_dl)


# In[18]:


all_y_pred = torch.cat([predictions_batch[-1] for predictions_batch in predictions])
all_y_pred.size()


# In[19]:


all_y_train = torch.cat([x[1] for x in train_dl]).numpy(force=True)
all_y_train.shape


# In[20]:


all_y_pred = torch.argmax(all_y_pred, dim=-1).numpy(force=True)
all_y_pred.shape


# In[21]:


print("Accuracy = ", np.mean(all_y_pred == all_y_train)*100)


# In[22]:


train_dl.shuffle = False


# In[23]:


all_x_train, all_y_train, all_c_train, all_c_pred, all_y_pred = [], [], [], [], []

all_correct = []

with torch.no_grad():
    seq_model = seq_model.to(device)

    for i, batch in enumerate(train_dl):
        x_train, y_train, c_train = batch
        
        all_x_train.append(x_train)
        all_y_train.append(y_train)
        all_c_train.append(c_train)
        
        x_train, y_train, c_train = x_train.to(device), y_train.to(device), c_train.to(device)
        batch = [x_train, y_train, c_train]

        outputs = seq_model.predict_step(batch, batch_idx=i)
        c_sem, c_pred, y_logits, tail_results = outputs

        all_c_pred.append(c_sem.detach().cpu())
        all_y_pred.append(y_logits.detach().cpu())

        y_pred = torch.argmax(y_logits, dim=-1)

        correct = (y_train == y_pred).numpy(force=True)
        all_correct.append(correct)


# In[24]:


print('accuracy = ', np.mean(np.hstack(all_correct)))


# In[25]:


# sequential_model_predictions = {
#     'all_x_train': all_x_train,
#     'all_y_train': all_y_train,
#     'all_c_train': all_c_train,
#     'all_c_pred': all_c_pred,
#     'all_y_pred': all_y_pred,
# }

# import pickle

# with open('/mnt/qb/work/bethge/bkr046/CEM/results/sequential_model_predictions_train.pickle', 'wb') as f:
#     pickle.dump(sequential_model_predictions, f)


# In[26]:


all_x_test, all_y_test, all_c_test, all_c_test_pred, all_y_test_pred = [], [], [], [], []

all_correct = []

with torch.no_grad():
    seq_model = seq_model.to(device)

    for i, batch in enumerate(test_dl):
        x_test, y_test, c_test = batch
        
        all_x_test.append(x_test)
        all_y_test.append(y_test)
        all_c_test.append(c_test)
        
        x_test, y_test, c_test = x_test.to(device), y_test.to(device), c_test.to(device)
        batch = [x_test, y_test, c_test]

        outputs = seq_model.predict_step(batch, batch_idx=i)
        c_sem, c_pred, y_logits = outputs

        all_c_test_pred.append(c_sem.detach().cpu())
        all_y_test_pred.append(y_logits.detach().cpu())

        y_pred = torch.argmax(y_logits, dim=-1)

        correct = (y_test == y_pred).numpy(force=True)
        all_correct.append(correct)


# In[27]:


print('accuracy = ', np.mean(np.hstack(all_correct)))


# In[29]:


y_pred = torch.vstack(all_y_test_pred)
y_pred.size()


# In[30]:


y_pred_labels = torch.argmax(y_pred, dim=-1)
y_pred_labels.size()


# In[31]:


y_test = torch.hstack(all_y_test)
y_test.size()


# In[32]:


torch.mean((y_pred_labels == y_test).float())


# In[28]:


sequential_model_predictions = {
    'all_x_test': all_x_test,
    'all_y_test': all_y_test, 
    'all_c_test': all_c_test, 
    'all_c_test_pred': all_c_test_pred, 
    'all_y_test_pred': all_y_test_pred
}

import pickle

with open('/mnt/qb/work/bethge/bkr046/CEM/results/sequential_model_predictions_test.pickle', 'wb') as f:
    pickle.dump(sequential_model_predictions, f)


# In[ ]:





# ## Making predictions using Concept Embedding Models

# In[8]:


seq_trainer = pl.Trainer(
    accelerator=accelerator,
    devices=devices,
)


# In[9]:


seq_trainer.test(model, test_dl)


# In[10]:


predictions = seq_trainer.predict(model, train_dl)


# In[11]:


batch = next(iter(train_dl))
model._run_step(batch, batch_idx=0)


# In[12]:


x, y, (c, competencies, prev_interventions) = model._unpack_batch(batch)
outputs = model._forward(
    x,
    intervention_idxs=None,
    c=c,
    y=y,
    train=False,
    competencies=competencies,
    prev_interventions=prev_interventions,
)

y_logits = outputs[-1]
y_pred = torch.argmax(y_logits, dim=-1)

print(y)
print(y_pred)


# In[13]:


batch[1]


# In[14]:


all_y_pred = torch.cat([predictions_batch[-1] for predictions_batch in predictions])
all_y_pred.size()


# In[15]:


all_y_train = torch.cat([x[1] for x in train_dl]).numpy(force=True)
all_y_train.shape


# In[16]:


all_y_pred = torch.argmax(all_y_pred, dim=-1).numpy(force=True)
all_y_pred.shape


# In[17]:


print("Accuracy = ", np.mean(all_y_pred == all_y_train)*100)


# In[18]:


train_dl.shuffle = False


# In[26]:


all_x_train, all_y_train, all_c_train, all_c_pred, all_y_pred, all_positive_embeddings, all_negative_embeddings = [], [], [], [], [], [], []

all_correct = []

with torch.no_grad():
    model = model.to(device)

    for i, batch in enumerate(train_dl):
        x_train, y_train, c_train = batch
        
        all_x_train.append(x_train)
        all_y_train.append(y_train)
        all_c_train.append(c_train)
        
        x_train, y_train, c_train = x_train.to(device), y_train.to(device), c_train.to(device)
        batch = [x_train, y_train, c_train]

        outputs = model.predict_step(batch, batch_idx=i, output_embeddings=True)
        c_sem, c_pred, y_logits, positive_embeddings, negative_embeddings = outputs

        all_c_pred.append(c_sem.detach().cpu())
        all_y_pred.append(y_logits.detach().cpu())

        all_positive_embeddings.append(positive_embeddings.detach().cpu())
        all_negative_embeddings.append(negative_embeddings.detach().cpu())

        y_pred = torch.argmax(y_logits, dim=-1)

        correct = (y_train == y_pred).numpy(force=True)
        all_correct.append(correct)


# In[27]:


positive_embeddings.size()


# In[28]:


c_sem.size()


# In[29]:


c_pred.size()


# In[30]:


print('accuracy = ', np.mean(np.hstack(all_correct)))


# In[31]:


y_pred = torch.vstack(all_y_pred)
y_pred.size()
y_pred_labels = torch.argmax(y_pred, dim=-1)
y_pred_labels.size()
y_train = torch.hstack(all_y_train)
y_train.size()
torch.mean((y_pred_labels == y_train).float())


# In[32]:


sequential_model_predictions = {
    'all_x_train': all_x_train,
    'all_y_train': all_y_train,
    'all_c_train': all_c_train,
    'all_c_pred': all_c_pred,
    'all_y_pred': all_y_pred,
    'all_positive_embeddings': all_positive_embeddings,
    'all_negative_embeddings': all_negative_embeddings,
}

import pickle

with open('/mnt/qb/work/bethge/bkr046/CEM/results/CEM_CUB_predictions_train.pickle', 'wb') as f:
    pickle.dump(sequential_model_predictions, f)


# In[38]:


all_x_test, all_y_test, all_c_test, all_c_test_pred, all_y_test_pred, all_positive_embeddings_test, all_negative_embeddings_test = [], [], [], [], [], [], []

all_correct = []

with torch.no_grad():
    model = model.to(device)

    for i, batch in enumerate(test_dl):
        x_test, y_test, c_test = batch
        
        all_x_test.append(x_test)
        all_y_test.append(y_test)
        all_c_test.append(c_test)
        
        x_test, y_test, c_test = x_test.to(device), y_test.to(device), c_test.to(device)
        batch = [x_test, y_test, c_test]

        outputs = model.predict_step(batch, batch_idx=i, output_embeddings=True)
        c_sem, c_pred, y_logits, positive_embeddings, negative_embeddings = outputs

        all_c_test_pred.append(c_sem.detach().cpu())
        all_y_test_pred.append(y_logits.detach().cpu())
        all_positive_embeddings_test.append(positive_embeddings.detach().cpu())
        all_negative_embeddings_test.append(negative_embeddings.detach().cpu())

        y_pred = torch.argmax(y_logits, dim=-1)

        correct = (y_test == y_pred).numpy(force=True)
        all_correct.append(correct)


# In[39]:


print('accuracy = ', np.mean(np.hstack(all_correct)))


# In[40]:


y_pred = torch.vstack(all_y_test_pred)
y_pred.size()


# In[41]:


y_pred_labels = torch.argmax(y_pred, dim=-1)
y_pred_labels.size()


# In[42]:


y_test = torch.hstack(all_y_test)
y_test.size()


# In[43]:


torch.mean((y_pred_labels == y_test).float())


# In[44]:


sequential_model_predictions = {
    'all_x_test': all_x_test,
    'all_y_test': all_y_test, 
    'all_c_test': all_c_test, 
    'all_c_test_pred': all_c_test_pred, 
    'all_y_test_pred': all_y_test_pred,
    'all_positive_embeddings_test': all_positive_embeddings_test,
    'all_negative_embeddings_test': all_negative_embeddings_test,
}

import pickle

with open('/mnt/qb/work/bethge/bkr046/CEM/results/CEM_CUB_predictions_test.pickle', 'wb') as f:
    pickle.dump(sequential_model_predictions, f)


# In[ ]:




