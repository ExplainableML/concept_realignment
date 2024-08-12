import torch
import numpy as np
import pickle
import os
import yaml
import optuna
import argparse
import time
from mmengine.config import Config
from pathlib import Path
import shutil

from torch.utils.data import TensorDataset, DataLoader

import torch.nn as nn
import torch.optim as optim

import torch.optim.lr_scheduler as lr_scheduler

from intervention_utils import ucp, random_intervention_policy, intervene, prepare_concept_map_tensors
from train_utils import sample_trajectory, compute_loss
from concept_corrector_models import LSTMConceptCorrector, GRUConceptCorrector, RNNConceptCorrector, NNConceptCorrector


# ========= TRAINING =========


def objective(trial, config, predictions_dict, device):
    init_time = time.time()

    weight = predictions_dict['imbalance']
    if weight is not None and predictions_dict['config']['weight_loss'] == True:
        weight = torch.tensor(weight).to(device)
    print('weight = ', weight)

    criterion = nn.BCELoss(weight=weight)  # Binary Cross-Entropy Loss
    
    model = config['optuna']['model']
    input_size = config['optuna']['input_size']
    output_size = config['optuna']['output_size']
    input_format = config['input_format']
    checkpoint_save_dir = config['optuna']['checkpoint_save_dir']

    num_layers = trial.suggest_categorical('num_layers', config['optuna']['num_layers'])
    hidden_size = trial.suggest_categorical('hidden_size', config['optuna']['hidden_size'])
    batch_size = trial.suggest_categorical('batch_size', config['optuna']['batch_size'])
    learning_rate = trial.suggest_float('learning_rate', config['optuna']['learning_rate']['min'], config['optuna']['learning_rate']['max'], log=True)
    epochs = config['optuna']['epochs']
    weight_decay = trial.suggest_float('weight_decay', config['optuna']['weight_decay']['min'], config['optuna']['weight_decay']['max'], log=True)
    intervention_policy_train_name = config['optuna']['intervention_policy_train']
    intervention_policy_validate_name = config['optuna']['intervention_policy_validate']

    if config['adapter_path'] is not None:
        adapter = torch.load(config['adapter_path'])
        adapter = adapter.to(device)
        print("USING ADAPTER")
    else:
        adapter = None

    concept_map = predictions_dict['concept_map']
    group2concepts = prepare_concept_map_tensors(concept_map)
    if group2concepts is not None:
        group2concepts = group2concepts.to(device)

    ## Training Loop

    # Instantiate the model
    ConceptCorrectorClass = eval(f'{model}ConceptCorrector')
    concept_corrector = ConceptCorrectorClass(input_size, hidden_size, num_layers, output_size, input_format)

    # if model == 'LSTM':
    #     concept_corrector = LSTMConceptCorrector(input_size, hidden_size, num_layers, output_size, input_format)

    # elif model == 'NN':
    #     concept_corrector = NNConceptCorrector(input_size, hidden_size, num_layers, output_size, input_format)

    # elif model == 'RNN':
    #     concept_corrector = RNNConceptCorrector(input_size, hidden_size, num_layers, output_size, input_format)

    # elif model == 'GRU':
    #     concept_corrector = GRUConceptCorrector(input_size, hidden_size, num_layers, output_size, input_format)

    # else:
    #     raise Exception(f"Model type {model} is not supported")

    concept_corrector = concept_corrector.to(device)

    # initialize weights
    for name, param in concept_corrector.named_parameters():
        if 'weight' in name and param.ndimension() >= 2:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)

    intervention_policy_train = eval(intervention_policy_train_name)
    intervention_policy_val = eval(intervention_policy_validate_name)

    optimizer = optim.Adam(concept_corrector.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # learning rate scheduler
    class GetLearningRate:
        def __init__(self, init_lr, multiplier=0.1):
            self.lr = init_lr
            self.multiplier = multiplier

        def __call__(self, epoch):
            return self.lr

        def update_lr(self):
            self.lr = self.lr * self.multiplier
            print("New Learning Rate is ", self.lr)

    
    get_lr = GetLearningRate(learning_rate, multiplier=0.1)
    scheduler = lr_scheduler.LambdaLR(optimizer, get_lr)


    best_model = None
    best_val_loss = float('inf')
    check_val_accuracy_interval = 10
    early_stop_counter = 0
    early_stop_patience = 6
    learning_rate_change_counter = 0
    learning_rate_change_patience = 2

    train_loader, val_loader = prepare_dataloaders(batch_size)

    num_concepts = next(iter(train_loader))[0].size(1)
    if config['optuna'].get('no_interventions', False):
        max_interventions = 0
        print("TRAIN CONCEPT CORRECTOR WITHOUT INTERVENTIONS")
    else:
        if 'max_interventions' in config['optuna']:
            max_interventions = config['optuna']['max_interventions']
        else:
            max_interventions = num_concepts if concept_map is None else group2concepts.size(0)
    print("Number of Concepts = ", num_concepts)
    print("Max Number of Interventions = ", max_interventions)

    for epoch in range(epochs):
        # Training
        concept_corrector.train()

        train_loss = 0

        for predicted_concepts, groundtruth_concepts, c_on_KL_div, c_off_KL_div in train_loader:
            optimizer.zero_grad()

            predicted_concepts, groundtruth_concepts = predicted_concepts.to(device), groundtruth_concepts.to(device)
            KL_divergences = {'c_on_KL_div': c_on_KL_div, 'c_off_KL_div': c_off_KL_div}

            initial_hidden = concept_corrector.prepare_initial_hidden(predicted_concepts.size(0), device)

            if adapter is not None:
                predicted_concepts = adapter.forward_single_timestep(predicted_concepts, torch.zeros_like(predicted_concepts), predicted_concepts, initial_hidden)
                predicted_concepts = predicted_concepts[0]

            all_inputs, all_masks, all_original_predictions, all_groundtruths = sample_trajectory(concept_corrector, predicted_concepts, group2concepts, groundtruth_concepts, KL_divergences, initial_hidden, intervention_policy_train, max_interventions, add_noise=True)
            
            out, _ = concept_corrector(all_inputs, all_masks, all_original_predictions, initial_hidden)

            loss = criterion(out, all_groundtruths)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        
        average_train_loss = train_loss / len(train_loader)
        
        # Print training and validation statistics
        if (epoch + 1) % check_val_accuracy_interval == 0:
            # Validation
            concept_corrector.eval()
            
            with torch.no_grad():
                val_loss = 0.0

                for predicted_concepts, groundtruth_concepts, c_on_KL_div, c_off_KL_div in val_loader:
                    predicted_concepts, groundtruth_concepts = predicted_concepts.to(device), groundtruth_concepts.to(device)
                    initial_hidden = concept_corrector.prepare_initial_hidden(predicted_concepts.size(0), device)
                    KL_divergences = {'c_on_KL_div': c_on_KL_div, 'c_off_KL_div': c_off_KL_div}
                    val_loss += compute_loss(concept_corrector, predicted_concepts, group2concepts, groundtruth_concepts, KL_divergences, initial_hidden, intervention_policy_val, max_interventions, criterion, add_noise=False).item()

                average_val_loss = val_loss / len(val_loader)

                if average_val_loss < best_val_loss:
                    best_val_loss = average_val_loss
                    # best_model = concept_corrector.state_dict()
                    if max_interventions == 0:
                        filename = f'{model}_lr={learning_rate}_hidden_size={hidden_size}_numlayers={num_layers}_batchsize={batch_size}_weightdecay={weight_decay}_inputformat={input_format}_trainpolicy={intervention_policy_train_name}_valpolicy={intervention_policy_validate_name}_nointerventions.pt'
                    else:
                        filename = f'{model}_lr={learning_rate}_hidden_size={hidden_size}_numlayers={num_layers}_batchsize={batch_size}_weightdecay={weight_decay}_inputformat={input_format}_trainpolicy={intervention_policy_train_name}_valpolicy={intervention_policy_validate_name}.pt'
                    path = os.path.join(checkpoint_save_dir, filename)
                    # torch.save(best_model, path)
                    torch.save(concept_corrector, path)
                    print("Saved the model at ", path, " at epoch ", epoch)
                    early_stop_counter = 0
                    learning_rate_change_counter = 0
                else:
                    early_stop_counter += 1

                time_elapsed = time.time() - init_time
                print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {average_train_loss:.4f}, Validation Loss: {average_val_loss:.4f}, Time Elapsed (sec): {time_elapsed:.4f}')

        trial.report(best_val_loss, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

        # if val accuracy doesn't improve, update lr (probably divide by 10)
        if learning_rate_change_counter > learning_rate_change_patience:
            print("Updating Learning Rate")
            get_lr.update_lr()

        # early stopping
        if early_stop_counter > early_stop_patience:
            print("Early Stopping at Epoch ", epoch)
            break

    return best_val_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', type=str, help='Path to the YAML file')
    parser.add_argument('--device_id', type=int, help='CUDA device id', default=0)
    parser.add_argument('--num_trials', type=int, help='number of hyperparameter combinations to evaluate', default=10)
    parser.add_argument('--input_format', type=str, help='what should be the input to the neural net?', default='original_and_intervened_inplace')
    parser.add_argument('--predictions_dict_path', type=str, help='location of the data used to train concept corrector (concept predictions of the base CBM, generated using run_trained_model_on_dataset.ipynb)')
    parser.add_argument('--adapter_path', type=str, help='location of adapter path')
    args = parser.parse_args()

    config = Config.fromfile(args.yaml_path)
    config.merge_from_dict(vars(args))

    # augment the name of the save directory to include the name of the base CBM model this concept corrector was trained on
    checkpoint_save_dir = config['optuna']['checkpoint_save_dir']
    base_CBM_filename = config['predictions_dict_path'].split('/')[-2:]
    new_checkpoint_save_dir = os.path.join(checkpoint_save_dir, config['optuna']['dataset'], *base_CBM_filename)
    config['optuna']['checkpoint_save_dir'] = new_checkpoint_save_dir
    # if using adapter, add it to the save dir name
    adapter_path = config.get('adapter_path', None)
    if adapter_path is not None:
        config['optuna']['checkpoint_save_dir'] += '_adapter'
    Path(config['optuna']['checkpoint_save_dir']).mkdir(parents=True, exist_ok=True)
    
    print("All Models will be stored in ", new_checkpoint_save_dir)
    
    device = 'cpu'
    # device = torch.device(f"cuda:{config.device_id}" if torch.cuda.is_available() else "cpu")

    # DATA
    # load the dataset for training concept corrector
    with open(config['predictions_dict_path'], 'rb') as f:
        predictions_dict = pickle.load(f)

    train_predictions = predictions_dict['predictions']['train']
    val_predictions = predictions_dict['predictions']['val']

    train_dataset = TensorDataset(train_predictions['predictions_c'], train_predictions['groundtruth_c'], train_predictions['concept_i_on_KL_div'], train_predictions['concept_i_off_KL_div'])
    val_dataset = TensorDataset(val_predictions['predictions_c'], val_predictions['groundtruth_c'], val_predictions['concept_i_on_KL_div'], val_predictions['concept_i_off_KL_div'])

    # Create data loaders for training and testing sets
    def prepare_dataloaders(batch_size):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader


    objective_wrapper = lambda trial: objective(trial, config, predictions_dict, device)

    study = optuna.create_study(direction='minimize')
    # study = optuna.load_study(study_name='RNN', storage='sqlite:///RNN.db')

    study.optimize(objective_wrapper, n_trials=config.num_trials)

    # Access the best parameters found during the optimization
    best_params = study.best_params
    best_value = study.best_value

    print("Best parameters:", best_params)
    print("Best value:", best_value)

    # move the best known model to the best_models folder
    if config['optuna']['no_interventions']:
        filename = f'{config["optuna"]["model"]}_lr={best_params["learning_rate"]}_hidden_size={best_params["hidden_size"]}_numlayers={best_params["num_layers"]}_batchsize={best_params["batch_size"]}_weightdecay={best_params["weight_decay"]}_inputformat={config["input_format"]}_trainpolicy={config["optuna"]["intervention_policy_train"]}_valpolicy={config["optuna"]["intervention_policy_validate"]}_nointerventions.pt'
    else:
        filename = f'{config["optuna"]["model"]}_lr={best_params["learning_rate"]}_hidden_size={best_params["hidden_size"]}_numlayers={best_params["num_layers"]}_batchsize={best_params["batch_size"]}_weightdecay={best_params["weight_decay"]}_inputformat={config["input_format"]}_trainpolicy={config["optuna"]["intervention_policy_train"]}_valpolicy={config["optuna"]["intervention_policy_validate"]}.pt'

    checkpoint_save_dir = config['optuna']['checkpoint_save_dir']
    origin_path = os.path.join(checkpoint_save_dir, filename)
    destination_dir = os.path.join(checkpoint_save_dir, 'best_models')
    destination_path = os.path.join(destination_dir, filename)
    
    Path(destination_dir).mkdir(parents=True, exist_ok=True)
    print(f'moving best model from {checkpoint_save_dir} to {os.path.join(checkpoint_save_dir, "best_models")}')
    shutil.copy(origin_path, destination_path)
