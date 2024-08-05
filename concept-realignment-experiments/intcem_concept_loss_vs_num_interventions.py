import torch
import numpy as np
from matplotlib import pyplot as plt
import pickle
import seaborn as sns
import pandas as pd

from torch.utils.data import TensorDataset, DataLoader

from intervention_utils import ucp, random_intervention_policy, intervene, prepare_concept_map_tensors, ectp
from train_utils import sample_trajectory, compute_loss
from model_utils import load_ind_seq_models, load_intcem_model
from concept_corrector_models import RNNConceptCorrector, LSTMConceptCorrector, NNConceptCorrector, GRUConceptCorrector
from plotting_utils import generate_lineplot, expand_tensor, concepts2embeddings, concepts2embeddingsSingleTimeStep, compute_concept_loss_and_accuracy_vs_num_interventions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data (CBM predictions, CBM model, etc.)
# CUB IntCEM
predictions_dict_path = '/mnt/qb/work/bethge/bkr046/CEM/results/predictions/IntAwareConceptEmbeddingModel_debug_concept_corrector_Retry_intervention_weight_1_horizon_rate_1.005_intervention_discount_1_task_discount_1.1/IntCEM_split=0.pickle'
adapter_path = None
dataset = 'cub'
model_type = 'intcem'
title = 'IntCEM on CUB Dataset'
intervention_policy = "ucp"
vanilla_predictions_dict_path = '/mnt/qb/work/bethge/bkr046/CEM/results/predictions/IntAwareConceptEmbeddingModelNO_concept_corrector_Retry_intervention_weight_1_horizon_rate_1.005_intervention_discount_1_task_discount_1.1/IntCEM_split=0.pickle'


with open(predictions_dict_path, 'rb') as f:
    predictions_dict = pickle.load(f)


# Load Model
if predictions_dict['model_type'] == 'IntCEM':
    predictions_dict['model_params']['rerun'] = False
    predictions_dict['model_params']['train_dl'] = None
    predictions_dict['model_params']['val_dl'] = None
    predictions_dict['model_params']['test_dl'] = None
    model, _ = load_intcem_model(predictions_dict['model_params'])
    
elif predictions_dict['model_type'] == 'independent' or predictions_dict['model_type'] == 'sequential':
    predictions_dict['model_params']['train_dl'] = None
    predictions_dict['model_params']['val_dl'] = None
    predictions_dict['model_params']['test_dl'] = None
    ind_model, _, seq_model, _ = load_ind_seq_models(predictions_dict['model_params'])

else:
    model = predictions_dict['model']

model.eval()


# Load Vanilla Model
with open(vanilla_predictions_dict_path, 'rb') as f:
    vanilla_predictions_dict = pickle.load(f)


if vanilla_predictions_dict['model_type'] == 'IntCEM':
    vanilla_predictions_dict['model_params']['train_dl'] = None
    vanilla_predictions_dict['model_params']['val_dl'] = None
    vanilla_predictions_dict['model_params']['test_dl'] = None

    vanilla_predictions_dict['model_params']['rerun'] = False
    vanilla_model, _ = load_intcem_model(vanilla_predictions_dict['model_params'])
    vanilla_model.eval()
    
else:
    raise Exception("Model type is not IntCEM, which it should be for this notebook")


num_concepts = predictions_dict['predictions']['test']['groundtruth_c'].size(1)

if 'imbalance' in predictions_dict and predictions_dict['config']['weight_loss'] == True:
    weight = torch.tensor(predictions_dict['imbalance']).to(device)
else:
    weight = None

print(weight)

criterion = torch.nn.BCELoss(weight=weight)

# Create Test DataLoader
concept_map = predictions_dict['concept_map']
group2concepts = prepare_concept_map_tensors(concept_map)
if group2concepts is not None:
    group2concepts = group2concepts.to(device) 


test_predictions = predictions_dict['predictions']['test']
test_predictions.keys()


vanilla_test_predictions = vanilla_predictions_dict['predictions']['test']
vanilla_test_predictions.keys()


if 'CEM' in predictions_dict['model_type']:
    print('adding embeddings to dataset')
    test_dataset = TensorDataset(test_predictions['predictions_c'], test_predictions['groundtruth_c'], test_predictions['groundtruths_y'], \
                                test_predictions['concept_i_on_KL_div'], test_predictions['concept_i_off_KL_div'], \
                                test_predictions['positive_embeddings'], test_predictions['negative_embeddings'])

else:
    test_dataset = TensorDataset(test_predictions['predictions_c'], test_predictions['groundtruth_c'], test_predictions['groundtruths_y'], \
                                 test_predictions['concept_i_on_KL_div'], test_predictions['concept_i_off_KL_div'])

# Create data loaders for training and testing sets
batch_size = 128#1024
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


if 'CEM' in vanilla_predictions_dict['model_type']:
    print('adding embeddings to dataset')
    vanilla_test_dataset = TensorDataset(vanilla_test_predictions['predictions_c'], vanilla_test_predictions['groundtruth_c'], vanilla_test_predictions['groundtruths_y'], \
                                vanilla_test_predictions['concept_i_on_KL_div'], vanilla_test_predictions['concept_i_off_KL_div'], \
                                vanilla_test_predictions['positive_embeddings'], vanilla_test_predictions['negative_embeddings'])

else:
    vanilla_test_dataset = TensorDataset(vanilla_test_predictions['predictions_c'], vanilla_test_predictions['groundtruth_c'], vanilla_test_predictions['groundtruths_y'], \
                                 vanilla_test_predictions['concept_i_on_KL_div'], vanilla_test_predictions['concept_i_off_KL_div'])

# Create data loaders for training and testing sets
batch_size = 128 # 1024
vanilla_test_loader = DataLoader(vanilla_test_dataset, batch_size=batch_size, shuffle=False)


# Run the plotting function without concept correction

# Load Adapter
if adapter_path is not None:
    adapter = torch.load(adapter_path)
    adapter = adapter.to(device)
else:
    adapter = None


num_trials = 10 if intervention_policy == 'random_intervention_policy' else 1

max_interventions = num_concepts if concept_map is None else group2concepts.size(0)
print('Max Number of Interventions = ', max_interventions)


if model_type == 'seq_cbm':
    label_predictor = seq_model.c2y_model

elif model_type in ('cem', 'intcem'):
    label_predictor = model.c2y_model  

else:
    raise Exception("Model not supported")

vanilla_label_predictor = vanilla_model.c2y_model
vanilla_label_predictor = vanilla_label_predictor.to(device)


class ReturnSameConcepts:
    def forward_single_timestep(self, inputs, already_intervened_concepts, original_predictions, hidden):
        return inputs, None

    def __call__(self, inputs, already_intervened_concepts, original_predictions, hidden):
        return inputs, None

    def prepare_initial_hidden(self, batch_size, device):
        return None

returnSameConcepts = ReturnSameConcepts()

concept_embeddings = ('CEM' in predictions_dict['model_type'])

# run the vanilla IntCEM without concept correction
all_concept_loss_no_correction, all_accuracy_no_correction = [], []

for _ in range(num_trials):
    concept_loss_vs_num_interventions_no_correction, accuracy_vs_num_interventions_no_correction = compute_concept_loss_and_accuracy_vs_num_interventions(vanilla_test_loader, group2concepts, returnSameConcepts, lambda x: None, eval(intervention_policy), vanilla_label_predictor, max_interventions, device, criterion, concept_embeddings=concept_embeddings, adapter=adapter)
    all_concept_loss_no_correction.append(concept_loss_vs_num_interventions_no_correction)
    all_accuracy_no_correction.append(accuracy_vs_num_interventions_no_correction)

all_concept_loss_no_correction = np.vstack(all_concept_loss_no_correction)
all_accuracy_no_correction = np.vstack(all_accuracy_no_correction)


vanilla_label_predictor = vanilla_label_predictor.cpu()

model = model.to(device)
concept_corrector = model.concept_corrector


class ConceptCorrectorWrapper:
    def __init__(self, concept_corrector):
        self.concept_corrector = concept_corrector

    def __call__(self, inputs, already_intervened_concepts, original_predictions, hidden):
        # reshape x to [batch_size x seq_length, num_concepts]
        original_size = inputs.size()
        inputs, already_intervened_concepts, original_predictions = inputs.reshape(-1, inputs.size(-1)), already_intervened_concepts.reshape(-1, already_intervened_concepts.size(-1)), original_predictions.reshape(-1, original_predictions.size(-1))
        out, _ = self.forward_single_timestep(inputs, already_intervened_concepts, original_predictions, hidden)

        # reshape out back into the original shape
        out = out.reshape(original_size)

        return out, None

    
    def forward_single_timestep(self, inputs, already_intervened_concepts, original_predictions, hidden):
        x = (1-already_intervened_concepts) * original_predictions + already_intervened_concepts * inputs
        # x = original_predictions
        out = self.concept_corrector(x, already_intervened_concepts)
        return out, None

    def prepare_initial_hidden(self, batch_size, device):
        return None

conceptCorrectorWrapper = ConceptCorrectorWrapper(concept_corrector)


concept_embeddings = ('CEM' in predictions_dict['model_type'])

# run the IntCEM jointly trained with concept corrector
all_concept_loss_with_correction, all_accuracy_with_correction = [], []
for _ in range(num_trials):
    concept_loss_vs_num_interventions_NN_corrector, accuracy_vs_num_interventions_NN_corrector = compute_concept_loss_and_accuracy_vs_num_interventions(test_loader, group2concepts, conceptCorrectorWrapper, lambda x: prepare_initial_hidden(x, 'NN', None), eval(intervention_policy), label_predictor, max_interventions, device, criterion, concept_embeddings=concept_embeddings, adapter=adapter)
    all_concept_loss_with_correction.append(concept_loss_vs_num_interventions_NN_corrector)
    all_accuracy_with_correction.append(accuracy_vs_num_interventions_NN_corrector)

all_concept_loss_with_correction = np.vstack(all_concept_loss_with_correction)
all_accuracy_with_correction = np.vstack(all_accuracy_with_correction)


# plot concept loss
with plt.style.context('seaborn-v0_8-whitegrid', after_reset=True):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    generate_lineplot(all_concept_loss_no_correction, ax=ax, kwargs={'label': 'Vanilla IntCEM', 'marker': 's'})
    generate_lineplot(all_concept_loss_with_correction, ax=ax, kwargs={'label': 'IntCEM with \nConcept Correction (Ours)', 'marker': 's'})

    plt.xlabel("Number of Intervened Concepts", fontsize=26)
    plt.ylabel("Concept Loss", fontsize=26)
    ax.tick_params(axis='both', which='major', labelsize=26)
    
    plt.title(title, fontsize=26)
    # plt.legend(fontsize=18)
    ax.get_legend().remove()

    plt.savefig(f'figs/concept_loss_{dataset}_{model_type}.pdf', bbox_inches='tight')
    plt.show()


# plot accuracy
with plt.style.context('seaborn-v0_8-whitegrid', after_reset=True):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

    generate_lineplot(all_accuracy_no_correction*100, ax=ax, kwargs={'label': 'IntCEM', 'marker': 's'})
    generate_lineplot(all_accuracy_with_correction*100, ax=ax, kwargs={'label': 'IntCEM + End-to-end Realignment', 'marker': 's'})

    plt.xlabel("Number of Intervened Concepts", fontsize=26)
    plt.ylabel("Classification Accuracy (%)", fontsize=26)
    ax.tick_params(axis='both', which='major', labelsize=26)
    
    plt.title(title, fontsize=26)
    plt.legend(fontsize=16)

    plt.savefig(f'figs/accuracy_{dataset}_{model_type}.pdf', bbox_inches='tight')
    plt.show()
