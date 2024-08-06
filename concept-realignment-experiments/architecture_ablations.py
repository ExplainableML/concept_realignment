import torch
import numpy as np
from matplotlib import pyplot as plt
import pickle
import seaborn as sns
import pandas as pd

from torch.utils.data import TensorDataset, DataLoader

from intervention_utils import ucp, random_intervention_policy, intervene, prepare_concept_map_tensors, ectp
from train_utils import sample_trajectory, compute_loss
from model_utils import load_ind_seq_models
from concept_corrector_models import RNNConceptCorrector, LSTMConceptCorrector, NNConceptCorrector, GRUConceptCorrector
from plotting_utils import generate_lineplot, expand_tensor, concepts2embeddings, concepts2embeddingsSingleTimeStep, compute_concept_loss_and_accuracy_vs_num_interventions


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load Data (CBM predictions, CBM model, etc.)
# CUB CBM Seq
predictions_dict_path = '/mnt/qb/work/bethge/bkr046/CEM/results/predictions/ConceptBottleneckModelNoInterventionInTraining/sequential_split=0.pickle'
adapter_path = None
dataset = 'cub'
model_type = 'seq_cbm'
title = 'Sequential CBM on CUB Dataset'
intervention_policy = 'ucp'


with open(predictions_dict_path, 'rb') as f:
    predictions_dict = pickle.load(f)


if predictions_dict['model_type'] == 'independent' or predictions_dict['model_type'] == 'sequential':
    predictions_dict['model_params']['train_dl'] = None
    predictions_dict['model_params']['val_dl'] = None
    predictions_dict['model_params']['test_dl'] = None
    ind_model, _, seq_model, _ = load_ind_seq_models(predictions_dict['model_params'])

else:
    predictions_dict['model_params']['train_dl'] = None
    predictions_dict['model_params']['val_dl'] = None
    predictions_dict['model_params']['test_dl'] = None

    predictions_dict['model_params']['rerun'] = False
    model, _ = load_intcem_model(predictions_dict['model_params'])
    model.eval()


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


if predictions_dict['model_type'] == 'CEM':
    test_dataset = TensorDataset(test_predictions['predictions_c'], test_predictions['groundtruth_c'], test_predictions['groundtruths_y'], \
                                 test_predictions['concept_i_on_KL_div'], test_predictions['concept_i_off_KL_div'], \
                                 test_predictions['positive_embeddings'], test_predictions['negative_embeddings'])

else:
    test_dataset = TensorDataset(test_predictions['predictions_c'], test_predictions['groundtruth_c'], test_predictions['groundtruths_y'], \
                                 test_predictions['concept_i_on_KL_div'], test_predictions['concept_i_off_KL_div'])

# Create data loaders for training and testing sets
batch_size = 1024
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Load Concept Corrector Model
def load_model(concept_corrector_path):

    nn_concept_corrector_model = torch.load(concept_corrector_path)

    if isinstance(nn_concept_corrector_model, dict):
        import re
        from pathlib import Path

        hidden_size = int(re.search(r'hidden_size=(\d+)_', concept_corrector_path).group(1))
        hidden_layers = int(re.search(r'numlayers=(\d+)_', concept_corrector_path).group(1))
        input_format = re.search(r'inputformat=([a-zA-Z_]+)_', concept_corrector_path).group(1)
        print('input_format = ', input_format)

        corrector_filename = Path(concept_corrector_path).name
        corrector_model_type = corrector_filename.split('_')[0]

        print(corrector_model_type, hidden_size, hidden_layers, input_format)

        corrector_class = eval(f'{corrector_model_type}ConceptCorrector')

        nn_concept_corrector_model = corrector_class(input_size=num_concepts, hidden_size=hidden_size, hidden_layers=hidden_layers, output_size=num_concepts, input_format=input_format)
        nn_concept_corrector_model.load_state_dict(torch.load(concept_corrector_path))

    nn_concept_corrector_model = nn_concept_corrector_model.to(device)    
    nn_concept_corrector_model.eval()

    return nn_concept_corrector_model



lstm_original_concepts_path = '/mnt/qb/work/bethge/bkr046/CEM/checkpoints/concept_corrector_saved_models/cub/ConceptBottleneckModelNoInterventionInTraining/sequential_split=0.pickle/best_models/LSTM_lr=0.03777518439770184_hidden_size=112_numlayers=1_batchsize=512_weightdecay=1.7676741971442668e-05_inputformat=original_and_intervened_inplace_trainpolicy=ucp_valpolicy=ucp.pt'
lstm_original_concepts_corrector = load_model(lstm_original_concepts_path)

lstm_previous_output_path = '/mnt/qb/work/bethge/bkr046/CEM/checkpoints/concept_corrector_saved_models/cub/ConceptBottleneckModelNoInterventionInTraining/sequential_split=0.pickle/best_models/LSTM_lr=0.09570736029542151_hidden_size=56_numlayers=1_batchsize=512_weightdecay=2.950603585572521e-05_inputformat=previous_output_trainpolicy=ucp_valpolicy=ucp.pt'
lstm_previous_output_corrector = load_model(lstm_previous_output_path)

nn_original_concepts_path = '/mnt/qb/work/bethge/bkr046/CEM/checkpoints/concept_corrector_saved_models/cub/ConceptBottleneckModelNoInterventionInTraining/sequential_split=0.pickle/best_models/NN_lr=0.02793541235937602_hidden_size=224_numlayers=3_batchsize=512_weightdecay=4.1084036364790424e-06_inputformat=original_and_intervened_inplace_trainpolicy=ucp_valpolicy=ucp.pt'
nn_original_concepts_corrector = load_model(nn_original_concepts_path)

nn_previous_output_path = '/mnt/qb/work/bethge/bkr046/CEM/checkpoints/concept_corrector_saved_models/cub/ConceptBottleneckModelNoInterventionInTraining/sequential_split=0.pickle/best_models/NN_lr=0.04441047380810595_hidden_size=224_numlayers=3_batchsize=512_weightdecay=9.413452814757727e-06_inputformat=previous_output_trainpolicy=ucp_valpolicy=ucp.pt'
nn_previous_output_corrector = load_model(nn_previous_output_path)


if adapter_path is not None:
    adapter = torch.load(adapter_path)
    adapter = adapter.to(device)
    adapter
else:
    adapter = None

num_trials = 10 if intervention_policy == 'random_intervention_policy' else 1

max_interventions = num_concepts if concept_map is None else group2concepts.size(0)
print('Max Number of Interventions = ', max_interventions)


if model_type == 'seq_cbm':
    label_predictor = seq_model.c2y_model

elif model_type == 'ind_cbm':
    label_predictor = ind_model.c2y_model

elif model_type == 'cem':
    label_predictor = model.c2y_model  

elif model_type == 'joint_CBM':
    label_predictor = model.c2y_model  

elif model_type == 'intcem':
    label_predictor = model.c2y_model  

else:
    raise Exception("Model not supported")

label_predictor = label_predictor.to(device)


def get_title(title):
    policy_names = {'ucp': 'UCP Interventions', 'random_intervention_policy': 'Random Interventions'}
    title += f'\n{policy_names[intervention_policy]}'
    
    return title


# Create Plot
class ReturnSameConcepts:
    def forward_single_timestep(self, inputs, already_intervened_concepts, original_predictions, hidden):
        return inputs, None

    def __call__(self, inputs, already_intervened_concepts, original_predictions, hidden):
        return inputs, None

    def prepare_initial_hidden(self, batch_size, device):
        return None

returnSameConcepts = ReturnSameConcepts()

concept_embeddings = (predictions_dict['model_type'] == 'CEM')

all_concept_loss_no_correction, all_accuracy_no_correction = [], []

for _ in range(num_trials):
    concept_loss_vs_num_interventions_no_correction, accuracy_vs_num_interventions_no_correction = compute_concept_loss_and_accuracy_vs_num_interventions(test_loader, group2concepts, returnSameConcepts, lambda x: None, eval(intervention_policy), label_predictor, max_interventions, device, criterion, concept_embeddings=concept_embeddings, adapter=adapter)
    all_concept_loss_no_correction.append(concept_loss_vs_num_interventions_no_correction)
    all_accuracy_no_correction.append(accuracy_vs_num_interventions_no_correction)

all_concept_loss_no_correction = np.vstack(all_concept_loss_no_correction)
all_accuracy_no_correction = np.vstack(all_accuracy_no_correction)


models = {'MLP + Original Concepts': nn_original_concepts_corrector, 
          'MLP + Previous Output': nn_previous_output_corrector,
          'LSTM + Original Concepts': lstm_original_concepts_corrector,
          'LSTM + Previous Output': lstm_previous_output_corrector,
          }

get_linestyle = lambda name: 'dashed' if "Previous" in name else 'solid'
get_color = lambda name: '#1f77b4' if "LSTM" in name else '#ff7f0e' 


concept_embeddings = (predictions_dict['model_type'] == 'CEM')

all_concept_loss_with_correction_all_models, all_accuracy_with_correction_all_models = {}, {}

for corrector_name in models:
    all_concept_loss_with_correction, all_accuracy_with_correction = [], []
    corrector_model = models[corrector_name]
    
    for _ in range(num_trials):
        
        concept_loss_vs_num_interventions_NN_corrector, accuracy_vs_num_interventions_NN_corrector = compute_concept_loss_and_accuracy_vs_num_interventions(test_loader, group2concepts, corrector_model, lambda x: prepare_initial_hidden(x, 'NN', None), eval(intervention_policy), label_predictor, max_interventions, device, criterion, concept_embeddings=concept_embeddings, adapter=adapter)
        all_concept_loss_with_correction.append(concept_loss_vs_num_interventions_NN_corrector)
        all_accuracy_with_correction.append(accuracy_vs_num_interventions_NN_corrector)

    all_concept_loss_with_correction = np.vstack(all_concept_loss_with_correction)
    all_accuracy_with_correction = np.vstack(all_accuracy_with_correction)

    all_concept_loss_with_correction_all_models[corrector_name] = all_concept_loss_with_correction
    all_accuracy_with_correction_all_models[corrector_name] = all_accuracy_with_correction    


with plt.style.context('seaborn-v0_8-whitegrid', after_reset=True):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))

    for name in all_concept_loss_with_correction_all_models:
        result = all_concept_loss_with_correction_all_models[name]
        # generate_lineplot(result, ax=ax, kwargs={'label': name, 'marker': 's', 'linestyle': get_linestyle(name), 'color': get_color(name)})
        generate_lineplot(result, ax=ax, kwargs={'label': name, 'marker': 's',})

    # generate_lineplot(all_concept_loss_no_correction, ax=ax, kwargs={'label': 'Without Realignment', 'marker': 's', 'color': 'black', 'linestyle': '--'})

    plt.title(get_title(title), fontsize=26)
    plt.xlabel("Number of Intervened Concepts", fontsize=26)
    plt.ylabel("Concept Loss", fontsize=26)
    
    ax.get_legend().remove()
    plt.tick_params(axis='both', which='major', labelsize=26)

    plt.savefig(f'figs/ablation_concept_loss_architecture_input_format_policy={intervention_policy}_{dataset}_{model_type}.pdf', bbox_inches='tight')
    plt.show()


with plt.style.context('seaborn-v0_8-whitegrid', after_reset=True):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))

    for name in all_accuracy_with_correction_all_models:
        result = all_accuracy_with_correction_all_models[name]
        # generate_lineplot(result*100, ax=ax, kwargs={'label': name, 'marker': 's', 'linestyle': get_linestyle(name), 'color': get_color(name)})
        generate_lineplot(result*100, ax=ax, kwargs={'label': name, 'marker': 's'})

    # generate_lineplot(all_accuracy_no_correction*100, ax=ax, kwargs={'label': 'Without Realignment', 'marker': 's', 'color': 'black', 'linestyle': '--'})

    plt.xlabel("Number of Intervened Concepts", fontsize=26)
    plt.ylabel("Classification Accuracy (%)", fontsize=26)
    
    plt.title(get_title(title), fontsize=26)
    ax.tick_params(axis='both', which='major', labelsize=26)

    # if intervention_policy == 'random_intervention_policy':
    #     ax.get_legend().remove()
    # else:
    #     ax.legend(fontsize=18)
    plt.legend(fontsize=18)

    plt.savefig(f'figs/ablation_accuracy_architecture_input_format_policy={intervention_policy}_{dataset}_{model_type}.pdf', bbox_inches='tight')
    plt.show()