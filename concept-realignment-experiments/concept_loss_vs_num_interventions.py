import torch
import numpy as np
from matplotlib import pyplot as plt
import pickle
import pandas as pd
import seaborn as sns

from torch.utils.data import TensorDataset, DataLoader

from intervention_utils import ucp, random_intervention_policy, intervene, prepare_concept_map_tensors, ectp
from train_utils import sample_trajectory, compute_loss
from model_utils import load_ind_seq_models, load_intcem_model
from concept_corrector_models import RNNConceptCorrector, LSTMConceptCorrector, NNConceptCorrector, GRUConceptCorrector
from plotting_utils import generate_lineplot, area_under_curve, auc, expand_tensor, concepts2embeddings, concepts2embeddingsSingleTimeStep, compute_concept_loss_and_accuracy_vs_num_interventions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load Data (CBM predictions, CBM model, etc.)

# # CelebA Seq
# predictions_dict_path = '/mnt/qb/work/bethge/bkr046/CEM/results/CelebA/predictions/ConceptBottleneckModelresnet34/sequential_split=0.pickle'
# concept_corrector_path = '/mnt/qb/work/bethge/bkr046/CEM/checkpoints/concept_corrector_saved_models/celeba/ConceptBottleneckModelresnet34/sequential_split=0.pickle/best_models/NN_lr=0.05160876899166813_hidden_size=4_numlayers=1_batchsize=512_weightdecay=1.936314929622239e-06_inputformat=original_and_intervened_inplace_trainpolicy=ucp_valpolicy=ucp.pt' #'/mnt/qb/work/bethge/bkr046/CEM/checkpoints/concept_corrector_saved_models/celeba/ConceptBottleneckModel/sequential_split=0.pickle/best_models/NN_lr=0.014672197002812248_hidden_size=16_numlayers=2_batchsize=1024_weightdecay=1.5002716476657947e-05_inputformat=original_and_intervened_inplace_trainpolicy=random_intervention_policy_valpolicy=random_intervention_policy.pt'
# adapter_path = None
# dataset = 'celeba'
# # model_type = 'seq_cbm'
# model_type = 'ind_cbm'
# intervention_policy = 'ucp'
# title = 'Sequential CBM on CelebA Dataset' if model_type == 'seq_cbm' else 'Independent CBM on CelebA Dataset'

# CelebA CEM
predictions_dict_path = '/mnt/qb/work/bethge/bkr046/CEM/results/CelebA/predictions/ConceptEmbeddingModel/CEM_split=0.pickle'
concept_corrector_path = '/mnt/qb/work/bethge/bkr046/CEM/checkpoints/concept_corrector_saved_models/celeba/ConceptEmbeddingModel/CEM_split=0.pickle/best_models/NN_lr=0.01854411379208437_hidden_size=16_numlayers=1_batchsize=512_weightdecay=2.347108936382209e-06_inputformat=original_and_intervened_inplace_trainpolicy=ucp_valpolicy=ucp.pt'
adapter_path = None
dataset = 'celeba'
model_type = 'cem'
title = 'CEM on CelebA Dataset'
intervention_policy = 'ucp'

# CelebA Joint
# predictions_dict_path = '/mnt/qb/work/bethge/bkr046/CEM/results/CelebA/predictions/ConceptBottleneckModelJoint_Sigmoid_NoInterventionInTraining/joint_CBM_split=0.pickle'
# concept_corrector_path = '/mnt/qb/work/bethge/bkr046/CEM/checkpoints/concept_corrector_saved_models/celeba/ConceptBottleneckModelJoint_Sigmoid_NoInterventionInTraining/CEM_split=0.pickle/best_models/NN_lr=0.02505221815499305_hidden_size=8_numlayers=1_batchsize=512_weightdecay=2.0688588449397356e-06_inputformat=original_and_intervened_inplace_trainpolicy=ucp_valpolicy=ucp.pt'
# adapter_path = None
# dataset = 'celeba'
# model_type = 'joint_CBM'
# title = 'Joint CBM on CelebA Dataset'
# intervention_policy = 'ucp'

# CelebA IntCEM
# predictions_dict_path = '/mnt/qb/work/bethge/bkr046/CEM/results/CelebA/predictions/IntAwareConceptEmbeddingModelLastOnly_intervention_weight_5_horizon_rate_1.005_intervention_discount_1_tau_1_max_horizon_6_task_discount_1.1_uniform_distr_True/IntCEM_split=0.pickle'
# # concept_corrector_path = '/mnt/qb/work/bethge/bkr046/CEM/checkpoints/concept_corrector_saved_models/cub/ConceptBottleneckModelNoInterventionInTraining/sequential_split=0.pickle/best_models/NN_lr=0.02793541235937602_hidden_size=224_numlayers=3_batchsize=512_weightdecay=4.1084036364790424e-06_inputformat=original_and_intervened_inplace_trainpolicy=ucp_valpolicy=ucp.pt'
# adapter_path = None
# dataset = 'celeba'
# model_type = 'intcem'
# # model_type = 'ind_cbm'
# title = 'IntCEM on CelebA Dataset'


# # CUB CBM Seq
# # predictions_dict_path = '/mnt/qb/work/bethge/bkr046/CEM/results/predictions/ConceptBottleneckModelNoInterventionInTraining/sequential_split=1.pickle'
# predictions_dict_path = '/mnt/qb/work/bethge/bkr046/CEM/results/predictions/ConceptBottleneckModelNoInterventionInTraining/sequential_split=0.pickle'
# # predictions_dict_path = '/mnt/qb/work/bethge/bkr046/CEM/results/predictions/ConceptBottleneckModelNoInterventionInTraining/sequential_split=2.pickle'
# concept_corrector_path = '/mnt/qb/work/bethge/bkr046/CEM/checkpoints/concept_corrector_saved_models/cub/ConceptBottleneckModelNoInterventionInTraining/sequential_split=0.pickle/best_models/NN_lr=0.02793541235937602_hidden_size=224_numlayers=3_batchsize=512_weightdecay=4.1084036364790424e-06_inputformat=original_and_intervened_inplace_trainpolicy=ucp_valpolicy=ucp.pt'
# adapter_path = None
# dataset = 'cub'
# # model_type = 'seq_cbm'
# model_type = 'ind_cbm'
# title = 'Sequential CBM on CUB Dataset' if model_type == 'seq_cbm' else 'Independent CBM on CUB Dataset'
# intervention_policy = 'ucp'     # 'random_intervention_policy' or 'ectp'

# CUB CBM Seq Random Interventions
# predictions_dict_path = '/mnt/qb/work/bethge/bkr046/CEM/results/predictions/ConceptBottleneckModelNoInterventionInTraining/sequential_split=0.pickle'
# concept_corrector_path = '/mnt/qb/work/bethge/bkr046/CEM/checkpoints/concept_corrector_saved_models/cub/ConceptBottleneckModelNoInterventionInTraining/sequential_split=0.pickle/best_models/NN_lr=0.06397450669159863_hidden_size=224_numlayers=3_batchsize=256_weightdecay=4.061721133544051e-05_inputformat=original_and_intervened_inplace_trainpolicy=random_intervention_policy_valpolicy=random_intervention_policy.pt'
# adapter_path = None
# dataset = 'cub'
# # model_type = 'seq_cbm'
# model_type = 'ind_cbm'
# title = 'Sequential CBM on CUB Dataset\nRandom Interventions' if model_type == 'seq_cbm' else 'Independent CBM on CelebA Dataset'


# CUB CBM Seq - LSTM
# predictions_dict_path = '/mnt/qb/work/bethge/bkr046/CEM/results/predictions/ConceptBottleneckModelNoInterventionInTraining/sequential_split=0.pickle'
# concept_corrector_path = '/mnt/qb/work/bethge/bkr046/CEM/checkpoints/concept_corrector_saved_models/cub/ConceptBottleneckModelNoInterventionInTraining/sequential_split=0.pickle/best_models/LSTM_lr=0.06749965107057619_hidden_size=224_numlayers=1_batchsize=128_weightdecay=1.4999408807166958e-05_inputformat=original_and_intervened_inplace_trainpolicy=ucp_valpolicy=ucp.pt'

# # CUB CEM
# predictions_dict_path = '/mnt/qb/work/bethge/bkr046/CEM/results/predictions/ConceptEmbeddingModel/CEM_split=0.pickle'
# concept_corrector_path = '/mnt/qb/work/bethge/bkr046/CEM/checkpoints/concept_corrector_saved_models/cub/ConceptEmbeddingModel/CEM_split=0.pickle/best_models/NN_lr=0.050438648822426664_hidden_size=224_numlayers=2_batchsize=512_weightdecay=2.223977350538135e-05_inputformat=original_and_intervened_inplace_trainpolicy=ucp_valpolicy=ucp.pt'
# adapter_path = None
# dataset = 'cub'
# model_type = 'cem'
# title = 'CEM on CUB Dataset'
# intervention_policy = 'ucp'     # 'random_intervention_policy' or 'ectp'

# CUB IntCEM
# predictions_dict_path = '/mnt/qb/work/bethge/bkr046/CEM/results/predictions/IntAwareConceptEmbeddingModelNO_concept_corrector_Retry_intervention_weight_1_horizon_rate_1.005_intervention_discount_1_task_discount_1.1/IntCEM_split=0.pickle'
# concept_corrector_path = '/mnt/qb/work/bethge/bkr046/CEM/checkpoints/concept_corrector_saved_models/cub/IntAwareConceptEmbeddingModelNO_concept_corrector_Retry_intervention_weight_1_horizon_rate_1.005_intervention_discount_1_task_discount_1.1/IntCEM_split=0.pickle/best_models/NN_lr=0.034882454452949456_hidden_size=224_numlayers=3_batchsize=512_weightdecay=3.305996318271062e-06_inputformat=original_and_intervened_inplace_trainpolicy=ucp_valpolicy=ucp.pt'
# adapter_path = None
# dataset = 'cub'
# model_type = 'intcem'
# title = 'IntCEM on CUB Dataset'

# # CUB Joint CBM
# predictions_dict_path = '/mnt/qb/work/bethge/bkr046/CEM/results/predictions/ConceptBottleneckModelSigmoid_NoInterventionInTraining/joint_CBM_split=0.pickle'
# concept_corrector_path = '/mnt/qb/work/bethge/bkr046/CEM/checkpoints/concept_corrector_saved_models/cub/ConceptBottleneckModelSigmoid_NoInterventionInTraining/CEM_split=0.pickle/best_models/NN_lr=0.05958620306584017_hidden_size=224_numlayers=2_batchsize=512_weightdecay=2.0981575354788064e-06_inputformat=original_and_intervened_inplace_trainpolicy=ucp_valpolicy=ucp.pt'
# # concept_corrector_path1 = '/mnt/qb/work/bethge/bkr046/CEM/checkpoints/concept_corrector_saved_models/cub/ConceptBottleneckModelSigmoid_NoInterventionInTraining/CEM_split=0.pickle/best_models/NN_lr=0.05958620306584017_hidden_size=224_numlayers=2_batchsize=512_weightdecay=2.0981575354788064e-06_inputformat=original_and_intervened_inplace_trainpolicy=ucp_valpolicy=ucp.pt'
# # concept_corrector_path2 = '/mnt/qb/work/bethge/bkr046/CEM/checkpoints/concept_corrector_saved_models/cub/ConceptBottleneckModelSigmoid_NoInterventionInTraining/joint_CBM_split=1.pickle/best_models/NN_lr=0.022360206237438725_hidden_size=224_numlayers=3_batchsize=512_weightdecay=1.5798353765436667e-05_inputformat=original_and_intervened_inplace_trainpolicy=random_intervention_policy_valpolicy=random_intervention_policy.pt'
# # concept_corrector_path3 = '/mnt/qb/work/bethge/bkr046/CEM/checkpoints/concept_corrector_saved_models/cub/ConceptBottleneckModelSigmoid_NoInterventionInTraining/joint_CBM_split=2.pickle/best_models/NN_lr=0.03241950792392786_hidden_size=224_numlayers=3_batchsize=512_weightdecay=1.6667332121191513e-06_inputformat=original_and_intervened_inplace_trainpolicy=random_intervention_policy_valpolicy=random_intervention_policy.pt'
# # all_concept_corrector_paths = [concept_corrector_path1, concept_corrector_path2, concept_corrector_path3]
# adapter_path = None
# dataset = 'cub'
# model_type = 'joint_CBM'
# title = 'Joint CBM on CUB Dataset'
# intervention_policy = 'ucp'     # 'random_intervention_policy' or 'ectp'

# # AWA2 Seq
# predictions_dict_path = '/mnt/qb/work/bethge/bkr046/CEM/results/awa2/predictions/ConceptBottleneckModelNoInterventionInTraining/sequential_split=0.pickle'
# concept_corrector_path = '/mnt/qb/work/bethge/bkr046/CEM/checkpoints/concept_corrector_saved_models/awa2/ConceptBottleneckModelNoInterventionInTraining/sequential_split=0.pickle/best_models/NN_lr=0.07687684011313238_hidden_size=170_numlayers=2_batchsize=1024_weightdecay=8.144128638971219e-06_inputformat=original_and_intervened_inplace_trainpolicy=ucp_valpolicy=ucp.pt'
# adapter_path = None
# dataset = 'awa2'
# model_type = 'seq_cbm'
# # model_type = 'ind_cbm'
# title = 'Sequential CBM on AwA2 Dataset' if model_type == 'seq_cbm' else 'Independent CBM on AwA2 Dataset'

# AWA2 CEM
# predictions_dict_path = '/mnt/qb/work/bethge/bkr046/CEM/results/awa2/predictions/ConceptEmbeddingModel/CEM_split=0.pickle'
# concept_corrector_path = '/mnt/qb/work/bethge/bkr046/CEM/checkpoints/concept_corrector_saved_models/awa2/ConceptEmbeddingModel/CEM_split=0.pickle/best_models/NN_lr=0.06637754861758774_hidden_size=170_numlayers=3_batchsize=1024_weightdecay=2.4995432618330056e-05_inputformat=original_and_intervened_inplace_trainpolicy=ucp_valpolicy=ucp.pt'
# adapter_path = None
# dataset = 'awa2'
# model_type = 'cem'
# title = 'CEM on AWA2 Dataset'
# intervention_policy = 'ucp'

# AWA2 Joint CBM
# predictions_dict_path = '/mnt/qb/work/bethge/bkr046/CEM/results/awa2/predictions/ConceptBottleneckModelSigmoidNoInterventionInTraining/joint_CBM_split=0.pickle'
# concept_corrector_path = '/mnt/qb/work/bethge/bkr046/CEM/checkpoints/concept_corrector_saved_models/awa2/ConceptBottleneckModelSigmoidNoInterventionInTraining/CEM_split=0.pickle/best_models/NN_lr=0.06212155084231236_hidden_size=170_numlayers=2_batchsize=1024_weightdecay=3.829159066073285e-05_inputformat=original_and_intervened_inplace_trainpolicy=ucp_valpolicy=ucp.pt'
# adapter_path = None
# dataset = 'awa2'
# model_type = 'joint_CBM'
# title = 'Joint CBM on AWA2 Dataset'
# intervention_policy = 'ucp'


with open(predictions_dict_path, 'rb') as f:
    predictions_dict = pickle.load(f)


# Load Model
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


if 'CEM' in predictions_dict['model_type']:
    test_dataset = TensorDataset(test_predictions['predictions_c'], test_predictions['groundtruth_c'], test_predictions['groundtruths_y'], \
                                 test_predictions['concept_i_on_KL_div'], test_predictions['concept_i_off_KL_div'], \
                                 test_predictions['positive_embeddings'], test_predictions['negative_embeddings'])

else:
    test_dataset = TensorDataset(test_predictions['predictions_c'], test_predictions['groundtruth_c'], test_predictions['groundtruths_y'], \
                                 test_predictions['concept_i_on_KL_div'], test_predictions['concept_i_off_KL_div'])

# Create data loaders for training and testing sets
batch_size = 1024
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Load Concept Corrector
nn_concept_corrector_model = torch.load(concept_corrector_path)

if isinstance(nn_concept_corrector_model, dict):
    import re
    from pathlib import Path

    hidden_size = int(re.search(r'hidden_size=(\d+)_', concept_corrector_path).group(1))
    hidden_layers = int(re.search(r'numlayers=(\d+)_', concept_corrector_path).group(1))
    input_format = re.search(r'inputformat=([a-zA-Z_]+)_', concept_corrector_path).group(1)

    corrector_filename = Path(concept_corrector_path).name
    corrector_model_type = corrector_filename.split('_')[0]

    print(corrector_model_type, hidden_size, hidden_layers, input_format)

    corrector_class = eval(f'{corrector_model_type}ConceptCorrector')

    nn_concept_corrector_model = corrector_class(input_size=num_concepts, hidden_size=hidden_size, hidden_layers=hidden_layers, output_size=num_concepts, input_format=input_format)
    nn_concept_corrector_model.load_state_dict(torch.load(concept_corrector_path))

nn_concept_corrector_model = nn_concept_corrector_model.to(device)    
nn_concept_corrector_model.eval()


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


# This is a baseline concept corrector, which basically does not do anything, just returns the concepts unchanged
class ReturnSameConcepts:
    def forward_single_timestep(self, inputs, already_intervened_concepts, original_predictions, hidden):
        return inputs, None

    def __call__(self, inputs, already_intervened_concepts, original_predictions, hidden):
        return inputs, None

    def prepare_initial_hidden(self, batch_size, device):
        return None

returnSameConcepts = ReturnSameConcepts()

concept_embeddings = ('CEM' in predictions_dict['model_type'])

# Run interventions without concept correction, collect concept loss and accuracy values
all_concept_loss_no_correction, all_accuracy_no_correction = [], []

for _ in range(num_trials):
    concept_loss_vs_num_interventions_no_correction, accuracy_vs_num_interventions_no_correction = compute_concept_loss_and_accuracy_vs_num_interventions(test_loader, group2concepts, returnSameConcepts, lambda x: None, eval(intervention_policy), label_predictor, max_interventions, device, criterion, concept_embeddings=concept_embeddings, adapter=adapter)
    all_concept_loss_no_correction.append(concept_loss_vs_num_interventions_no_correction)
    all_accuracy_no_correction.append(accuracy_vs_num_interventions_no_correction)

all_concept_loss_no_correction = np.vstack(all_concept_loss_no_correction)
all_accuracy_no_correction = np.vstack(all_accuracy_no_correction)


# Run plotting function with concept correction, collect concept loss and accuracy values
all_concept_loss_with_correction, all_accuracy_with_correction = [], []

for _ in range(num_trials):
    concept_loss_vs_num_interventions_NN_corrector, accuracy_vs_num_interventions_NN_corrector = compute_concept_loss_and_accuracy_vs_num_interventions(test_loader, group2concepts, nn_concept_corrector_model, lambda x: prepare_initial_hidden(x, 'NN', None), eval(intervention_policy), label_predictor, max_interventions, device, criterion, concept_embeddings=concept_embeddings, adapter=adapter)
    all_concept_loss_with_correction.append(concept_loss_vs_num_interventions_NN_corrector)
    all_accuracy_with_correction.append(accuracy_vs_num_interventions_NN_corrector)

all_concept_loss_with_correction = np.vstack(all_concept_loss_with_correction)
all_accuracy_with_correction = np.vstack(all_accuracy_with_correction)


# Create Plot
# Plot concept loss values
with plt.style.context('seaborn-v0_8-whitegrid', after_reset=True):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    
    generate_lineplot(all_concept_loss_no_correction, ax=ax, kwargs={'label': 'Without Concept Realignment', 'marker': 's', 'markersize': 8})
    print('auc = ', auc(all_concept_loss_no_correction))
    generate_lineplot(all_concept_loss_with_correction, ax=ax, kwargs={'label': 'With Concept Realignment (Ours)', 'marker': 's', 'markersize': 8})
    print('auc = ', auc(all_concept_loss_with_correction))

    plt.xlabel("Number of Intervened Concepts", fontsize=26)
    plt.ylabel("Concept Loss", fontsize=26)
    
    plt.title(title, fontsize=26)
    plt.legend(fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=26)

    plt.savefig(f'figs/concept_loss_{dataset}_{model_type}.pdf', bbox_inches='tight')
    plt.show()


# Plot accuracy values
from matplotlib import pyplot as plt

with plt.style.context('seaborn-v0_8-whitegrid', after_reset=True):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

    combined = np.hstack((all_accuracy_no_correction, all_accuracy_with_correction))
    # min_v, max_v = np.min(combined), np.max(combined)
    min_v, max_v = None, None

    generate_lineplot(all_accuracy_no_correction*100, ax=ax, kwargs={'label': 'Without Concept Realignment', 'marker': 's', 'markersize': 8})
    print('auc = ', auc(all_accuracy_no_correction*100, min_v, max_v))
    generate_lineplot(all_accuracy_with_correction*100, ax=ax, kwargs={'label': 'With Concept Realignment (Ours)', 'marker': 's', 'markersize': 8})
    print('auc = ', auc(all_accuracy_with_correction*100, min_v, max_v))

    plt.xlabel("Number of Intervened Concepts", fontsize=26)
    plt.ylabel("Classification Accuracy (%)", fontsize=26)
    
    plt.title(title, fontsize=26)
    plt.legend(fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=26)

    if dataset == 'awa2':
        plt.xlim(-1, 50)

    plt.savefig(f'figs/accuracy_{dataset}_{model_type}.pdf', bbox_inches='tight')
    plt.show()

