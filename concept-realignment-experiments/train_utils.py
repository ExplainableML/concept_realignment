import torch
from intervention_utils import ucp, random_intervention_policy, intervene


def sample_trajectory(concept_corrector, concepts, group2concepts, groundtruth_concepts, KL_divergences, initial_hidden, intervention_policy, max_interventions, add_noise=False, use_original_predictions_for_policy=False):
    '''
    should take in as input the concept corrector model, original concept predictions, groundtruth concept predictions, and intervention policy
    then do a rollout of intervention --> concept correction --> intervention --> concept correction ...
    and return all_inputs, all_masks, all_groundtruths
    where all_inputs is [x1, x2, x3, ...], xi being the input to the concept correction model at time t
    and all_masks are the masks that are fed to the concept corrector along with all_inputs
    all_grountruths is [y1, y2, y3, ...] where each yi is the target value the concept corrector should output at time t
    KL_divergences is a dict of KL divergences by turning concepts on and off
    '''

    with torch.no_grad():
        all_inputs = []
        all_masks = []
        all_original_predictions = []
        all_groundtruths = []

        hidden = initial_hidden

        already_intervened_concepts = torch.zeros_like(concepts)

        original_predictions = concepts.detach().clone()
        
        all_inputs.append(concepts.detach().clone())
        all_masks.append(already_intervened_concepts.detach().clone())
        all_original_predictions.append(original_predictions)
        all_groundtruths.append(groundtruth_concepts.detach().clone())

        out, hidden = concept_corrector.forward_single_timestep(concepts, already_intervened_concepts, original_predictions, hidden)

        for num_interventions in range(max_interventions):
            # intervene
            concepts, already_intervened_concepts = intervene(concepts, original_predictions, group2concepts, already_intervened_concepts, groundtruth_concepts, KL_divergences, intervention_policy, add_noise, use_original_predictions_for_policy)
            if torch.max(already_intervened_concepts) > 1:
                import pickle
                with open('prev.pickle', 'wb') as f:
                    pickle.dump(prev, f)

            all_inputs.append(concepts.detach().clone())
            all_masks.append(already_intervened_concepts.detach().clone())
            all_original_predictions.append(original_predictions)
            all_groundtruths.append(groundtruth_concepts.detach().clone())
            
            # pass the concepts through concept corrector module
            updated_concepts, hidden = concept_corrector.forward_single_timestep(concepts, already_intervened_concepts, original_predictions, hidden)

            concepts = torch.squeeze(updated_concepts, dim=1)

        all_inputs, all_masks, all_original_predictions, all_groundtruths = torch.stack(all_inputs, dim=1), torch.stack(all_masks, dim=1), torch.stack(all_original_predictions, dim=1), torch.stack(all_groundtruths, dim=1)
        
        if torch.min(all_inputs) < 0 or torch.max(all_inputs) > 1:
            print("Inside Sample Trajectory All Inputs outside range")

        return all_inputs, all_masks, all_original_predictions, all_groundtruths


def compute_loss(concept_corrector, concepts, group2concepts, groundtruth_concepts, KL_divergences, initial_hidden, intervention_policy, max_interventions, criterion, add_noise=False):
    all_inputs, all_masks, all_original_predictions, all_groundtruths = sample_trajectory(concept_corrector, concepts, group2concepts, groundtruth_concepts, KL_divergences, initial_hidden, intervention_policy, max_interventions, add_noise)
    
    hidden = initial_hidden
    
    out, _ = concept_corrector(all_inputs, all_masks, all_original_predictions, hidden)
    
    return criterion(out, all_groundtruths)