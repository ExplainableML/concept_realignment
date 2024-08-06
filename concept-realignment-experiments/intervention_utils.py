import torch


# Intervention Policies
def ucp(concepts, already_intervened_concepts, KL_divergences, add_noise=False):
    '''
    concepts: 2D torch tensor of shape (batch_size, num_concepts)
    already_intervened_concepts: 2D torch tensor of shape (batch_size, num_concepts). 1 if concept has been intervened on, 0 otherwise.
    '''
    importances = 1/(torch.abs(concepts - 0.5) + 1e-8)
    if add_noise:
        importances += torch.rand_like(concepts) * 0.2
    importances[already_intervened_concepts == 1] = -1e10
    # importances *= (1-already_intervened_concepts)  # if already_intervened_concepts == 1, importance should be zero

    # return torch.argmax(importances, dim=1)
    return importances


def random_intervention_policy(concepts, already_intervened_concepts, KL_divergences, add_noise=False):
    '''
    concepts: 2D torch tensor of shape (batch_size, num_concepts)
    already_intervened_concepts: 2D torch tensor of shape (batch_size, num_concepts). 1 if concept has been intervened on, 0 otherwise.
    '''
    # this is similar to ucp, but importances are assigned randomly
    importances = torch.rand_like(concepts)

    # if concept has already been intervened on, then importance should be -1
    importances[already_intervened_concepts == 1] = -1e10
    # importances = (1 - already_intervened_concepts) * importances + already_intervened_concepts * (-1)

    # return torch.argmax(importances, dim=1)
    return importances


# KL_divergences is a dict of KL divergences when a concept is turned on 
# and when the concept is turned off
def ectp(concepts, already_intervened_concepts, KL_divergences, add_noise=False):
    num_concepts = concepts.size(1)

    c_on_KL_div, c_off_KL_div = KL_divergences['c_on_KL_div'], KL_divergences['c_off_KL_div']
    importances = (1 - concepts) * c_off_KL_div + concepts * c_on_KL_div
    if add_noise:
        importances += torch.rand_like(concepts) * 0.2
    # importances[already_intervened_concepts == 1] = -float('inf')
    importances[already_intervened_concepts == 1] = -1e10
    
    # return torch.argmax(importances, dim=1)
    return importances


# Function to Apply Intervention Policy
def intervene(concepts, original_concepts, group2concepts, already_intervened_concepts, groundtruth_concepts, KL_divergences, intervention_policy, add_noise, use_original_concepts_for_policy=False, return_selected_concepts=False):
    if use_original_concepts_for_policy:
        importances = intervention_policy(original_concepts, already_intervened_concepts, KL_divergences, add_noise)
    else:
        importances = intervention_policy(concepts, already_intervened_concepts, KL_divergences, add_noise)
    
    if group2concepts is None:
        # concepts_to_intervene = intervention_policy(concepts, concept_map, already_intervened_concepts, KL_divergences, add_noise)
        concepts_to_intervene = torch.argmax(importances, dim=1)
        
        concepts[range(concepts.size(0)), concepts_to_intervene] = groundtruth_concepts[range(concepts.size(0)), concepts_to_intervene]
        already_intervened_concepts[range(concepts.size(0)), concepts_to_intervene] = 1

        if not return_selected_concepts:
            return concepts, already_intervened_concepts
        else:
            return concepts, already_intervened_concepts, concepts_to_intervene

    else:
        # compute group importances
        concepts2group = group2concepts.t().clone() # size: num_concepts x num_groups
        concepts2group /= torch.sum(concepts2group, dim=0)
        assert torch.all(torch.isclose(torch.sum(concepts2group, dim=0), torch.ones_like(torch.sum(concepts2group, dim=0))))
        group_importances = importances @ concepts2group

        # select best group
        selected_groups = torch.argmax(group_importances, dim=1)    # this would be a 1D tensor of length batchsize
        # we want to convert it into a 2D tensor of size batch_size x num_groups such that if sample i has selected jth group
        # then selected_groups_matrix[i, j] = 1 and everything else 0
        selected_groups_matrix = torch.zeros_like(group_importances)
        selected_groups_matrix[range(concepts.size(0)), selected_groups] = 1

        # now project this from the space of groups back into the space of concepts
        # you will get a mask, which is 1 if a concept should be intervened and 0 otherwise
        mask = selected_groups_matrix @ group2concepts

        # now intervene
        concepts_new = mask * groundtruth_concepts + (1-mask) * concepts
        prev = already_intervened_concepts.clone()
        intervened_concepts_new = already_intervened_concepts + mask

        if torch.min(already_intervened_concepts) < 0 or torch.max(already_intervened_concepts) > 1:
            print('already intervened concepts out of range')
            print(torch.min(already_intervened_concepts), torch.max(already_intervened_concepts))
            print('range of previous: ')
            print(torch.min(prev), torch.max(prev))
            raise Exception("already intervened concepts out of range")

        if not return_selected_concepts:
            return concepts_new, intervened_concepts_new
        else:
            return concepts_new, intervened_concepts_new, selected_groups


def prepare_concept_map_tensors(concept_map_dict):
    '''
    concept_map_dict should be a dict of the form: {group_name: [idx of concepts in group]}
    should return:
    group2concepts of shape (num_groups x num_concepts)
    you can obtain concepts2group by transposing group2concepts
    '''
    print("preparing concept map tensor")
    
    if concept_map_dict is None:
        return None

    num_concepts = 0
    for concepts_in_group in concept_map_dict.values():
        num_concepts += len(concepts_in_group)
    print('number of concepts = ', num_concepts)
    
    group2concepts = [torch.zeros(num_concepts).scatter_(0, torch.tensor(concepts_in_group), 1) for concepts_in_group in concept_map_dict.values()]
    group2concepts = torch.vstack(group2concepts)

    assert group2concepts.size() == (len(concept_map_dict), num_concepts)

    return group2concepts


