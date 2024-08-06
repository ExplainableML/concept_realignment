import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from train_utils import sample_trajectory, compute_loss


def generate_lineplot(my_array, ax, kwargs):
    '''
    expects tensor of shape num_trials x max_interventions
    '''
    num_trials, max_interventions = my_array.shape

    trial_column = []
    num_interventions_column = []
    value_column = []

    for trial in range(num_trials):
        for num_interventions in range(max_interventions):
            trial_column.append(trial)
            num_interventions_column.append(num_interventions)
            value_column.append(my_array[trial, num_interventions])

    # Create a Pandas DataFrame
    df = pd.DataFrame({
        'trial': trial_column,
        'num_interventions': num_interventions_column,
        'value': value_column
    })

    sns.set_theme()
    sns.lineplot(data=df, x='num_interventions', y='value', ax=ax, **kwargs)


def area_under_curve(x, y, min_v, max_v):
    if min_v is not None:
        y = (y - min_v) / max_v

    area = 0
    for i in range(1, len(x)):
        # Calculate the width of the trapezoid
        width = x[i] - x[i - 1]

        # Calculate the average height of the trapezoid
        avg_height = (y[i] + y[i - 1]) / 2

        # Calculate the area of the trapezoid and add it to the total area
        area += width * avg_height

    return area


def auc(array, min_v=None, max_v=None):
    return area_under_curve(list(range(array.shape[1])), np.mean(array, axis=0), min_v, max_v)


def expand_tensor(original_tensor, N):
    # this takes in a tensor of shape (A, B, C) and returns a tensor of shape (A, N, B, C)
    # by copying the original tensor N times
    expanded_tensor = torch.unsqueeze(original_tensor, dim=1)
    tiled_tensor = torch.tile(expanded_tensor, (1, N, 1, 1))

    return tiled_tensor


def concepts2embeddings(concepts, positive_embeddings, negative_embeddings):
    '''
    concepts: (batch_size, num_intervention_steps, num_concepts)
    positive_embeddings: (batch_size, num_concepts, embedding_dim)

    we first have to make the shapes of these two tensors similar:
    batch_size, num_intervention_steps, num_concepts, embedding_dim
    '''
    batch_size, num_intervention_steps, num_concepts = concepts.size()

    positive_embeddings = expand_tensor(positive_embeddings, num_intervention_steps)
    negative_embeddings = expand_tensor(negative_embeddings, num_intervention_steps)
    concepts = torch.unsqueeze(concepts, dim=3).to(positive_embeddings.device)

    embeddings = (
            positive_embeddings * concepts +
            negative_embeddings * (1 - concepts)
    )

    return embeddings


def concepts2embeddingsSingleTimeStep(concepts, positive_embeddings, negative_embeddings):
    '''
    concepts: (batch_size, num_concepts)
    positive_embeddings: (batch_size, num_concepts, embedding_dim)

    we first have to make the shapes of these two tensors similar:
    batch_size, num_concepts, embedding_dim
    '''
    batch_size, num_concepts = concepts.size()

    concepts = torch.unsqueeze(concepts, dim=2).to(positive_embeddings.device)

    embeddings = (
            positive_embeddings * concepts +
            negative_embeddings * (1 - concepts)
    )

    return embeddings


def compute_concept_loss_and_accuracy_vs_num_interventions(dataloader, group2concepts, concept_corrector, get_hidden,
                                                           intervention_policy, c2y_model, max_interventions, device,
                                                           criterion, concept_embeddings=False,
                                                           use_original_predictions_for_policy=False, adapter=None):
    # concept_embedding should be set True for Concept Embedding Models (whenever concepts are represented by embeddings)
    all_predicted_concepts, all_groundtruth_concepts, all_correct = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            predicted_concepts, groundtruth_concepts, groundtruth_labels = batch[0], batch[1], batch[2]
            predicted_concepts, groundtruth_concepts = predicted_concepts.to(device), groundtruth_concepts.to(device)

            # update concepts using concept corrector
            hidden = concept_corrector.prepare_initial_hidden(predicted_concepts.size(0), device)
            # if concept_embeddings:
            #     positive_embeddings, negative_embeddings = batch[5], batch[6]
            #     embeddings = {'positive_embeddings': positive_embeddings, 'negative_embeddings': negative_embeddings}
            # else:
            #     embeddings = None
            # print('embeddings = ', embeddings.keys())

            c_on_KL_div, c_off_KL_div = batch[3], batch[4]
            KL_divergences = {'c_on_KL_div': c_on_KL_div, 'c_off_KL_div': c_off_KL_div}
            if adapter is not None:
                predicted_concepts = adapter.forward_single_timestep(predicted_concepts,
                                                                     torch.zeros_like(predicted_concepts),
                                                                     predicted_concepts, hidden)
                predicted_concepts = predicted_concepts[0]
            minibatch_inputs, minibatch_masks, minibatch_original_predictions, minibatch_groundtruths = sample_trajectory(
                concept_corrector, predicted_concepts, group2concepts, groundtruth_concepts, KL_divergences, hidden,
                intervention_policy, max_interventions,
                use_original_predictions_for_policy=use_original_predictions_for_policy)

            minibatch_outputs, _ = concept_corrector(minibatch_inputs, minibatch_masks, minibatch_original_predictions,
                                                     hidden)
            all_predicted_concepts.append(minibatch_outputs)
            all_groundtruth_concepts.append(minibatch_groundtruths)

            # use updated concepts to predict labels
            # minibatch_outputs is of shape (batch_size, num_intervention_steps, num_concepts)
            # first reshape it into (batch_size * num_intervention_steps, num_concepts) so that it can be passed to the label predictor
            batch_size, num_intervention_steps, num_concepts = minibatch_outputs.size()

            # if concepts are represented by scalars (so, not CEM)
            if concept_embeddings == False:
                c = minibatch_outputs.reshape((batch_size * num_intervention_steps, num_concepts)).to(device)
                y_pred_logits = c2y_model(c).cpu()

            else:
                positive_embeddings, negative_embeddings = batch[5], batch[6]
                embeddings = concepts2embeddings(minibatch_outputs, positive_embeddings, negative_embeddings)
                # embeddings are of the shape (batch_size, num_intervention_steps, num_concepts, embedding_dim)
                # to pass them through the model, we need to reshape into (-1, num_concepts*embedding_dim)
                (batch_size, num_intervention_steps, num_concepts, embedding_dim) = embeddings.size()
                embeddings_reshaped = torch.reshape(embeddings, (-1, num_concepts * embedding_dim)).to(device)
                print('USING EMBEDDINGS OF SHAPE: ', embeddings_reshaped.size())
                print(c2y_model)
                y_pred_logits = c2y_model(embeddings_reshaped).cpu()

            if y_pred_logits.size(1) > 1:
                y_pred_labels = torch.argmax(y_pred_logits, dim=-1)
            else:
                y_pred_labels = (torch.sigmoid(y_pred_logits) >= 0.5).float().flatten()

            # now, reshape it back into (batch_size, num_intervention_steps)
            y_pred_labels = y_pred_labels.reshape((batch_size, num_intervention_steps))

            # need to repeat groundtruth_labels to match the shape of y_pred_labels
            groundtruth_labels = groundtruth_labels.unsqueeze(1).expand(-1, y_pred_labels.size(1))

            correct = (y_pred_labels == groundtruth_labels)
            all_correct.append(correct)

    all_predicted_concepts = torch.cat(all_predicted_concepts, dim=0)
    all_groundtruth_concepts = torch.cat(all_groundtruth_concepts, dim=0)
    all_correct = torch.vstack(all_correct)

    concept_loss_vs_num_interventions = []

    for i in range(all_predicted_concepts.size(1)):
        loss = criterion(all_predicted_concepts[:, i, :], all_groundtruth_concepts[:, i, :])
        concept_loss_vs_num_interventions.append(loss.item())

    accuracy_vs_num_interventions = torch.mean(all_correct.float(), dim=0)

    return concept_loss_vs_num_interventions, accuracy_vs_num_interventions