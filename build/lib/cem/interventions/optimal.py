import numpy as np
import itertools
import torch
from .coop import CooP
from cem.interventions.intervention_policy import InterventionPolicy

class GreedyOptimal(CooP):
    def __init__(
        self,
        concept_group_map,
        cbm,
        n_tasks,
        num_groups_intervened=0,
        acquisition_weight=1,
        importance_weight=1,
        acquisition_costs=None,
        group_based=True,
        eps=1e-8,
        include_prior=True,
        **kwargs
    ):
        CooP.__init__(
            self,
            num_groups_intervened=num_groups_intervened,
            concept_group_map=concept_group_map,
            cbm=cbm,
            concept_entropy_weight=0,
            importance_weight=importance_weight,
            acquisition_weight=acquisition_weight,
            acquisition_costs=acquisition_costs,
            group_based=group_based,
            eps=eps,
            include_prior=include_prior,
            n_tasks=n_tasks,
            **kwargs,
        )
        self._optimal = True

class TrueOptimal(InterventionPolicy):
    def __init__(
        self,
        concept_group_map,
        cbm,
        num_groups_intervened=0,
        acquisition_costs=None,
        acquisition_weight=1,
        importance_weight=1,
        group_based=True,
        eps=1e-8,
        include_prior=True,
        **kwargs
    ):
        self.num_groups_intervened = num_groups_intervened
        self.concept_group_map = concept_group_map
        self.acquisition_costs = acquisition_costs
        self.group_based = group_based
        self._optimal = False
        self.cbm = cbm
        self.eps = eps
        self.acquisition_weight = acquisition_weight
        self.importance_weight = importance_weight
        self.include_prior = include_prior

    def _importance_scores(
        self,
        x,
        c,
        y,
        concepts_to_intervene,
        latent,
    ):
        # Then just look at the value of the probability of the known truth
        # class, as this is what we want to maximize!
        # See how the predictions change
        _, _, y_pred_logits, _, _ = self.cbm(
            x,
            intervention_idxs=concepts_to_intervene,
            c=c,
            latent=latent,
        )
        return np.array([
            y_pred_logits[i, label].detach().cpu().numpy()
            for i, label in enumerate(y)
        ])


    def _opt_score(
        self,
        x,
        c,
        y,
        concepts_to_intervene,
        latent,
    ):
        #  First compute the test accuracy for this intervention set
        importance_scores = self._importance_scores(
            x=x,
            c=c,
            y=y,
            concepts_to_intervene=concepts_to_intervene,
            latent=latent,
        )
        scores = self.importance_weight * importance_scores

        # Finally include the aquisition cost
        if self.acquisition_costs is not None:
            scores += self.acquisition_costs * self.acquisition_weight
        return scores

    def __call__(
        self,
        x,
        pred_c,
        c,
        y=None,
        competencies=None,
        prev_interventions=None,
        prior_distribution=None,
    ):
        if prev_interventions is None:
            mask = np.zeros((x.shape[0], c.shape[-1]), dtype=np.int64)
        else:
            mask = prev_interventions.detach().cpu().numpy()
        if not self.include_prior:
            prior_distribution = None
        if self.num_groups_intervened == 0:
            return mask, c
        _, _, _, _, latent = self.cbm(x)
        scores = []
        intervened_concepts = []
        concept_group_names = list(self.concept_group_map.keys())
        for intervention_idxs in itertools.combinations(
            set(range(len(concept_group_names))),
            self.num_groups_intervened,
        ):
            real_intervention_idxs = []
            for group_idx in intervention_idxs:
                real_intervention_idxs.extend(
                    self.concept_group_map[concept_group_names[group_idx]]
                )
            intervention_idxs = sorted(real_intervention_idxs)
            intervened_concepts.append(intervention_idxs)
            current_scores = self._opt_score(
                x=x,
                c=c,
                y=y,
                concepts_to_intervene=intervention_idxs,
                latent=latent,
            )
            scores.append(np.expand_dims(current_scores, axis=-1))
        scores = np.concatenate(scores, axis=-1)
        best_scores = np.argmax(scores, axis=-1)
        mask = np.zeros(c.shape, dtype=np.int32)
        for sample_idx in range(x.shape[0]):
            best_score_idx = best_scores[sample_idx]
            # Set the concepts of the best-scored model to be intervened
            # for this sample
            curr_mask = np.zeros((c.shape[-1],), dtype=np.int32)
            for idx in intervened_concepts[best_score_idx]:
                mask[sample_idx, idx] = 1
        return mask, c
