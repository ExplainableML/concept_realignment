# SPECIFY WHERE THE CONDA ENVIRONMENTS ARE SAVED IN LINE 14
# SPECIFY THE PATH TO THE PRESENT DIRECTORY IN LINE 17
# SPECIFY THE PATH TO THE PREDICTIONS DICTIONARY AFTER --predictions_dict_path

#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <nickname>"
    exit 1
fi

nickname=$1

source /mnt/qb/work/bethge/bkr046/anaconda3/etc/profile.d/conda.sh
conda activate CEM

cd /home/bethge/bkr046/CBM-intervention-concept-correction/concept-realignment-experiments

# Run commands in parallel

case $nickname in
    "CUB_CEM")
        python3 train_concept_corrector_optuna.py --yaml_path configs/hyperparam_optimization/CUB/nn.yml --device_id 0 \
        --num_trials 50 --input_format original_and_intervened_inplace \
        --predictions_dict_path /mnt/qb/work/bethge/bkr046/CEM/results/predictions/ConceptEmbeddingModel/CEM_split=0.pickle
        ;;
    "CUB_IntCEM")
        python3 train_concept_corrector_optuna.py --yaml_path configs/hyperparam_optimization/CUB/nn.yml --device_id 0 \
        --num_trials 50 --input_format original_and_intervened_inplace \
        --predictions_dict_path /mnt/qb/work/bethge/bkr046/CEM/results/predictions/IntAwareConceptEmbeddingModelNO_concept_corrector_Retry_intervention_weight_1_horizon_rate_1.005_intervention_discount_1_task_discount_1.1/IntCEM_split=0.pickle
        ;;
    "CUB_Seq_CBM_LSTM_original_concepts")
        python3 train_concept_corrector_optuna.py --yaml_path configs/hyperparam_optimization/CUB/lstm.yml --device_id 0 \
        --num_trials 50 --input_format original_and_intervened_inplace \
        --predictions_dict_path /mnt/qb/work/bethge/bkr046/CEM/results/predictions/ConceptBottleneckModelNoInterventionInTraining/sequential_split=0.pickle
        ;;
    "CUB_Seq_CBM_LSTM_previous_concepts")
        python3 train_concept_corrector_optuna.py --yaml_path configs/hyperparam_optimization/CUB/lstm.yml --device_id 0 \
        --num_trials 50 --input_format previous_output \
        --predictions_dict_path /mnt/qb/work/bethge/bkr046/CEM/results/predictions/ConceptBottleneckModelNoInterventionInTraining/sequential_split=0.pickle
        ;;
    "CUB_Seq_CBM_NN_previous_concepts")
        python3 train_concept_corrector_optuna.py --yaml_path configs/hyperparam_optimization/CUB/nn.yml --device_id 0 \
        --num_trials 50 --input_format previous_output \
        --predictions_dict_path /mnt/qb/work/bethge/bkr046/CEM/results/predictions/ConceptBottleneckModelNoInterventionInTraining/sequential_split=0.pickle
        ;;
    "CUB_Seq_CBM")
        python3 train_concept_corrector_optuna.py --yaml_path configs/hyperparam_optimization/CUB/nn.yml --device_id 0 \
        --num_trials 50 --input_format original_and_intervened_inplace \
        --predictions_dict_path /mnt/qb/work/bethge/bkr046/CEM/results/predictions/ConceptBottleneckModelNoInterventionInTraining/sequential_split=0.pickle
        ;;
    "CUB_Ind_CBM")
        python3 train_concept_corrector_optuna.py --yaml_path configs/hyperparam_optimization/CUB/nn.yml --device_id 0 \
        --num_trials 50 --input_format original_and_intervened_inplace \
        --predictions_dict_path /mnt/qb/work/bethge/bkr046/CEM/results/predictions/ConceptBottleneckModelNoInterventionInTraining/independent_split=0.pickle
        ;;
    "CUB_Joint_CBM")
        python3 train_concept_corrector_optuna.py --yaml_path configs/hyperparam_optimization/CUB/nn.yml --device_id 0 \
        --num_trials 50 --input_format original_and_intervened_inplace \
        --predictions_dict_path /mnt/qb/work/bethge/bkr046/CEM/results/predictions/ConceptBottleneckModelSigmoid_NoInterventionInTraining/joint_CBM_split=2.pickle
        #--predictions_dict_path /mnt/qb/work/bethge/bkr046/CEM/results/predictions/ConceptBottleneckModelSigmoid_NoInterventionInTraining/CEM_split=0.pickle
        ;;
    "CelebA_Seq_CBM")
        python3 train_concept_corrector_optuna.py --yaml_path configs/hyperparam_optimization/CelebA/nn.yml --device_id 0 \
        --num_trials 50 --input_format original_and_intervened_inplace \
        --predictions_dict_path /mnt/qb/work/bethge/bkr046/CEM/results/CelebA/predictions/ConceptBottleneckModelresnet34/sequential_split=0.pickle
        ;;
    "CelebA_Joint_CBM")
        python3 train_concept_corrector_optuna.py --yaml_path configs/hyperparam_optimization/CelebA/nn.yml --device_id 0 \
        --num_trials 50 --input_format original_and_intervened_inplace \
        --predictions_dict_path /mnt/qb/work/bethge/bkr046/CEM/results/CelebA/predictions/ConceptBottleneckModelJoint_Sigmoid_NoInterventionInTraining/CEM_split=0.pickle
        ;;
    "CelebA_CEM")
        python3 train_concept_corrector_optuna.py --yaml_path configs/hyperparam_optimization/CelebA/nn.yml --device_id 0 \
        --num_trials 50 --input_format original_and_intervened_inplace \
        --predictions_dict_path /mnt/qb/work/bethge/bkr046/CEM/results/CelebA/predictions/ConceptEmbeddingModel/CEM_split=0.pickle
        ;;
    "AWA_Seq_CBM")
        python3 train_concept_corrector_optuna.py --yaml_path configs/hyperparam_optimization/AWA2/nn.yml --device_id 0 \
        --num_trials 50 --input_format original_and_intervened_inplace \
        --predictions_dict_path /mnt/qb/work/bethge/bkr046/CEM/results/awa2/predictions/ConceptBottleneckModelNoInterventionInTraining/sequential_split=0.pickle
        ;;
    "AWA_Joint_CBM")
        python3 train_concept_corrector_optuna.py --yaml_path configs/hyperparam_optimization/AWA2/nn.yml --device_id 0 \
        --num_trials 50 --input_format original_and_intervened_inplace \
        --predictions_dict_path /mnt/qb/work/bethge/bkr046/CEM/results/awa2/predictions/ConceptBottleneckModelSigmoidNoInterventionInTraining/CEM_split=0.pickle
        ;;
    "AWA_CEM")
        python3 train_concept_corrector_optuna.py --yaml_path configs/hyperparam_optimization/AWA2/nn.yml --device_id 0 \
        --num_trials 50 --input_format original_and_intervened_inplace \
        --predictions_dict_path /mnt/qb/work/bethge/bkr046/CEM/results/awa2/predictions/ConceptEmbeddingModel/CEM_split=0.pickle
        ;;
    *)
        echo "Invalid nickname: $nickname"
        exit 1
        ;;
esac