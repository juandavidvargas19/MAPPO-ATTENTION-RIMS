#!/bin/bash
#EXAMPLE: 
#./RIMS.sh TERRITORY_I 10 LSTM 100 1 1 101 1
#./RIMS.sh TERRITORY_I 10 RIM 100 6 3 102 1

#pip install omegaconf ignite pytorch-ignite torchvision scikit-learn pytorch_lightning timm

# Check for the number of inputs and set defaults if necessary
environment=${1:-default_environment}
seed=${2:-default_seed}
module=${3:-default_module}
hidden=${4:-default_hidden}
units=${5:-default_units}
topk=${6:-default_topk}
run=${7:-default_run}
rollout=${8:-default_rollout}

general_dir="/home/ubunto/ICML-RUNS/CODES"

repo=$general_dir/MAPPO-ATTENTION-RIMS

# Default values in case the environment does not match
agents=0
substrate=""
episode_length=0
bottom=0
sup=0

export PYTHONPATH="$PYTHONPATH:$repo"
export CUDA_HOME=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/12.2.2
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME



# Set agents and substrate name based on the environment
case "$environment" in
    "HARVEST")
        echo "RUNNING HARVEST"
        agents=16
        substrate="allelopathic_harvest__open"
        episode_length=2000
        bottom=32
        sup=24
        ;;
    "TERRITORY_O" | "TERRITORY_R" | "TERRITORY_I")
        case "$environment" in
            "TERRITORY_O")
                echo "RUNNING territory__open"
                substrate="territory__open"
                agents=9
                bottom=18  # Adjust these values as necessary for the environment
                sup=15
                ;;
            "TERRITORY_R")
                echo "RUNNING territory__rooms"
                substrate="territory__rooms"
                agents=9
                bottom=18  # Adjust these values as necessary for the environment
                sup=15
                ;;
            "TERRITORY_I")
                echo "RUNNING territory__inside_out"
                substrate="territory__inside_out"
                agents=5
                bottom=10  # Adjust these values as necessary for the environment
                sup=8
                ;;
        esac
        episode_length=1000
        ;;
    "PREDATOR_RF" | "PREDATOR_O" | "PREDATOR_AH" | "PREDATOR_OR")
        case "$environment" in
            "PREDATOR_RF")
                echo "RUNNING predator_prey__random_forest"
                substrate="predator_prey__random_forest"
                agents=12
                bottom=24  # Adjust these values as necessary for the environment
                sup=18
                ;;
            "PREDATOR_O")
                echo "RUNNING predator_prey__open"
                substrate="predator_prey__open"
                agents=12
                bottom=24  # Adjust these values as necessary for the environment
                sup=18
                ;;
            "PREDATOR_AH")
                echo "RUNNING predator_prey__alley_hunt"
                substrate="predator_prey__alley_hunt"
                agents=10
                bottom=20  # Adjust these values as necessary for the environment
                sup=15
                ;;
            "PREDATOR_OR")
                echo "RUNNING predator_prey__orchard"
                substrate="predator_prey__orchard"
                agents=12
                bottom=24  # Adjust these values as necessary for the environment
                sup=18
                ;;
        esac
        episode_length=1000
        ;;
    "CHICKEN_A" | "CHICKEN_R")
        case "$environment" in
            "CHICKEN_A")
                echo "RUNNING chicken_in_the_matrix__arena"
                substrate="chicken_in_the_matrix__arena"
                agents=8
                bottom=16  # Adjust these values as necessary for the environment
                sup=12
                ;;
            "CHICKEN_R")
                echo "RUNNING chicken_in_the_matrix__repeated"
                substrate="chicken_in_the_matrix__repeated"
                agents=2
                bottom=4  # Adjust these values as necessary for the environment
                sup=3
                ;;
        esac
        episode_length=1000
        ;;
    "COOKING_ASY" | "COOKING_CRAMPED" | "COOKING_RING" | "COOKING_CIRCUIT" | "COOKING_FORCED" | "COOKING_EIGHT" | "COOKING_CROWDED")
        case "$environment" in
            "COOKING_ASY")
                echo "RUNNING collaborative_cooking__asymmetric"
                substrate="collaborative_cooking__asymmetric"
                agents=2
                bottom=4  # Adjust these values as necessary for the environment
                sup=3
                ;;
            "COOKING_CRAMPED")
                echo "RUNNING collaborative_cooking__cramped"
                substrate="collaborative_cooking__cramped"
                agents=2
                bottom=4  # Adjust these values as necessary for the environment
                sup=3
                ;;
            "COOKING_RING")
                echo "RUNNING collaborative_cooking__ring"
                substrate="collaborative_cooking__ring"
                agents=2
                bottom=4  # Adjust these values as necessary for the environment
                sup=3
                ;;
            "COOKING_CIRCUIT")
                echo "RUNNING collaborative_cooking__circuit"
                substrate="collaborative_cooking__circuit"
                agents=2
                bottom=4  # Adjust these values as necessary for the environment
                sup=3
                ;;
            "COOKING_FORCED")
                echo "RUNNING collaborative_cooking__forced"
                substrate="collaborative_cooking__forced"
                agents=2
                bottom=4  # Adjust these values as necessary for the environment
                sup=3
                ;;
            "COOKING_EIGHT")
                echo "RUNNING collaborative_cooking__figure_eight"
                substrate="collaborative_cooking__figure_eight"
                agents=6
                bottom=12  # Adjust these values as necessary for the environment
                sup=9
                ;;
            "COOKING_CROWDED")
                echo "RUNNING collaborative_cooking__crowded"
                substrate="collaborative_cooking__crowded"
                agents=9
                bottom=18  # Adjust these values as necessary for the environment
                sup=14
                ;;
        esac
        episode_length=1000
        ;;
    "SCISSORS_A" | "SCISSORS_R" | "SCISSORS_O")
        case "$environment" in
            "SCISSORS_A")
                echo "RUNNING running_with_scissors_in_the_matrix__arena"
                substrate="running_with_scissors_in_the_matrix__arena"
                agents=8
                bottom=16  # Adjust these values as necessary for the environment
                sup=12
                ;;
            "SCISSORS_R")
                echo "RUNNING running_with_scissors_in_the_matrix__repeated"
                substrate="running_with_scissors_in_the_matrix__repeated"
                agents=2
                bottom=4  # Adjust these values as necessary for the environment
                sup=3
                ;;
            "SCISSORS_O")
                echo "RUNNING running_with_scissors_in_the_matrix__one_shot"
                substrate="running_with_scissors_in_the_matrix__one_shot"
                agents=2
                bottom=4  # Adjust these values as necessary for the environment
                sup=3
                ;;
        esac
        episode_length=1000
        ;;
    "STAG_HUNT")
        echo "RUNNING stag_hunt_in_the_matrix__arena"
        substrate="stag_hunt_in_the_matrix__arena"
        agents=8
        episode_length=1000
        bottom=16  # Adjust these values as necessary for the environment
        sup=12
        ;;
    "CLEAN")
        echo "RUNNING clean_up"
        agents=7
        substrate="clean_up"
        episode_length=1000
        bottom=14
        sup=11
        ;;
    "STRAVINSKY")
        echo "RUNNING bach or stravinsky"
        agents=8
        substrate="bach_or_stravinsky_in_the_matrix__arena"
        episode_length=1000
        bottom=16
        sup=12
        ;;
    "COOKING")
        echo "RUNNING coolaborative cooking"
        agents=2
        substrate="collaborative_cooking__cramped"
        episode_length=1000
        bottom=4
        sup=3
        ;;
    "RATIONAL_C")
        echo "RUNNING rationalizable_coordination_in_the_matrix_repeated"
        substrate="rationalizable_coordination_in_the_matrix__repeated"
        agents=2
        episode_length=1000
        bottom=4
        sup=3
        ;;
    "FACTORY_C")
        echo "RUNNING factory_commons_either_or"
        substrate="factory_commons__either_or"
        agents=3
        episode_length=1000
        bottom=6
        sup=5
        ;;
    "HARVEST_P" | "HARVEST_C" | "HARVEST_O")
        case "$environment" in
            "HARVEST_P")
                echo "RUNNING commons_harvest_partnership"
                substrate="commons_harvest__partnership"
                agents=4
                bottom=8
                sup=6
                ;;
            "HARVEST_C")
                echo "RUNNING commons_harvest_closed"
                substrate="commons_harvest__closed"
                agents=6
                bottom=12
                sup=9
                ;;
            "HARVEST_O")
                echo "RUNNING commons_harvest_open"
                substrate="commons_harvest_open"
                agents=7
                bottom=14
                sup=11
                ;;
        esac
        episode_length=1000
        ;;
    "CHEMISTRY_2D" | "CHEMISTRY_3D" )
        case "$environment" in
            "CHEMISTRY_2D")
                echo "RUNNING chemistry_two_metabolic_cycles_with_distractors"
                substrate="chemistry__two_metabolic_cycles_with_distractors"

                ;;
            "CHEMISTRY_3D")
                echo "RUNNING chemistry__three_metabolic_cycles_with_plentiful_distractors"
                substrate="chemistry__three_metabolic_cycles_with_plentiful_distractors"
                ;;
        esac
        agents=8
        bottom=16
        sup=12
        episode_length=1000
        ;;
    "PURE_COORD")
        echo "RUNNING pure_coordination_in_the_matrix_repeated"
        substrate="pure_coordination_in_the_matrix_repeated"
        agents=2
        episode_length=1000
        bottom=4
        sup=3
        ;;
    "PRISONERS_A" | "PRISONERS_R" )
        case "$environment" in
            "PRISONERS_A")
                echo "RUNNING prisoners_dilemma_in_the_matrix__arena"
                agents=8
                substrate="prisoners_dilemma_in_the_matrix__arena"
                bottom=16
                sup=12
                ;;
            "PRISONERS_R")
                echo "RUNNING prisoners_dilemma_in_the_matrix_repeated"
                substrate="prisoners_dilemma_in_the_matrix__repeated"
                agents=2
                bottom=4
                sup=3
                ;;
        esac
        episode_length=1000
        ;;
    "DAYCARE")
        echo "RUNNING daycare"
        substrate="daycare"
        agents=2
        episode_length=1000
        bottom=4
        sup=3
        ;;
    *)
        echo "Unknown environment: $environment"
        exit 1
        ;;
esac

echo "Agents: $agents"
echo "Substrate: $substrate"
echo "Environment: $environment"
echo "Seed: $seed"
echo "Module: $module"
echo "Hidden: $hidden"
echo "Modules: $units"
echo "Topk: $topk"
echo "Run number: $run"

echo "directory is $repo"


if ["$substrate" = "allelopathic_harvest__open" | "$substrate"="rationalizable_coordination_in_the_matrix__repeated"]; then
    if ["$substrate" = "allelopathic_harvest__open"]; then
        episode_length=2000
    else
        episode_length=1000
    fi
    echo "fixed len substrate"
fi

entropy=0.006
lr_actor=0.00009
lr_critic=0.0001
env_steps=20000

if ls $repo/onpolicy/scripts/results/Meltingpot/collaborative_cooking__circuit_0/$substrate/mappo/check/run$run/models/*.pt 1> /dev/null 2>&1; then
    echo "Pre-trained Model exists"
    load_model=True
    path_dir=$repo/onpolicy/scripts/results/Meltingpot/collaborative_cooking__circuit_0/$substrate/mappo/check/run$run/models
else
    echo "Pre-trained Model does not exist"
    load_model="False"
    path_dir=None
fi

# Execute the program based on the module
if [ "$module" = "RIM" ]; then
    echo "Executing program for RIM"
    CUDA_VISIBLE_DEVICES=0 python $repo/onpolicy/scripts/train/train_meltingpot.py --load_model ${load_model} --model_dir ${path_dir} --num_env_steps ${env_steps} --run_num ${run} --rim_num_units ${units} --rim_topk ${topk} --use_valuenorm False --use_popart True --env_name "Meltingpot" --experiment_name "check" --substrate_name "${substrate}" --num_agents ${agents} --seed ${seed} --n_rollout_threads ${rollout} --use_wandb False --share_policy False --use_centralized_V False --use_attention True --use_naive_recurrent_policy False --use_recurrent_policy False --hidden_size ${hidden} --use_gae True --episode_length ${episode_length} --attention_module ${module} --algorithm_name mappo 

elif [ "$module" = "SCOFF" ]; then
    echo "Executing program for SCOFF"
    CUDA_VISIBLE_DEVICES=0 python $repo/onpolicy/scripts/train/train_meltingpot.py --load_model ${load_model} --model_dir ${path_dir} --num_env_steps ${env_steps} --run_num ${run} --scoff_num_units ${units} --scoff_topk ${topk} --use_valuenorm False --use_popart True --env_name "Meltingpot" --experiment_name "check" --substrate_name "${substrate}" --num_agents ${agents} --seed ${seed} --n_rollout_threads ${rollout} --use_wandb False --share_policy False --use_centralized_V False --use_attention True --use_naive_recurrent_policy False --use_recurrent_policy False --hidden_size ${hidden} --use_gae True --episode_length ${episode_length} --attention_module ${module} --algorithm_name mappo 
elif [ "$module" = "LSTM" ]; then
    echo "Executing program for LSTM"
    CUDA_VISIBLE_DEVICES=0 python $repo/onpolicy/scripts/train/train_meltingpot.py --load_model ${load_model} --model_dir ${path_dir} --num_env_steps ${env_steps} --run_num ${run} --use_valuenorm False --use_popart True --env_name "Meltingpot" --experiment_name "check" --substrate_name "${substrate}" --num_agents ${agents} --seed ${seed} --n_rollout_threads ${rollout} --use_wandb False --share_policy False --use_centralized_V False --use_attention False --use_naive_recurrent_policy True --use_recurrent_policy True --hidden_size ${hidden} --use_gae True --episode_length ${episode_length} --attention_module ${module} --algorithm_name mappo
else
    echo "Module is neither RIM nor SCOFF, nor LSTM"
fi
