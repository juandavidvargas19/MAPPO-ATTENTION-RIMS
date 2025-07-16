## Installation

Clone the repository and install the requirements:

```bash
#Download conda installer here: https://www.anaconda.com/download/success
bash $anacoda_file
conda create --name RIMS python=3.11.7
conda activate RIMS
git clone https://github.com/juandavidvargas19/MAPPO-ATTENTION-RIMS.git
cd MAPPO-ATTENTION-RIMS
pip install -r requirements.txt
pip install "ray[cpp]" 
git clone -b main https://github.com/deepmind/meltingpot
cd meltingpot
pip install --editable .[dev]
```

## Experiments

MARL experiments include Harvest Cleaner, Harvest Planter, Chemistry 3D, and Territory Inside Out environments. However there are more than 20 environments compatible that you can try.

For each of this cases, you need to change the generic variable = $general_dir to $local_repo_directory/MARL. Then for each case you need to run:


Standard run (1 RNN)
```bash
./RIMS.sh TERRITORY_I 1 LSTM 100 1 1 1 1
```

RIMS (multiple RNNs)
```bash
./RIMS.sh TERRITORY_I 1 RIM 100 5 2 2 1
```

You can change TERRITORY_I for the corresponding environment. Please see the script in meltingpot.sh to change the corresponding flag name for each environment.

The necessary inputs are as follows:
- environment=${1:-default_environment}
- seed=${2:-default_seed}
- module=${3:-default_module}
- hidden=${4:-default_hidden_size}
- units=${5:-default_number_rnn_modules}
- topk=${6:-default_topk}
- run=${7:-default_run}
- rollout=${8:-default_rollout}


## Questions and Answers


## 1. How can I modify the architecture of the Actor and the Critic?

There are 4 key components to the architecture: 

### 1. **r_mappo.py** - Training Algorithm
Implements the PPO training algorithm with loss computation, gradient updates, and entropy annealing. Controls the entire training loop and optimization process.

### 2. **rMAPPOPolicy.py** - Policy Interface  
Wraps the actor and critic networks, manages their optimizers, and provides unified methods for action selection and evaluation. Acts as the bridge between training and networks.

### 3. **r_actor_critic.py** - Network Architectures
Contains the actual neural network definitions for both actor and critic, including base feature extractors, attention mechanisms (RIM/SCOFF), and output layers.

### 4. **rnn.py / rim_cell.py** - Recurrent/Attention Modules
Implements the core computational units (GRU/LSTM in rnn.py or RIM attention mechanisms in rim_cell.py) that process sequential information within the actor and critic networks.

## 2. How can I modify the loss functions?

The loss functions can be found in onpolicy/algorithms/r_mappo/r_mappo.py.

### Actor Loss

**Location:** `ppo_update()` method, lines with `policy_loss` calculation

**Components Used:**
- **Clipped Surrogate Loss**: Main PPO clipped objective using importance sampling weights
- **Entropy Regularization**: Entropy coefficient (with cosine annealing schedule)

**Formula:**
```python
# Clipped surrogate loss
surr1 = imp_weights * adv_targ
surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

# Total actor loss
total_loss = (policy_loss - dist_entropy * self.entropy_coef)
```

### Critic Loss

**Location:** `cal_value_loss()` method

**Components Used:**
- **Value Function Loss**: Either MSE or Huber loss (configurable)
- **Value Clipping**: Optional clipped value loss
- **Value Normalization**: Optional PopArt or ValueNorm

**Formula:**
```python
# Value loss (MSE or Huber)
if self._use_huber_loss:
    value_loss_original = huber_loss(error_original, self.huber_delta)
else:
    value_loss_original = mse_loss(error_original)

# Total critic loss
total_critic_loss = (value_loss * self.value_loss_coef)
```

**Key Difference:** Actor and critic are trained separately with their own optimizers, not jointly.

## 3. How can I debug RIMs?


### Debug Location
- **Actor/Critic Architecture**: `onpolicy/algorithms/r_mappo/algorithm/r_actor_critic.py`
- **RIM Computation**: `onpolicy/algorithms/utils/rim_cell.py`

### Main RIM Components

#### 1. **RIMCell** - Core Processing Unit
- **Input Attention**: Selects top-k RIMs to activate based on attention scores
- **Independent Dynamics**: Each RIM runs its own GRU/LSTM independently  
- **Communication Attention**: Activated RIMs communicate through attention mechanism
- **Gradient Blocking**: Inactive RIMs have gradients blocked to maintain specialization

#### 2. **Key Methods to Debug**
- `input_attention_mask()` - Debug RIM selection and activation
- `communication_attention()` - Debug inter-RIM communication
- `forward()` - Debug overall flow and state updates

#### 3. **RIM Architecture Concept**
RIMs are modular recurrent units that:
- **Specialize** on different aspects of the input
- **Compete** for activation via attention (only top-k active)
- **Communicate** sparsely through attention bottleneck
- **Maintain** independent parameters and dynamics

The key insight: RIMs learn to specialize on different environmental factors, leading to better generalization when only some factors change between training and evaluation.


## 4. How can I debug the baseline LSTM?

### Debug Location
- **Actor/Critic Architecture**: `onpolicy/algorithms/r_mappo/algorithm/r_actor_critic.py`
- **RNN Computation**: `onpolicy/algorithms/utils/rnn.py`

### Main RNN Components

#### **RNNLayer** - Core Processing Unit
- **GRU Cell**: Uses `nn.GRU` with configurable layers (`recurrent_N`)
- **Initialization**: Orthogonal or Xavier uniform weight initialization
- **Layer Normalization**: Applied to output for stability

The baseline GRU processes sequences while properly handling episode boundaries through masking, unlike RIMs which add attention-based specialization and communication.


## 5. How can I add additional parameters?

- **Configuration File**: `onpolicy/config.py`

### Steps to Add New Parameters

#### 1. **Add Parser Argument**
In the `get_config()` function, add your new parameter using `parser.add_argument()`:

```python
parser.add_argument("--your_parameter_name", type=str, default="default_value",
                    help="Description of what this parameter does")
```

#### 2. **Parameter Types and Options**
- **String**: `type=str`
- **Integer**: `type=int`
- **Float**: `type=float`
- **Boolean**: `type=str2bool`
- **Choices**: `choices=["option1", "option2", "option3"]`


#### 3. **Accessing Parameters in Code**
After adding the parameter, access it anywhere in your code using:
```python
all_args.your_parameter_name
```