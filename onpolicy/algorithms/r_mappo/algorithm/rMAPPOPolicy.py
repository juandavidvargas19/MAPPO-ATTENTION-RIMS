import torch
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic
from onpolicy.utils.util import update_linear_schedule
import torch_optimizer as optim2


class R_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space
        self.optimizer=args.optimizer


        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)

        # actor_parameters = sum(p.numel() for p in self.actor.parameters() if p.requires_grad)
        # critic_parameters = sum(p.numel() for p in self.critic.parameters() if p.requires_grad)


        if self.optimizer == 'ADAM':
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                    lr=self.lr, eps=self.opti_eps,
                                                    weight_decay=self.weight_decay,amsgrad=False)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                     lr=self.critic_lr,
                                                     eps=self.opti_eps,
                                                     weight_decay=self.weight_decay,amsgrad=False)
        elif self.optimizer == 'AMS':
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                    lr=self.lr, eps=self.opti_eps,
                                                    weight_decay=self.weight_decay, amsgrad=True)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                     lr=self.critic_lr,
                                                     eps=self.opti_eps,
                                                     weight_decay=self.weight_decay, amsgrad=True)
        elif self.optimizer == 'ADAMW':
            self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(),
                                                    lr=self.lr, eps=self.opti_eps,
                                                    weight_decay=self.weight_decay,amsgrad=False)
            self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(),
                                                     lr=self.critic_lr,
                                                     eps=self.opti_eps,
                                                     weight_decay=self.weight_decay,amsgrad=False)
        elif self.optimizer == 'AMSW':
            self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(),
                                                    lr=self.lr, eps=self.opti_eps,
                                                    weight_decay=self.weight_decay, amsgrad=True)
            self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(),
                                                     lr=self.critic_lr,
                                                     eps=self.opti_eps,
                                                     weight_decay=self.weight_decay, amsgrad=True)
        elif self.optimizer=='ADAMAX':
            self.actor_optimizer = torch.optim.Adamax(self.actor.parameters(),
                                                      lr=self.lr, eps=self.opti_eps,
                                                      weight_decay=self.weight_decay)
            self.critic_optimizer = torch.optim.Adamax(self.critic.parameters(),
                                                       lr=self.critic_lr,
                                                       eps=self.opti_eps,
                                                       weight_decay=self.weight_decay)  
        elif self.optimizer == 'RMS':
            self.actor_optimizer = torch.optim.RMSprop(self.actor.parameters(),
                                                      lr=self.lr, eps=self.opti_eps,
                                                      weight_decay=self.weight_decay)
            self.critic_optimizer = torch.optim.RMSprop(self.critic.parameters(),
                                                       lr=self.critic_lr,
                                                       eps=self.opti_eps,
                                                       weight_decay=self.weight_decay)  
        elif self.optimizer == 'SGD':
            self.actor_optimizer = torch.optim.SGD(self.actor.parameters(),
                                                        lr=self.lr, momentum=0.9,
                                                        weight_decay=self.weight_decay)
            self.critic_optimizer = torch.optim.SGD(self.critic.parameters(),
                                                         lr=self.critic_lr, momentum=0.9,
                                                         weight_decay=self.weight_decay)
            
        elif self.optimizer == 'POO':
            self.actor_optimizer = optim2.Shampoo(self.actor.parameters(),
                                                      lr=self.lr, epsilon=self.opti_eps,
                                                      weight_decay=self.weight_decay)
            self.critic_optimizer = optim2.Shampoo(self.critic.parameters(),
                                                       lr=self.critic_lr,
                                                       epsilon=self.opti_eps,
                                                       weight_decay=self.weight_decay)
        elif self.optimizer == 'SWT':
            self.actor_optimizer = optim2.SWATS(self.actor.parameters(),
                                                      lr=self.lr, eps=self.opti_eps,
                                                      weight_decay=self.weight_decay)   
            self.critic_optimizer = optim2.SWATS(self.critic.parameters(),
                                                       lr=self.critic_lr,
                                                       eps=self.opti_eps,
                                                       weight_decay=self.weight_decay)
            

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    # def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
    #                 deterministic=False):
    #     """
    #     Compute actions and value function predictions for the given inputs.
    #     :param cent_obs (np.ndarray): centralized input to the critic.
    #     :param obs (np.ndarray): local agent inputs to the actor.
    #     :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
    #     :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
    #     :param masks: (np.ndarray) denotes points at which RNN states should be reset.
    #     :param available_actions: (np.ndarray) denotes which actions are available to agent
    #                               (if None, all actions available)
    #     :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

    #     :return values: (torch.Tensor) value function predictions.
    #     :return actions: (torch.Tensor) actions to take.
    #     :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
    #     :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
    #     :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
    #     """
    #     actions, action_log_probs, rnn_states_actor = self.actor(obs,
    #                                                              rnn_states_actor,
    #                                                              masks,
    #                                                              available_actions,
    #                                                              deterministic
    #                                                              )

    #     values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
    #     return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        [...]
        """
        # Convert numpy arrays to PyTorch tensors and move them to the correct device
        cent_obs = torch.tensor(cent_obs).to(self.device)
        obs = torch.tensor(obs).to(self.device)
        rnn_states_actor = torch.tensor(rnn_states_actor).to(self.device)
        rnn_states_critic = torch.tensor(rnn_states_critic).to(self.device)
        masks = torch.tensor(masks).to(self.device)

        if available_actions is not None:
            available_actions = torch.tensor(available_actions).to(self.device)

        # Now call the actor and critic with tensors on the correct device
        actions, action_log_probs, rnn_states_actor = self.actor(obs,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 available_actions,
                                                                 deterministic
                                                                 )

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks)

        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor
