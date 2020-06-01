import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import gym
import pickle

# delete cart velocity state observation
# made a standard cartpole env as POMDP!!!!!!!!!!!!!!!!!!!
STATE_DIM = 4
ACTION_DIM = 2
STEP = 1000
SAMPLE_NUMS = 1000
OPTIM_BATCH = 1000
EPISODE_NUM = 600


# actor using a LSTM + fc network architecture to estimate hidden states.
class DiscreteLSTM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=(128, 128), activation='tanh'):
        super().__init__()
        self.is_disc_action = True
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.fc1 = nn.Linear(state_dim, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[1], hidden_size[1])

        self.action_head = nn.Linear(hidden_size[1], action_dim)
        self.value_head = nn.Linear(hidden_size[1], 1)

        nn.init.normal_(self.fc1.weight, mean=0., std=0.1)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.normal_(self.fc2.weight, mean=0., std=0.1)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.normal_(self.action_head.weight, mean=0., std=0.1)
        nn.init.constant_(self.action_head.bias, 0.0)
        nn.init.normal_(self.value_head.weight, mean=0., std=0.1)
        nn.init.constant_(self.value_head.bias, 0.0)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        log_prob = F.log_softmax(self.action_head(x), dim=1)
        value = self.value_head(x)
        return log_prob, value


def roll_out(network, task, sample_nums, init_state, dtype, device):
    states = []
    actions = []
    log_probs = []
    rewards = []
    masks = []
    is_done = False
    final_r = 0
    score = 0
    state = init_state

    for j in range(sample_nums):
        states.append(state)
        state = torch.tensor(state, dtype=dtype, device=device).unsqueeze(0)
        log_softmax_action, _ = network(state)

        softmax_action = torch.exp(log_softmax_action)
        action = np.random.choice(ACTION_DIM, p=softmax_action.cpu().detach().numpy()[0])
        next_state, reward, done, _ = task.step(action)

        one_hot_action = [int(k == action) for k in range(ACTION_DIM)]
        action = torch.tensor(action, dtype=torch.long, device=device).unsqueeze(0).unsqueeze(0)
        log_softmax_action = log_softmax_action.gather(1, action).squeeze(0).cpu().detach().numpy()
        mask = 0 if done else 1

        actions.append(one_hot_action)
        log_probs.append(log_softmax_action)
        rewards.append(reward)
        masks.append(mask)

        state = next_state
        if done:
            is_done = True
            state = task.reset()
            score = j+1
            break
    if not is_done:
        final_state = torch.tensor(state, dtype=dtype, device=device).unsqueeze(0)
        c_out, _ = network(final_state)
        final_r = c_out.cpu().data.numpy()
        score = sample_nums
    return states, actions, log_probs, rewards, masks, final_r, state, score


def discount_reward(r, gamma, final_r):
    discounted_r = np.zeros_like(r)
    running_add = final_r
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def to_device(device, *args):
    return [x.to(device) for x in args]


def estimate_advantages(rewards, masks, values, gamma, tau, device):
    rewards, masks, values = to_device(torch.device('cpu'), rewards, masks, values)
    tensor_type = type(rewards)
    deltas = tensor_type(rewards.size(0), 1)
    advantages = tensor_type(rewards.size(0), 1)

    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    returns = values + advantages
    advantages = (advantages - advantages.mean()) / advantages.std()

    advantages, returns = to_device(device, advantages, returns)
    return advantages, returns


def main():
    device = torch.device('cuda', index=0) if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32
    # init a task generator for data fetching
    task = gym.make("CartPole-v1")
    init_state = task.reset()
    network = DiscreteLSTM(4, 2)
    network.to(device)
    network_optim = torch.optim.Adam(network.parameters(), lr=4e-4)

    for step in range(STEP):
        # sample
        score_sum = 0
        s_buff = []
        a_buff = []
        r_buff = []
        m_buff = []
        lp_buff = []
        for samp_i in range(EPISODE_NUM):
            states, actions, log_probs, rewards, masks, final_r, current_state, score = \
                roll_out(network, task, SAMPLE_NUMS, init_state, dtype, device)
            init_state = current_state
            score_sum += score
            s_buff.extend(states)
            a_buff.extend(actions)
            r_buff.extend(rewards)
            m_buff.extend(masks)
            lp_buff.extend(log_probs)

        # preprocess
        score_avg = score_sum / EPISODE_NUM
        s_pad = torch.tensor(s_buff, dtype=dtype, device=device)
        a_pad = torch.tensor(a_buff, dtype=dtype, device=device)
        r_pad = torch.tensor(r_buff, dtype=dtype, device=device)
        m_pad = torch.tensor(m_buff, dtype=dtype, device=device)
        lp_pad = torch.tensor(lp_buff, dtype=dtype, device=device)
        with torch.no_grad():
            _, values = network(s_pad)
        ad_pad, r_pad = estimate_advantages(r_pad, m_pad, values, 0.99, 0.95, device)

        # train
        actor_loss_sum = 0
        value_loss_sum = 0
        aux_loss_sum = 0
        aux_sum = 0
        for epoch_i in range(5):
            # shuffle
            perm = np.arange(s_pad.shape[0])
            np.random.shuffle(perm)
            perm = torch.tensor(perm, dtype=torch.long, device=device)
            state = s_pad[perm].clone()
            action = a_pad[perm].clone()
            advantage = ad_pad[perm].clone()
            qvalue = r_pad[perm].clone()
            fixed_log_prob = lp_pad[perm].clone()

            aux_opt_sum = 0
            for opt_i in range(int(state.shape[0]/OPTIM_BATCH)):
                ind = slice(opt_i * OPTIM_BATCH, min((opt_i + 1) * OPTIM_BATCH, state.shape[0]))
                s_batch, a_batch, q_batch, ad_batch, flb_batch = state[ind], action[ind], qvalue[ind], advantage[ind], fixed_log_prob[ind]

                # train actor network
                network_optim.zero_grad()
                lb_batch, vs = network(s_batch)
                lb_batch = torch.sum(lb_batch * a_batch, 1).unsqueeze(1)
                ratio = torch.exp(lb_batch - flb_batch)
                surr1 = ratio * ad_batch
                surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * ad_batch
                actor_loss = -torch.min(surr1, surr2).mean()

                # train value network
                target_values = q_batch
                values = vs
                criterion = nn.MSELoss()
                value_loss = criterion(values, target_values)
                loss = value_loss + actor_loss
                loss.backward()
                network_optim.step()

        # actor_loss_avg = actor_loss_sum.data.numpy() / 25
        # value_loss_avg = value_loss_sum.data.numpy() / 25
        # aux_loss_avg = aux_loss_sum.data.numpy() / aux_sum
        # print('step:', step, '| actor_loss:', actor_loss_avg, '| critic_loss:', value_loss_avg, '| aux_loss:', aux_loss_avg, '| score:', score_avg)
        print('step:', step, '| score:', score_avg)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
