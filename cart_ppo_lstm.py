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
A_HIDDEN = 128
C_HIDDEN = 128
OPTIM_BATCH = 100
AUX_OPTIM_BATCH = 2
EPISODE_NUM = 600


# actor using a LSTM + fc network architecture to estimate hidden states.
class DiscreteLSTM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=(128, 256, 64), activation='tanh'):
        super().__init__()
        self.is_disc_action = True
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.fc1 = nn.Linear(state_dim, hidden_size[0])
        self.fc11 = nn.Linear(hidden_size[0], hidden_size[1])
        self.lstm = nn.LSTM(hidden_size[1], hidden_size[0], batch_first=True)
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[2])

        self.action_head = nn.Linear(hidden_size[2], action_dim)
        self.value_head = nn.Linear(hidden_size[2], 1)

        nn.init.normal_(self.fc1.weight, mean=0., std=0.1)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.normal_(self.fc2.weight, mean=0., std=0.1)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.normal_(self.action_head.weight, mean=0., std=0.1)
        nn.init.constant_(self.action_head.bias, 0.0)
        nn.init.normal_(self.value_head.weight, mean=0., std=0.1)
        nn.init.constant_(self.value_head.bias, 0.0)

    def forward(self, x, hidden):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc11(x))
        x, hidden = self.lstm(x, hidden)
        x = self.activation(self.fc2(x))
        log_prob = F.log_softmax(self.action_head(x), dim=2)
        value = self.value_head(x)
        return log_prob, value, hidden


def roll_out(network, task, sample_nums, init_state, dtype, device):
    states = []
    actions = []
    log_probs = []
    rewards = []
    is_done = False
    final_r = 0
    score = 0
    state = init_state
    a_hx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0).to(device)
    a_cx = torch.zeros(A_HIDDEN).unsqueeze(0).unsqueeze(0).to(device)
    c_hx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0).to(device)
    c_cx = torch.zeros(C_HIDDEN).unsqueeze(0).unsqueeze(0).to(device)

    for j in range(sample_nums):
        states.append(state)
        state = torch.tensor(state, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        log_softmax_action, _, (a_hx, a_cx) = network(state, (a_hx, a_cx))

        softmax_action = torch.exp(log_softmax_action)
        action = np.random.choice(ACTION_DIM, p=softmax_action.cpu().detach().numpy()[0][0])
        next_state, reward, done, _ = task.step(action)

        one_hot_action = [int(k == action) for k in range(ACTION_DIM)]
        action = torch.tensor(action, dtype=torch.long, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        log_softmax_action = log_softmax_action.gather(2, action).cpu().detach().numpy()

        actions.append(one_hot_action)
        log_probs.append(log_softmax_action)
        rewards.append(reward)

        state = next_state
        if done:
            is_done = True
            state = task.reset()
            score = j+1
            break
    if not is_done:
        final_state = torch.tensor(state, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        c_out, _ = network(final_state, (c_hx, c_cx))
        final_r = c_out.cpu().data.numpy()
        score = sample_nums
    return states, actions, log_probs, rewards, final_r, state, score


def discount_reward(r, gamma, final_r):
    discounted_r = np.zeros_like(r)
    running_add = final_r
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


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
        q_buff = []
        lp_buff = []
        for samp_i in range(EPISODE_NUM):
            states, actions, log_probs, rewards, final_r, current_state, score = \
                roll_out(network, task, SAMPLE_NUMS, init_state, dtype, device)
            init_state = current_state
            score_sum += score
            qvalues = discount_reward(rewards, 0.99, final_r)
            s_buff.append(states)
            a_buff.append(actions)
            q_buff.append(qvalues)
            lp_buff.append(log_probs)

        # preprocess
        score_avg = score_sum / EPISODE_NUM
        s_pad = []
        a_pad = []
        q_pad = []
        lp_pad = []
        for element in s_buff:
            s_pad.append(torch.tensor(element, dtype=dtype))
        s_pad = pad_sequence(s_pad, batch_first=True).to(device)
        for element in a_buff:
            a_pad.append(torch.tensor(element, dtype=dtype))
        a_pad = pad_sequence(a_pad, batch_first=True).to(device)
        for element in q_buff:
            q_pad.append(torch.tensor(element, dtype=dtype))
        q_pad = pad_sequence(q_pad, batch_first=True).to(device)
        for element in q_buff:
            lp_pad.append(torch.tensor(element, dtype=dtype))
        lp_pad = pad_sequence(lp_pad, batch_first=True).to(device)

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
            qvalue = q_pad[perm].clone()
            fixed_log_prob = lp_pad[perm].clone()

            aux_opt_sum = 0
            for opt_i in range(int(EPISODE_NUM/OPTIM_BATCH)):
                ind = slice(opt_i * OPTIM_BATCH, min((opt_i + 1) * OPTIM_BATCH, len(state)))
                batch = min((opt_i+1)*OPTIM_BATCH, len(state)) - opt_i*OPTIM_BATCH
                s_batch, a_batch, q_batch, flb_batch = state[ind], action[ind], qvalue[ind], fixed_log_prob[ind]

                # train actor network
                a_hx = torch.zeros(1, batch, A_HIDDEN).to(device)
                a_cx = torch.zeros(1, batch, A_HIDDEN).to(device)
                network_optim.zero_grad()
                lb_batch, vs, _ = network(s_batch, (a_hx, a_cx))
                vs.detach()
                qs = q_batch.unsqueeze(2)
                advantages = (qs - vs).squeeze()
                lb_batch = torch.sum(lb_batch * a_batch, 2)
                ratio = torch.exp(lb_batch - flb_batch)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # train value network
                target_values = qs
                c_hx = torch.zeros(1, batch, C_HIDDEN).to(device)
                c_cx = torch.zeros(1, batch, C_HIDDEN).to(device)
                _, values, _ = network(s_batch, (c_hx, c_cx))
                value_loss = F.smooth_l1_loss(values, target_values)
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
