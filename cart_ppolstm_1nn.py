import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Categorical
import gym
from examples.logger import Logger

logger = Logger('./logs')

EPOCH = 2
TIME_NUMS = 20
HIDDEN_SIZE = 128
OPTIM_BATCH = 1
SEQ_NUM = 1
gamma = 0.98
lmbda = 0.95
eps_clip = 0.1
print_interval = 20


class PPOLSTM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=(64, HIDDEN_SIZE)):
        super().__init__()
        self.data = []

        self.fc1 = nn.Linear(state_dim, hidden_size[0])
        self.lstm = nn.LSTM(hidden_size[0], hidden_size[1], batch_first=True)
        self.action_head = nn.Linear(hidden_size[1], action_dim)
        self.value_head = nn.Linear(hidden_size[1], 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)

        self.init()

    def init(self):
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.kaiming_normal_(self.action_head.weight, nonlinearity='relu')
        nn.init.constant_(self.action_head.bias, 0.0)
        nn.init.kaiming_normal_(self.value_head.weight, nonlinearity='relu')
        nn.init.constant_(self.value_head.bias, 0.0)
        nn.init.orthogonal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.constant_(self.lstm.bias_ih_l0, 1.0)
        nn.init.constant_(self.lstm.bias_hh_l0, 1.0)

    # def pi(self, x, hidden):
    #     x = F.relu(self.fc1(x))
    #     x, hidden = self.lstm(x, hidden)
    #     x = self.action_head(x)
    #     prob = F.softmax(x, dim=2)
    #     return prob, hidden
    #
    # def v(self, x, hidden):
    #     x = F.relu(self.fc1(x))
    #     x, _ = self.lstm(x, hidden)
    #     v = self.value_head(x)
    #     return v
    def forward(self, x, hidden):
        x = F.relu(self.fc1(x))
        x, hidden = self.lstm(x, hidden)
        pi_x = self.action_head(x)
        prob = F.softmax(pi_x, dim=2)
        v = self.value_head(x)
        return prob, v, hidden

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_l, a_l, r_l, ns_l, fp_l, dm_l, pm_l, hin_l, hout_l = [], [], [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, ns, fp, dm, pm, hin, hout = transition
            s_l.append(s)
            a_l.append([a])
            r_l.append([r])
            ns_l.append(ns)
            fp_l.append([fp])
            dm_l.append([dm])
            pm_l.append([pm])
            hin_l.append(hin)
            hout_l.append(hout)
        self.data = []
        return s_l, a_l, r_l, ns_l, fp_l, dm_l, pm_l, hin_l[0], hout_l[0]


def roll_out(model, env, first_state, hidden_in, time_nums, dtype, device):
    model.to('cpu')
    score = 0
    s = first_state
    h_in = hidden_in
    done_flag = False
    for t in range(time_nums):
        state = torch.from_numpy(s).to(dtype).unsqueeze(0).unsqueeze(0)
        # prob, h_out = model.pi(state, h_in)
        prob, _, h_out = model(state, h_in)
        prob = prob.view(-1)
        m = Categorical(prob)
        a = m.sample().item()
        ns, r, done, _ = env.step(a)
        pad_mask = 1
        done_mask = 0 if done else 1

        model.put_data((s, a, r, ns, prob[a].item(), done_mask, pad_mask, h_in, h_out))

        s = ns
        h_in = h_out
        score += r
        done_flag = done
        if done:
            s = env.reset()
            hx = torch.zeros(1, 1, HIDDEN_SIZE)
            cx = torch.zeros(1, 1, HIDDEN_SIZE)
            h_in = (hx, cx)
            break
    model.to(device)
    return s, h_in, done_flag, score


def advantage_cal(model, states, next_states, dones, rewards, first_hidden, second_hidden, device, dtype):
    states = torch.tensor(states, dtype=dtype, device=device).unsqueeze(0)
    next_states = torch.tensor(next_states, dtype=dtype, device=device).unsqueeze(0)
    dones = torch.tensor(dones, dtype=dtype, device=device)
    rewards = torch.tensor(rewards, dtype=dtype, device=device)

    # v_next = model.v(next_states, second_hidden).squeeze(0)
    _, v_next, _ = model(next_states, second_hidden)
    v_next = v_next.squeeze(0)
    td_target = rewards + gamma * v_next * dones
    # v = model.v(states, first_hidden).squeeze(0)
    _, v, _ = model(states, first_hidden)
    v = v.squeeze(0)
    delta = (td_target - v).cpu().detach().numpy()

    advantage_list = []
    advantage = 0
    for item in delta[::-1]:
        advantage = gamma * lmbda * advantage + item[0]
        advantage_list.append([advantage])
    advantage_list.reverse()
    td_target = td_target.cpu().detach().numpy().tolist()
    return advantage_list, td_target


def main():
    device = torch.device('cuda', index=0) if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32
    env = gym.make("CartPole-v1")
    model = PPOLSTM(4, 2)
    model.to(device)
    epi_i = 0
    score_sum = 0
    s_first = env.reset()
    hx = torch.zeros(1, 1, HIDDEN_SIZE, dtype=dtype)
    cx = torch.zeros(1, 1, HIDDEN_SIZE, dtype=dtype)
    hin_first = (hx, cx)
    policy_loss_sum = 0
    value_loss_sum = 0
    tb_log_i = 1
    tb_sum_i = 0
    while epi_i < 10000:
        s_first, hin_first, done_flag, score = roll_out(model, env, s_first, hin_first, TIME_NUMS, dtype, device)
        if done_flag:
            epi_i += 1
        score_sum += score
        s_l, a_l, r_l, ns_l, fp_l, dm_l, pm_l, (h1, c1), (h2, c2) = model.make_batch()
        first_hidden = (h1.to(device).detach(), c1.to(device).detach())
        second_hidden = (h2.to(device).detach(), c2.to(device).detach())
        ad_l, q_l = advantage_cal(model, s_l, ns_l, dm_l, r_l, first_hidden, second_hidden, device, dtype)

        # prepossess
        s_b = torch.tensor(s_l, dtype=dtype, device=device).unsqueeze(0)
        a_b = torch.tensor(a_l, dtype=torch.long, device=device)
        fp_b = torch.tensor(fp_l, dtype=dtype, device=device)
        pm_b = torch.tensor(pm_l, dtype=dtype, device=device)
        ad_b = torch.tensor(ad_l, dtype=dtype, device=device)
        q_b = torch.tensor(q_l, dtype=dtype, device=device)

        # train
        for _ in range(EPOCH):
            # p, _ = model.pi(s_b, first_hidden)
            p, _, _ = model(s_b, first_hidden)
            p_b = p.squeeze(0).gather(1, a_b)
            ratio = torch.exp(torch.log(p_b) - torch.log(fp_b))
            surr1 = ratio * ad_b
            surr2 = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * ad_b
            policy_loss = -torch.min(surr1, surr2).mean()
            policy_loss_sum += policy_loss

            # v = model.v(s_b, first_hidden)
            _, v, _ = model(s_b, first_hidden)
            v_b = v.squeeze(0)
            value_loss = F.smooth_l1_loss(v_b, q_b)
            value_loss_sum += value_loss

            loss = policy_loss + value_loss
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            tb_sum_i += 1

        # tensorboard and log
        if epi_i % print_interval == 0 and epi_i != 0 and done_flag:
            policy_loss_avg = policy_loss_sum / tb_sum_i
            value_loss_avg = value_loss_sum / tb_sum_i
            score_avg = score_sum / print_interval

            print('episode: {} avg score: {:.1f}'.format(epi_i, score_avg))

            info = {
                'policy_loss': policy_loss_avg,
                'value_loss': value_loss_avg,
                'score': score_avg
            }
            for tag, value in info.items():
                logger.scalar_summary(tag, value, tb_log_i)
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.cpu().detach().numpy(), tb_log_i)
                logger.histo_summary(tag + '/grad', value.grad.cpu().detach().numpy(), tb_log_i)
            policy_loss_sum = 0
            value_loss_sum = 0
            score_sum = 0
            tb_sum_i = 0
            tb_log_i += 1

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
