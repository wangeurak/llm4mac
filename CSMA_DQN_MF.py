import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import threading
import os
import interact_handle as IH

# Hyper Parameters
BATCH_SIZE = 8
LR = 0.0001                  # learning rate
EPSILON = 0.5                # greedy policy
EPSILON_DECAY = 0.9
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 10    # target update frequency
MEMORY_CAPACITY = 100

NODE_NUM = 10
N_STATES = 7 + NODE_NUM * 4
N_ACTIONS = 16

total_path = "D:\\CSMA_test\\"
device = torch.device("gpu")
method_str = "mf-dqn"
state_str = "-stable-hops"
suffix = method_str + state_str + str(NODE_NUM)
train_again = 0


# DQN net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES + 1, 64)
        self.out = nn.Linear(64, N_ACTIONS)

        # initialization
        nn.init.constant_(self.fc1.weight, 1)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.out.weight, 1)
        nn.init.constant_(self.out.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        actions_value = self.out(x)
        return actions_value


# DQN object
class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2 + 1))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.loss_list = []

    def choose_action(self, state, mean_action, train_flag, epslion):
        state.append(mean_action)
        state_temp = torch.unsqueeze(torch.FloatTensor(state), 0)
        state.pop()

        if np.random.uniform() < epslion and train_flag and not train_again:
            action = np.random.randint(0, N_ACTIONS)
            action = action
        else:
            actions_value = self.eval_net.forward(state_temp)
            action = torch.max(actions_value, 1)[1].data.numpy()
            # 从numpy数组到标量值
            action = action[0]
        return action

    def store_transition(self, s, a, r, a_mean, s_):
        transition = np.hstack((s, [a, r], a_mean, s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_a_mean = torch.FloatTensor(b_memory[:, N_STATES + 2:N_STATES + 3])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        b_S = torch.cat((b_s, b_a_mean), 1)
        b_S_ = torch.cat((b_s_, b_a_mean), 1)

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_S).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_S_).detach()  # detach from graph, don't backpropagate
        # 维度从(batch_size,N_action) - > (batch_size,1)
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)

        loss = self.loss_func(q_eval, q_target)

        self.loss_list.append(float(loss))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# multi-threads
class Node(threading.Thread):
    def __init__(self, nodeID ,epsilon):
        threading.Thread.__init__(self)
        self.nodeID = nodeID
        self.epsilon = epsilon

    def run(self):
        print("node", self.nodeID, "start to collect data...")
        self.dqn_train()

    def dqn_train(self):
        dqn = DQN()
        if train_again:
            dqn.eval_net = torch.load("./"+suffix+"_model/" + str(self.nodeID) + "-" + suffix + ".pth")
            if self.nodeID == 0:
                print("train again")

        state_path = total_path + "state" + str(self.nodeID) + ".txt"
        action_path = total_path + "action" + str(self.nodeID) + ".txt"
        reward_path = total_path + "reward" + str(self.nodeID) + ".txt"
        time_path = total_path + "time" + str(self.nodeID) + ".txt"
        other_path = total_path + "other" + str(self.nodeID) + ".txt"
        time_r = 0
        done = 0
        print_flag = 0
        s = []
        mean_action = 0

        for i_episode in range(20000):
            time = 0
            if done == 0:
                while 1:
                    if IH.check_file(state_path) and IH.check_file(other_path):
                        s = IH.read_file(state_path, 0)
                        mean_action = IH.read_file(other_path, 1)
                        break

            while 1:
                a = dqn.choose_action(s, mean_action, 1, self.epsilon)
                IH.write_file(action_path, a)

                done = 0

                while 1:
                    if IH.check_file(state_path) and IH.check_file(reward_path) and IH.check_file(time_path) \
                            and IH.check_file(other_path):
                        s_ = IH.read_file(state_path, 0)
                        r = IH.read_file(reward_path, 2)
                        a_mean = IH.read_file(other_path, 1)
                        time = time + 1

                        if time == 20:
                            done = 1
                            time = 0
                            print_flag = 1
                            if self.epsilon > 0.01:
                                self.epsilon = self.epsilon * EPSILON_DECAY
                        break

                dqn.store_transition(s, a, r, a_mean, s_)
                time_r += r

                # if dqn.memory_counter == MEMORY_CAPACITY:
                #     self.epsilon = 0.1

                if dqn.memory_counter > MEMORY_CAPACITY:
                    dqn.learn()

                if print_flag:
                    if self.nodeID == 0:
                       print("=====episode: ", i_episode, "=======")
                    time_r = 0
                    print_flag = 0

                s = s_

                if done:
                    if i_episode > 15:
                        torch.save(dqn.eval_net, "./"+suffix+"_model/" + str(self.nodeID) + "-" + suffix + ".pth")
                        print("node" + str(self.nodeID) + "模型已保存")
                    break


if __name__ == '__main__':

    if not os.path.exists('./' + suffix +'_model'):
        os.mkdir('./' + suffix +'_model')

    threadLock = threading.Lock()
    NodeSet = []
    print('\nCollecting experience...')
    for i in range(0, NODE_NUM):
        node = Node(i, EPSILON)
        node.start()
        NodeSet.append(node)

    for t in NodeSet:
        t.join()
    print('训练结束')