import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import threading
import os
import asyncio
from collections import deque
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import interact_handle as IH

# 超参数
BATCH_SIZE = 8
LR_ACTOR = 5e-5  # 演员（模型+线性头）学习率
LR_CRITIC = 1e-4  # 评论家学习率
GAMMA = 0.99      # 奖励折扣因子
EPS_CLIP = 0.2    # PPO 剪切参数
MEMORY_CAPACITY = 100
NODE_NUM = 10
N_STATES = 7 + NODE_NUM * 4
N_ACTIONS = 16
ENTROPY_BETA = 0.01  # 熵正则化系数
KL_PENALTY = 0.01    # KL 散度惩罚系数

total_path = "E:\\CSMA_test\\"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
method_str = "llm4mac-qwen-ppo"
state_str = "-stable-hops"
suffix = method_str + state_str + str(NODE_NUM)
train_again = 0

# 带策略头的 Qwen 模型
class QwenWithPolicyHead(nn.Module):
    def __init__(self, model_path, num_actions=16):
        super(QwenWithPolicyHead, self).__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.hidden_size = self.base_model.config.hidden_size
        self.policy_head = nn.Linear(self.hidden_size, num_actions)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        last_hidden = hidden_states[:, -1, :]
        action_logits = self.policy_head(last_hidden)
        action_probs = self.softmax(action_logits)
        return action_probs

# 评论家网络
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(N_STATES + 1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.out.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.out(x)
        return value

# PPO 代理
class PPOAgent:
    def __init__(self):
        self.model = QwenWithPolicyHead(model_path="D:\\Qwen\\Qwen2.5-0.5B-Instruct").to(device)
        self.tokenizer = self.model.tokenizer
        self.critic = Critic().to(device)
        lora_config = LoraConfig(
            r=4,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "policy_head"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.actor_optimizer = torch.optim.Adam(self.model.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.memory = deque(maxlen=MEMORY_CAPACITY)
        self.old_log_probs = []
        self.learn_step_counter = 0

    def sie_prompt(self, state, mean_action, node_id):
        role = f"<Role>Node_{node_id}</Role>"
        target = "<Target>Optimize_MAC_Throughput</Target>"
        action_candidates = "<Action_Candidates>" + ",".join([str(i) for i in range(N_ACTIONS)]) + "</Action_Candidates>"
        state_str = "<State>" + ",".join([str(s) for s in state]) + "</State>"
        mean_action_str = f"<Mean_Action>{mean_action}</Mean_Action>"
        return f"{role}{target}{action_candidates}{state_str}{mean_action_str}"

    def choose_action(self, state, mean_action, node_id, train_flag):
        prompt = self.sie_prompt(state, mean_action, node_id)
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        action_probs = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        dist = torch.distributions.Categorical(probs=action_probs)
        if train_flag:
            action = dist.sample()
            log_prob = dist.log_prob(action)
            self.old_log_probs.append(log_prob)
        else:
            action = torch.argmax(action_probs)
            log_prob = None
        return action.item(), log_prob, action_probs

    def store_transition(self, state, action, reward, mean_action, state_, log_prob, action_probs):
        self.memory.append((state, action, reward, mean_action, state_, log_prob, action_probs))

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = np.random.choice(len(self.memory), BATCH_SIZE, replace=False)
        batch_data = [self.memory[i] for i in batch]
        b_s = torch.FloatTensor([d[0] for d in batch_data]).to(device)
        b_a = torch.LongTensor([d[1] for d in batch_data]).to(device)
        b_r = torch.FloatTensor([d[2] for d in batch_data]).to(device)
        b_a_mean = torch.FloatTensor([d[3] for d in batch_data]).to(device)
        b_s_ = torch.FloatTensor([d[4] for d in batch_data]).to(device)
        b_old_log_probs = torch.stack([d[5] for d in batch_data]).to(device)
        b_action_probs = torch.stack([d[6] for d in batch_data]).to(device)

        b_s_with_mean = torch.cat((b_s, b_a_mean.unsqueeze(1)), dim=1)
        b_s_with_mean_ = torch.cat((b_s_, b_a_mean.unsqueeze(1)), dim=1)

        values = self.critic(b_s_with_mean).squeeze()
        next_values = self.critic(b_s_with_mean_).squeeze().detach()
        advantages = b_r + GAMMA * next_values - values
        returns = b_r + GAMMA * next_values

        critic_loss = F.mse_loss(values, returns)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_log_probs = []
        for i in range(BATCH_SIZE):
            prompt = self.sie_prompt(b_s[i].tolist(), b_a_mean[i].item(), node_id=0)
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
            new_probs = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            dist = torch.distributions.Categorical(probs=new_probs)
            new_log_probs.append(dist.log_prob(b_a[i]))
        new_log_probs = torch.stack(new_log_probs)

        ratio = torch.exp(new_log_probs - b_old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        entropy = -(b_action_probs * torch.log(b_action_probs + 1e-10)).sum(dim=1).mean()
        kl_div = torch.distributions.kl_divergence(
            torch.distributions.Categorical(probs=b_action_probs),
            torch.distributions.Categorical(probs=b_action_probs)
        ).mean()
        total_loss = policy_loss - ENTROPY_BETA * entropy + KL_PENALTY * kl_div

        self.actor_optimizer.zero_grad()
        total_loss.backward()
        self.actor_optimizer.step()

        self.memory.clear()
        self.old_log_probs = []
        self.learn_step_counter += 1

# 节点类（保持与之前一致）
class Node(threading.Thread):
    def __init__(self, node_id):
        threading.Thread.__init__(self)
        self.node_id = node_id

    def run(self):
        print(f"节点 {self.node_id} 开始收集数据...")
        asyncio.run(self.ppo_train())

    async def ppo_train(self):
        agent = PPOAgent()
        if train_again:
            agent.critic.load_state_dict(torch.load(f"./{suffix}_model/{self.node_id}-{suffix}_critic.pth"))
            if self.node_id == 0:
                print("再次训练")

        state_path = f"{total_path}state{self.node_id}.txt"
        action_path = f"{total_path}action{self.node_id}.txt"
        reward_path = f"{total_path}reward{self.node_id}.txt"
        time_path = f"{total_path}time{self.node_id}.txt"
        other_path = f"{total_path}other{self.node_id}.txt"

        time_r = 0
        done = 0
        print_flag = 0
        s = []
        mean_action = 0

        for i_episode in range(20000):
            time = 0
            if not done:
                while True:
                    if IH.check_file(state_path) and IH.check_file(other_path):
                        s = IH.read_file(state_path, 0)
                        mean_action = IH.read_file(other_path, 1)
                        break

            while True:
                a, log_prob, action_probs = agent.choose_action(s, mean_action, self.node_id, train_flag=True)
                IH.write_file(action_path, a)
                done = 0

                while True:
                    if (IH.check_file(state_path) and IH.check_file(reward_path) and 
                        IH.check_file(time_path) and IH.check_file(other_path)):
                        s_ = IH.read_file(state_path, 0)
                        r = IH.read_file(reward_path, 2)
                        a_mean = IH.read_file(other_path, 1)
                        time += 1

                        if time == 20:
                            done = 1
                            time = 0
                            print_flag = 1
                        break

                agent.store_transition(s, a, r, a_mean, s_, log_prob, action_probs)
                time_r += r

                if len(agent.memory) >= MEMORY_CAPACITY:
                    agent.learn()

                if print_flag:
                    if self.node_id == 0:
                        print(f"=====回合: {i_episode} =======")
                    time_r = 0
                    print_flag = 0

                s = s_

                if done:
                    if i_episode > 15:
                        torch.save(agent.critic.state_dict(), 
                                 f"./{suffix}_model/{self.node_id}-{suffix}_critic.pth")
                        agent.model.save_pretrained(f"./{suffix}_model/{self.node_id}-{suffix}_actor")
                        agent.tokenizer.save_pretrained(f"./{suffix}_model/{self.node_id}-{suffix}_actor")
                        print(f"节点 {self.node_id} 模型已保存")
                    break

                await asyncio.sleep(0.01)

if __name__ == '__main__':
    if not os.path.exists(f'./{suffix}_model'):
        os.mkdir(f'./{suffix}_model')

    thread_lock = threading.Lock()
    node_set = []
    print('\n收集经验...')
    for i in range(NODE_NUM):
        node = Node(i)
        node.start()
        node_set.append(node)

    for t in node_set:
        t.join()
    print('训练完成')