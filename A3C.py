import torch
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import vista
import matplotlib.pyplot as plt

def tensor(shape, flag=False):
    if flag:
        val = torch.zeros(shape)
    else:
        st_dev = 1.0 / np.sqrt(shape[0])
        val = torch.randn(shape) * st_dev
    return val.detach().requires_grad_(True)

def get_predictions(state):
    linear = torch.matmul(state, model['w1']) + model['b1']
    h = torch.tanh(linear)
    mult_linear = torch.matmul(h, model['w_actor']) + model['b_actor']
    mu = torch.tanh(mult_linear)
    std = torch.exp(model['log_std'])
    value = torch.matmul(h, model['w_critic']) + model['b_critic']
    return Normal(mu, std), value

first_dim = 4
second_dim = 5
third_dim = 2
gamma = 0.99
max_iter = 20
limit = 200 
reward_history = []

model = {'w1': tensor((first_dim, second_dim)),'b1': tensor((second_dim,), flag=True),'w_actor': tensor((second_dim, third_dim)),'b_actor': tensor((third_dim,), flag=True),'w_critic': tensor((second_dim, 1)),'b_critic': tensor((1,), flag=True),'log_std': torch.zeros(third_dim, requires_grad=True)}

# Make sure the car turns exactly at the rate the steering wheel is moving
with torch.no_grad():
    model['w_actor'].mul_(0.01)

optimizer = optim.Adam(model.values(), lr=1e-4)

#change directory to current file
path = "/Users/riddhi/Desktop/aiea/vista_traces/20210726-131322_lexus_devens_center"
world = vista.World([path])
car = world.spawn_agent({})
world.reset()

iters = 0
while iters < limit:
    optimizer.zero_grad()
    t_start = iters
    states = []
    actions = []
    rewards = []
    probability = []
    values = []
    
    #iterate until condition is reached
    while not car.done and (iters - t_start < max_iter):
        state_list = [car.relative_state.x, car.relative_state.yaw, car.speed, car.ego_dynamics.steering]
        s_t = torch.tensor(np.array(state_list), dtype=torch.float32)
        
        dist, val = get_predictions(s_t)
        action = dist.sample()
        
        car.step_dynamics(action.detach().numpy().flatten())
        
        # Use the reward formula: 1- |x|
        reward = 1.0 - abs(car.relative_state.x) - abs(car.relative_state.yaw)
        
        #if car crashes, add a penalty
        if car.done:
            reward = -2.0 
        
        rewards.append(reward)
        probability.append(dist.log_prob(action).sum(-1))
        values.append(val)
        iters += 1

    #if car is done running, reset the world
    if car.done:
        let_go = torch.zeros(1)
        world.reset()
    #otherwise, keep moving the car and continue to get the predictions
    else:
        final_vals = [car.relative_state.x, car.relative_state.yaw, car.speed, car.ego_dynamics.steering]
        final = torch.tensor(np.array(final_vals), dtype=torch.float32)
        i, y = get_predictions(final)
        let_go = y.detach()

    #apply the A3C formula
    policy_loss = 0
    value_loss = 0
    for idx in reversed(range(len(rewards))):
        let_go = rewards[idx] + gamma * let_go
        advantage = let_go - values[idx]
        policy_loss -= probability[idx] * advantage.detach()
        value_loss += 0.5 * advantage.pow(2)
    #calculate the average reward
    total_loss = (policy_loss + value_loss).mean()
    total_loss.backward()

    #adds stability to prevent large error gradients
    torch.nn.utils.clip_grad_norm_(model.values(), max_norm=0.5)
    optimizer.step()

    reward_history.append(float(np.mean(rewards)))
    #print out the data
    if len(reward_history) % 10 == 0:
        print(f"Total Steps {iters} | Average Reward: {reward_history}")

# Plot the results
plt.figure()
plt.plot(reward_history)
plt.title("A3C model")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.grid(True)
plt.show()
