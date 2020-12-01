import gym
import os
from stable_baselines3.common.monitor import Monitor as M
from stable_baselines3 import A2C
from random import randint
from csv import reader

model = A2C.load("./best_models/random_60_10000")

env = gym.make('LunarLander-v2')


# read csv file as a list of lists
with open('./moderate_dataset/urgan_test_samples.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Pass reader object to list() to get a list of lists
    list_of_rows = list(csv_reader)

test_samples = [[float(j) for j in i] for i in list_of_rows]

TEST_LEVEL_NUMS = 20

cumulated_reward_ls = []
last_reward_ls = []

for i in range(TEST_LEVEL_NUMS):
  env.load_terrain([8.51, 8.77, 10.4, 7.39, 3.33, 3.33, 3.33, 10.6, 9.99, 9.13, 9.67])
  init_position = randint(1,18)
  env.set_initial_x(init_position)
  # Logs will be saved in log_dir/monitor.csv
  obs = env.reset()
  
  # plt.imshow(env.render(mode='rgb_array'))
  # plt.show()
  # print(env.terrain_y_values)
  cumulated_reward = 0.0
  last_reward = 0.0
  while True:
      
      action, _states = model.predict(obs, deterministic=True)
      obs, reward, done, info = env.step(action)
      cumulated_reward += reward
      last_reward = reward
      im = env.render('rgb_array') 
      
      if done: 
        break;

  print("Cumulated Reward", cumulated_reward, "Last Reward", last_reward)
  cumulated_reward_ls.append(cumulated_reward)
  last_reward_ls.append(last_reward)
#   plt.imshow(env.render(mode='rgb_array'))
#   plt.show()
print(" -------------- Final Average Results -----------------")
print("Mean Cumulated Reward", sum(cumulated_reward_ls)/len(cumulated_reward_ls), "Mean Last Reward", sum(last_reward_ls)/len(last_reward_ls))
              
env.close()