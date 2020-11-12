import gym

#load gym environment
env = gym.make("LunarLander-v2")

#load a specific terrain (values range from 0 to about 6 -- specifically 400/60 because that were the hard-coded values)
env.load_terrain([2]*12)

#reset the environment (will load the last terrain if there is one)
observation = env.reset()

while True:
    env.render()
    action = 2   
    observation, reward, done, info = env.step(action) 

    if done: 
      break
            
