# Generating Game Levels
In this blog, we explore using Generative Adversarial Networks (GANs) to create new synthetic game levels which can be used to train an agent to play the game through Reinforcement Learning (RL). Prior work in generating game levels using GANs (see Fig. 1) has focused on generating levels that look like real game levels and are playable. However we focus on testing if these game levels can also be used to train an RL agent and possibly help the RL agent in learning a better generalized policy for playing the game.

**Fig. 1.** GAN generated levels:

<img src="figures/GAN%20level1.png" width=300> <img src="figures/GAN%20level1.png" width=300>

## Real Lunar Lander v2 Level
In our implementation we use the LunarLander-v2 gym environment provided by [openai](https://gym.openai.com/envs/LunarLander-v2/). The game randomly generates a level each time it is loaded. Each level is characterized by **11 points** as shown below in Fig. 2. The **terrain points** are evenly distributed horizontally along the width of the level. The height of each point (except for points in the landing zone) is uniformly sampled from **\[0, 6.66)**. The landing zone is defined by 3 points in the middle each at a height of **3.33**. 

**Fig. 2.** Original Lunar Lander level:

<img src="figures/terrain.png" width=300>

## Modified Lunar Lander v2 Level
The original game is fairly simple for an agent to learn a decent policy as the landing zone is always in the center and the ship always starts at the top rigt above the landing zone. Even though the ship is given an initial displacement to the left or right, the terrain peaks are not high enough to be obstacles for the ship. Moreover as the height of the terrain points are randomly sampled, there is not much of a distribution for a GAN to learn. Therefore we modify the terrain points such that the heights of the points are sampled from a Gaussian distribution with **mean = 9** and **standard deviation = 2**.

This ensures that the "real" lunar lander levels have a specific underlying distribution that the GAN would be expected to learn and it makes the levels more difficult for an RL agent to learn a decent policy for landing in the landing zone.

## Unrolled GANs
**Fig. 3.** GAN architecture:

<img src="figures/generator.png" width=315> <img src="figures/discriminator.png" width=300>

# Training the RL Agent

# Code and Resources
- [Drive](https://drive.google.com/drive/folders/1ZU8QwG1WK8pGDoHGHVkUil-lqrs0Fz6P?usp=sharing)
- [Game_Repo](https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py)
