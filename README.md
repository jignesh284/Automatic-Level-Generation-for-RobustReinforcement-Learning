# Generating Game Levels
In this blog, we explore using Generative Adversarial Networks (GANs) to create new synthetic game levels which can be used to train an agent to play the game through Reinforcement Learning (RL). Prior work in generating game levels using GANs (see Fig. 1) has focused on generating levels that look like real game levels and are playable. However we focus on testing if these game levels can also be used to train an RL agent and possibly help the RL agent in learning a better generalized policy for playing the game.

**Fig. 1.** GAN generated levels:

<div style="text-align: center;">
    <img src="figures/GAN%20level1.png" width=300> <img src="figures/GAN%20level1.png" width=300>
</div>

## Real Lunar Lander v2 Level
In our implementation we use the LunarLander-v2 gym environment provided by [openai](https://gym.openai.com/envs/LunarLander-v2/). The game randomly generates a level each time it is loaded. Each level is characterized by **11 points** as shown below in Fig. 2. The **terrain points** are evenly distributed horizontally along the width of the level. The height of each point (except for points in the landing zone) is uniformly sampled from **\[0, 6.66)**. The landing zone is defined by 3 points in the middle each at a height of **3.33**. 

**Fig. 2.** Original Lunar Lander level:

<div style="text-align: center;">
    <img src="figures/terrain.png" width=300>
</div>

## Modified Lunar Lander v2 Level
The original game is fairly simple for an agent to learn a decent policy as the landing zone is always in the center and the ship always starts at the top rigt above the landing zone. Even though the ship is given an initial displacement to the left or right, the terrain peaks are not high enough to be obstacles for the ship. Moreover as the height of the terrain points are randomly sampled, there is not much of a distribution for a GAN to learn. Therefore we modify the terrain points such that the heights of the points are sampled from a Gaussian distribution with **mean = 9** and **standard deviation = 2**.

This ensures that the "real" lunar lander levels have a specific underlying distribution that the GAN would be expected to learn and it makes the levels more difficult for an RL agent to learn a decent policy for landing in the landing zone.

## Unrolled GANs
**Fig. 3.** GAN architecture:

<div style="text-align: center;">
    <img src="figures/generator.png" width=315> <img src="figures/discriminator.png" width=300>
</div>

# Training the RL Agent
We evaluate the generalizability of the agent by comparing across 3 conditions:  
* Using only 10 original levels 
* Using 10 original levels + 40 levels that are randomly sampled from a uniform distribution
* 10 original levels + 40 levels sampled from the GAN 
  
We reason that in more realistic scenarios, it may be difficult to create levels by hand, so we treat original levels as a scarse resource. Sampling from a GAN that approximates the true distribution of levels, however, is inexpensive, so we can have many more levels from a GAN. To make sure that it is not simply the inclusion of more diverse levels that increases the generalizability of the RL agent, we have our third condition which samples additional levels from a distribution that is not the same as the original distribution. This is akin to just randomly filling a game level with random assets that do not necessarily fulfil the game designers' stylistic criteria (in real-life these criteria are listed in internal style guides and are important for establishing the brand of a game).

To select the RL agent, we used different techniques for randomly-generated levels to see which algorithm is best at learning the task of playing Lunar Lander. We compared across Vanilla Policy Gradients, Advantage Actor-Critic, and Proximal Policy Optimization. Training curves for each method are shown below:

<div style="text-align: center;">
    <img src="figures/vpg.png" width=200> <img src="figures/a2c.png" width=200> <img src="figures/ppo.png" width=200>
</div>

We found that under several different initializations, A2C had the best maximum performance, and did not run into as many catastrophic failures over time.

We vary the number of training steps per level and how many levels were loaded, such that there were always 600,000 training steps (from previous experiments, peak values most often occurred before 300,000 timesteps and were not surpassed afterward). The mean cumulative reward for each condition and variations of training steps per level are shown below.

|  | 60 level loads, 10,000 timesteps per level | 100 level loads, 6,000 timesteps per level | 300 level loads, 2,000 timesteps per level | 600 level loads, 1,000 timesteps per level | 1000 level loads, 600 timesteps per level |
|-|-|-|-|-|-|
| 10 Original Levels (Original) | -63.9585831613715  +-  23.6296709499651 | -49.0820748383898  +-  25.2000656334681 | -29.2757488596109  +-  22.8957172909531 | -96.8522412955124  +-  13.0334122886884 | -51.6172432632894  +-  23.8856737297661 |
| 10 Original + 40 GAN-generated (Combined) | -17.6600965450973  +-  12.3021398180898 | 9.27202873913585  +-  23.1057813492394 | 0.0836289060744647  +-  38.1860668637518 | 38.6076335975186  +-  58.0603246977128 | -30.2135256269367  +-  13.7702798852628 |
| 10 Original + 40 "Random" Levels (Random) | -18.911273287633  +-  10.1782321600483 | -46.9979439293063  +-  41.8384183308556 | -24.02170300328  +-  22.3397621547855 | -46.0751791131674  +-  18.2597466440784 | -29.8595076694415  +-  29.2875552623666 |




# Code and Resources
- [Drive](https://drive.google.com/drive/folders/1ZU8QwG1WK8pGDoHGHVkUil-lqrs0Fz6P?usp=sharing)
- [Game_Repo](https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py)
