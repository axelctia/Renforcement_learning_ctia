# Skiing game 

## Project goal

This project is written to code an algorithm of Reinforcement Learning in python code that solve Skiing, an Atari game of gymnasium API. We have used SARSA algorithm to this problem.

## SARSA Algorithm

SARSA (**S**tate-**A**ction-**R**eward-**S**tate-Action) is a type of reinforcement learning algorithm that works by updating the **Q-values** of state-action pairs based on the reward obtained from the **current action** and the expected rewards of the **next action**. The SARSA algorithm is an on-policy learning method that uses an epsilon-greedy policy to select actions

## Skiing problem

(version : ALE/Skiing-v5) 

The Skiing problem is an environment in OpenAI's Gym that involves a skiing game where the player controls a skier who must navigate downhill through a series of gates while avoiding obstacles such as trees and rocks. The goal is to run through all gates (between the poles) in the fastest time.You are penalized five seconds for each gate you miss. If you hit a gate or a tree, your skier will jump back up and keep going. The problem is particularly challenging due to the continuous and high-dimensional state space and the sparse reward signal.

### Environnement

To create Skiing-v5 environment, you have to import gymnasium and use his`make` fonction (env= gymnasium.make( “ALE/Skiing-v5”) )

### Observation space

In the Skiing-v5 environment in the Gymnasium API, the observation space is a **3D NumPy array** with **dimensions (210, 160, 3)**. This means that:

- The observation consists of a series of grayscale images with a resolution of 210x160 pixels.
- Each pixel in the image is represented by three values corresponding to the red, green, and blue color channels.

### Action space

The action space is discrete and consists of three possible actions that the agent can take at each timestep:

- **Move left**: The agent moves the skier to the left.
- **Move right**: The agent moves the skier to the right.
- **Do nothing**: The agent does not take any action.

These actions are represented by integers in the range [0, 2], where **0** corresponds to **doing nothing** , **1** corresponds to **moving right** , and **2** corresponds  to **moving left**.

## Steps to download this repository

Here are the steps to download this GitHub repository using the Bash command line:

 * Open a terminal window or Bash command prompt on your computer.
 * Navigate to the directory where you want to download the GitHub repository using the `cd` command.
 * Type the following command:

```
git clone https://github.com/axelctia/Renforcement_learning_ctia.git
```
 * Press Enter to execute the command. This will download the GitHub repository and create a new directory containing the project code in the current directory.

## Package required to the project
To read the project code, you need to install some packages.
 * Tqdm
 * Numpy
 * Gymnasium
 * Gymnasium [Atari]
 * Gymnasium [accept-rom-license]

Please execute the command below in terminal at the directory where you have downloaded this repository to install them:

```bash
python -m pip install -r requirements.txt
```
# 🛑 Note

If you launch main.py, the model will be train. But, if you don't want to train the model, you can use an existing model. For that, follow these steps :

* Unzip "objects.zip" in the same folder of main.py
* Launch "main.py"
* Enjoy !
