import gymnasium as gym
import minihack
env = gym.make("MiniHack-River-v0")
env.reset() # each reset generates a new environment instance
env.step(0)  # move agent '@' north
env.render()
