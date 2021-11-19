import time
import numpy as np
import tensorflow as tf
from gym import wrappers
from mujoco_py import GlfwContext


GlfwContext(offscreen=True)

from mujoco_py.generated import const


class Play:
    def __init__(self, env, agent, max_episode=4):
        self.env = env
        # self.env = wrappers.Monitor(env, "./videos", video_callable=lambda episode_id: True, force=True)
        self.max_episode = max_episode
        self.agent = agent
        self.agent.load_weights()
        self.agent.set_to_eval_mode()
        # self.device = device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(self):

        num_of_objects=4
        for _ in range(self.max_episode):
            # env_dict = self.env.reset()
            env_dict = self.env.reset_play(num_of_objects)
            state = env_dict["observation"]
            achieved_goal = env_dict["achieved_goal"]
            desired_goal = env_dict["desired_goal"]
            while np.linalg.norm(achieved_goal - desired_goal) <= 0.05:
                env_dict = self.env.reset_play(num_of_objects)
                state = env_dict["observation"]
                achieved_goal = env_dict["achieved_goal"]
                desired_goal = env_dict["desired_goal"]
            self.env.render(mode="human")
            done = False
            episode_reward = 0

            i = 0
            while i < num_of_objects:

                while not done:
                    action = self.agent.choose_action(state, desired_goal, train_mode=False)
                    next_env_dict, r, done, _ = self.env.step(action,i)
                    next_state = next_env_dict["observation"]
                    next_desired_goal = next_env_dict["desired_goal"]
                    episode_reward += r
                    state = next_state.copy()
                    desired_goal = next_desired_goal.copy()
                    I = self.env.render(mode="human")  # mode = "rgb_array
                    # self.env.viewer.cam.type = const.CAMERA_FREE
                    # self.env.viewer.cam.fixedcamid = 0
                    # I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
                    # cv2.imshow("I", I)
                    # cv2.waitKey(2)
                    
                
                done = False
                action = [-100,100,100,1]
                next_env_dict, r, done, _ = self.env.step_play(action,i)
                self.env.render(mode="human") 
                i += 1
            
            print(f"episode_reward:{episode_reward:3.3f}")
            action = [-100,100,100,1]
            next_env_dict, r, done, _ = self.env.step_play(action,i-1)
            self.env.render(mode="human") 
            #self.env.pause()

        self.env.close()
