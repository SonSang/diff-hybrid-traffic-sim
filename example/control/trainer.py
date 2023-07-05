from copy import deepcopy
import gc
import os
from gym import spaces
import torch as th
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from highway_env.envs.common.abstract import AbstractEnv
from example.control.controller import Controller
from PIL import Image

class Trainer:

    '''
    Trainer that trains a controller to solve the given traffic control problem
    using gradient-based optimization algorithms. Here we use simple gradient 
    descent to minimize the negative sum of rewards for given episodes.
    '''

    def __init__(self, env: AbstractEnv, network_size = [256, 256], lr = 1e-3):

        self.env = env

        input_size = self.env.observation_space.shape
        output_size = self.env.action_space.shape

        assert len(input_size) == 1 and len(output_size) == 1, ""

        input_size = input_size[0]
        output_size = output_size[0]

        self.controller = Controller(input_size, output_size, network_size)
        self.optimizer = th.optim.Adam(self.controller.parameters(), lr)
        self.best_eval_result = -float('inf')
        
        self.writer = None

    def train(self, 
                num_episode_per_epoch: int, 
                num_epoch: int,
                num_eval_epoch: int,
                num_eval_episode: int,
                log_path: str,):

        '''
        Train network.

        @ num_episode_per_epoch: Simulate this number of episodes and collect their rewards.
        Then, update once by minimizing the negative sum of these rewards.
        @ num_epoch: Total number of epochs to train.
        @ num_eval_epoch: Evaluate every [num_eval_epoch] epoch.
        @ num_eval_episode: Number of episodes used for evaluation.
        @ log_path: Path to store intermediate learning results.
        '''

        # init logging path;

        if not os.path.exists(log_path):

            os.makedirs(log_path)
            
        self.writer = SummaryWriter(log_path)

        epoch_tqdm = tqdm(range(num_epoch))

        self.best_eval_result = -float('inf')

        for epoch in epoch_tqdm:

            # evaluate;

            if epoch % num_eval_epoch == 0:

                self.evaluate(epoch, num_eval_episode, log_path)

            # train;

            self.controller.train(True)

            loss = self.train_epoch(num_episode_per_epoch)
            
            self.writer.add_scalar("loss/train", loss, epoch)
            
            epoch_tqdm.set_description("Loss: {:.6f}".format(loss))

            # free torch memory;

            gc.collect()
            th.cuda.empty_cache()

            self.save(log_path + "/model.zip")

    def evaluate(self, epoch: int, num_episode: int, log_path: str):
        
        '''
        Evaluate network.

        @ num_episode: Number of episodes to evaluate.
        '''

        self.controller.train(False)

        with th.no_grad():

            total_reward = 0

            for episode in range(num_episode):

                reward, _, info = self.run_episode(False)
                total_reward += reward
                
                # save img;
                img_list = info['img']
                img_path = log_path + '/eval_render/epoch_{}_episode_{}'.format(epoch, episode)
                
                if not os.path.exists(img_path):
                    os.makedirs(img_path)
                
                for i, img in enumerate(img_list):
                    
                    if img is not None:
                        
                        im = Image.fromarray(img)
                        im.save(img_path + "/{:08d}.png".format(i))
                        
        avg_reward = total_reward / num_episode

        self.writer.add_scalar("loss/eval", -avg_reward, epoch)
        
        with open(log_path + "/eval.txt", 'a') as f:
            f.write("{:08f}\n".format(-avg_reward))

        if avg_reward > self.best_eval_result:

            self.best_eval_result = avg_reward

            if not os.path.exists(log_path + "/best"):

                os.makedirs(log_path + "/best")

            self.save(log_path + "/best/model.zip")

    def train_epoch(self, num_episode):

        total_reward = 0

        for episode in range(num_episode):

            curr_reward, action, info = self.run_episode(True)

            total_reward = total_reward + curr_reward

            # epoch_tqdm.set_description("Train Epoch {}: Avg Reward = {:.2f}".format(epoch, total_reward / (episode + 1)))

        action.retain_grad()

        loss = (-total_reward) / num_episode

        self.optimizer.zero_grad()
        loss.backward()

        #print(action.grad)
        self.optimizer.step()
        
        return loss

    def run_episode(self, differentiable: bool):

        episode_reward = 0

        tenv = deepcopy(self.env)

        obs = tenv.observe()
        #print(obs[:10])

        # print(obs)

        while True:

            action = self.controller(th.tensor(obs))

            if isinstance(tenv.action_space, spaces.Box):

                low = th.tensor(tenv.action_space.low)
                high = th.tensor(tenv.action_space.high)

                action = low + (high - low) * th.sigmoid(action)

            #print(action)
                
            obs, reward, terminal, info = tenv.step(action, differentiable)

            episode_reward = episode_reward + reward

            if terminal:

                break

        # self.env.reset()

        # print(episode_reward)

        return episode_reward, action, info


    def save(self, path: str):

        '''
        Save model at the given path.
        '''

        th.save({
            'controller_state_dict': self.controller.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, path)

    def load(self, path: str):

        '''
        Load model at the given path.
        '''

        checkpoint = th.load(path)
        
        self.controller.load_state_dict(checkpoint['controller_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])