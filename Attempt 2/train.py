#!/usr/bin/env python3 

import numpy as np
import argparse
from copy import deepcopy
import torch
import gym

from normalized_env import NormalizedEnv
from evaluator import Evaluator
from ddpg import DDPG
from utils import *

gym.undo_logger_setup()

def train(num_iterations, agent, env, evaluate, validate_steps, output, max_episode_length=None, debug=False):

    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    while step < num_iterations:
        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)

        # agent pick action ...
        if step <= args.warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(observation)
        
        # env response with next_observation, reward, terminate_info
        observation2, reward, done, info = env.step(action)
        observation2 = deepcopy(observation2)
        if max_episode_length and episode_steps >= max_episode_length -1:
            done = True

        # agent observe and update policy
        agent.observe(reward, observation2, done)
        if step > args.warmup :
            agent.update_policy()
        
        # [optional] evaluate
        if evaluate is not None and validate_steps > 0 and step % validate_steps == 0:
            policy = lambda x: agent.select_action(x, decay_epsilon=False)
            validate_reward = evaluate(env, policy, debug=False, visualize=False)
            if debug: prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))

        # [optional] save intermideate model
        if step % int(num_iterations/3) == 0:
            agent.save_model(output)

        # update 
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if done: # end of episode
            if debug: prGreen('#{}: episode_reward:{} steps:{}'.format(episode,episode_reward,step))

            agent.memory.append(
                observation,
                agent.select_action(observation),
                0., False
            )

            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1

def test(num_episodes, agent, env, evaluate, model_path, visualize=True, debug=False):

    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    for i in range(num_episodes):
        validate_reward = evaluate(env, policy, debug=debug, visualize=visualize, save=False)
        if debug: prYellow('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--env', default='Pendulum-v0', type=str, help='open-ai gym environment')
    parser.add_argument('--hidden1', default=400, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=100, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=6000000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma') 
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu') 
    parser.add_argument('--validate_episodes', default=20, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=500, type=int, help='')
    parser.add_argument('--validate_steps', default=2000, type=int, help='how many steps to perform a validate experiment')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='') 
    parser.add_argument('--train_iter', default=200000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    # parser.add_argument('--l2norm', default=0.01, type=float, help='l2 weight decay') # TODO
    # parser.add_argument('--cuda', dest='cuda', action='store_true') # TODO

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)
    if args.resume == 'default':
        args.resume = 'output/{}-run0'.format(args.env)

    env = NormalizedEnv(gym.make(args.env))

    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)

    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]


    agent = DDPG(nb_states, nb_actions, args)
    evaluate = Evaluator(args.validate_episodes, 
        args.validate_steps, args.output, max_episode_length=args.max_episode_length)

    if args.mode == 'train':
        train(args.train_iter, agent, env, evaluate, 
            args.validate_steps, args.output, max_episode_length=args.max_episode_length, debug=args.debug)

    elif args.mode == 'test':
        test(args.validate_episodes, agent, env, evaluate, args.resume,
            visualize=True, debug=args.debug)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))

# from planner import Planner, save_model 
# import torch
# import torch.utils.tensorboard as tb
# import numpy as np
# from utils import load_data
# import dense_transforms

# def train(args):
#     from os import path
#     model = Planner()
#     train_logger, valid_logger = None, None
#     if args.log_dir is not None:
#         train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

#     """
#     Your code here, modify your HW4 code
    
#     """
#     import torch

#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#     model = model.to(device)
#     if args.continue_training:
#         model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'model.th')))

#     loss = torch.nn.L1Loss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
#     import inspect
#     transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})

#     train_data = load_data('drive_data', transform=transform, num_workers=args.num_workers)

#     global_step = 0
#     for epoch in range(args.num_epoch):
#         model.train()
#         losses = []
#         for img, label in train_data:
#             img, label = img.to(device), label.to(device)

#             pred = model(img)
#             loss_val = loss(pred, label)

#             if train_logger is not None:
#                 train_logger.add_scalar('loss', loss_val, global_step)
#                 if global_step % 100 == 0:
#                     log(train_logger, img, label, pred, global_step)

#             optimizer.zero_grad()
#             loss_val.backward()
#             optimizer.step()
#             global_step += 1
            
#             losses.append(loss_val.detach().cpu().numpy())
        
#         avg_loss = np.mean(losses)
#         if train_logger is None:
#             print('epoch %-3d \t loss = %0.3f' % (epoch, avg_loss))
#         save_model(model)

#     save_model(model)

# def log(logger, img, label, pred, global_step):
#     """
#     logger: train_logger/valid_logger
#     img: image tensor from data loader
#     label: ground-truth aim point
#     pred: predited aim point
#     global_step: iteration
#     """
#     import matplotlib.pyplot as plt
#     import torchvision.transforms.functional as TF
#     fig, ax = plt.subplots(1, 1)
#     ax.imshow(TF.to_pil_image(img[0].cpu()))
#     WH2 = np.array([img.size(-1), img.size(-2)])/2
#     ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
#     ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
#     logger.add_figure('viz', fig, global_step)
#     del ax, fig



# if __name__ == '__main__':
#     import argparse

#     parser = argparse.ArgumentParser()

#     parser.add_argument('--log_dir')
#     # Put custom arguments here
#     parser.add_argument('-n', '--num_epoch', type=int, default=150)
#     parser.add_argument('-w', '--num_workers', type=int, default=4)
#     parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
#     parser.add_argument('-c', '--continue_training', action='store_true')
#     parser.add_argument('-t', '--transform', default='Compose([ColorJitter(0.2, 0.5, 0.5, 0.2), RandomHorizontalFlip(), ToTensor()])')

#     args = parser.parse_args()
#     train(args)