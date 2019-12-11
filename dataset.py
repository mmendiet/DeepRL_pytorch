#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from agent import *
from component import *
from utils import *
import torchvision
import torch
from skimage import io

# PREFIX = '.'
#PREFIX = '/local/data'
PREFIX = '/1TB/Datasets/Atari/data_acvp2'

def dqn_pixel_atari(name):
    config = Config()
    config.history_length = 4
    config.task_fn = lambda: PixelAtari(name, no_op=30, frame_skip=4, normalized_state=False,
                                        history_length=config.history_length)
    action_dim = config.task_fn().action_dim
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01)
    config.network_fn = lambda: NatureConvNet(config.history_length, action_dim)
    #config.policy_fn = lambda: GreedyPolicy(epsilon=1.0, final_step=1000000, min_epsilon=0.1)
    #config.replay_fn = lambda: Replay(memory_size=1000000, batch_size=32, dtype=np.uint8)
    config.policy_fn = lambda: GreedyPolicy(epsilon=1.0, final_step=100000, min_epsilon=0.1)
    config.replay_fn = lambda: Replay(memory_size=100000, batch_size=32, dtype=np.uint8)
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.max_episode_length = 0
    config.exploration_steps = 50000
    config.logger = Logger('./log', logger)
    config.test_interval = 100 #how many episodes to train
    config.test_repetitions = 1
    config.double_q = False
    return DQNAgent(config)

def train_dqn(game):
    agent = dqn_pixel_atari(game)
    run_episodes(agent)

def episode(env, agent):
    config = agent.config
    policy = GreedyPolicy(epsilon=0.3, final_step=1, min_epsilon=0.3)
    state = env.reset()
    history_buffer = [state] * config.history_length
    state = np.vstack(history_buffer)
    total_reward = 0.0
    steps = 0
    while True:
        value = agent.learning_network.predict(np.stack([state]), False)
        value = value.cpu().data.numpy().flatten()
        action = policy.sample(value)
        next_state, reward, done, info = env.step(action)
        history_buffer.pop(0)
        history_buffer.append(next_state)
        state = np.vstack(history_buffer)
        done = (done or (config.max_episode_length and steps > config.max_episode_length))
        steps += 1
        total_reward += reward
        if done:
            break
    return total_reward, steps

def generate_dateset(game):
    agent = dqn_pixel_atari(game)
    model_file = 'data/%s-%s-model-%s.bin' % (agent.__class__.__name__, agent.config.tag, agent.task.name)
    with open(model_file, 'rb') as f:
        saved_state = torch.load(model_file, map_location=lambda storage, loc: storage)
        agent.learning_network.load_state_dict(saved_state)

    env = gym.make(game)
    env = EpisodicLifeEnv(env)
    env = MaxAndSkipEnv(env, skip=4)
    dataset_env = DatasetEnv(env)
    env = ProcessFrame(dataset_env, 84)
    env = NormalizeFrame(env)

    ep = 0
    #max_ep = 1
    mkdir('%s/dataset/%s' % (PREFIX, game))
    obs_sum = 0.0
    obs_count = 0
    obs_limit = 450000
    while (obs_count < obs_limit):
    #while True:
        rewards, steps = episode(env, agent)
        path = '%s/dataset/%s/%05d' % (PREFIX, game, ep)
        mkdir(path)
        logger.info('Episode %d, reward %f, steps %d' % (ep, rewards, steps))
        with open('%s/action.bin' % (path), 'wb') as f:
            pickle.dump(dataset_env.saved_actions, f)
        with open('%s/reward.bin' % (path), 'wb') as f:
            pickle.dump(dataset_env.saved_rewards, f)
        obs_sum += np.asarray(dataset_env.saved_obs).sum(0)
        obs_count += len(dataset_env.saved_obs)
        for ind, obs in enumerate(dataset_env.saved_obs):
            io.imsave('%s/%05d.png' % (path, ind), obs)
        dataset_env.clear_saved()
        ep += 1
        #if ep >= max_ep:
            #break
    obs_mean = np.transpose(obs_sum, (2, 0, 1)) / obs_count
    with open('%s/dataset/%s/meta.bin' % (PREFIX, game), 'wb') as f:
        pickle.dump({'episodes': ep,
                     'mean_obs': obs_mean}, f)

def generate_dateset2(multiGame):
    multi = 'PongBowl'
    ep = 0
    mkdir('%s/dataset/%s' % (PREFIX, multi))
    obs_sum = 0.0
    obs_count = 0
    obs_limit = 450000
    count = 1
    splitEp = 0
    for game in multiGame:
        agent = dqn_pixel_atari(game)
        model_file = 'data/%s-%s-model-%s.bin' % (agent.__class__.__name__, agent.config.tag, agent.task.name)
        with open(model_file, 'rb') as f:
            saved_state = torch.load(model_file, map_location=lambda storage, loc: storage)
            agent.learning_network.load_state_dict(saved_state)

        env = gym.make(game)
        env = EpisodicLifeEnv(env)
        env = MaxAndSkipEnv(env, skip=4)
        dataset_env = DatasetEnv(env)
        env = ProcessFrame(dataset_env, 84)
        env = NormalizeFrame(env)

        while (obs_count < (obs_limit*count)):
            rewards, steps = episode(env, agent)
            path = '%s/dataset/%s/%05d' % (PREFIX, multi, ep)
            mkdir(path)
            logger.info('Episode %d, reward %f, steps %d' % (ep, rewards, steps))
            with open('%s/action.bin' % (path), 'wb') as f:
                pickle.dump(dataset_env.saved_actions, f)
            with open('%s/reward.bin' % (path), 'wb') as f:
                pickle.dump(dataset_env.saved_rewards, f)
            obs_sum += np.asarray(dataset_env.saved_obs).sum(0)
            obs_count += len(dataset_env.saved_obs)
            for ind, obs in enumerate(dataset_env.saved_obs):
                io.imsave('%s/%05d.png' % (path, ind), obs)
            dataset_env.clear_saved()
            ep += 1
            #if ep >= max_ep:
                #break
        count += 1
        splitEp = ep
    obs_mean = np.transpose(obs_sum, (2, 0, 1)) / obs_count
    with open('%s/dataset/%s/meta.bin' % (PREFIX, multi), 'wb') as f:
        pickle.dump({'episodes': ep,
                    'mean_obs': obs_mean}, f)
    print("Split Episode:  " + str(splitEp))

if __name__ == '__main__':
    #mkdir('dataset')
    #game = 'BowlingNoFrameskip-v4'
    #multiGames = ['PongNoFrameskip-v4', 'BoxingNoFrameskip-v4', 'BowlingNoFrameskip-v4', 'FreewayNoFrameskip-v4',
        #'PongNoFrameskip-v4','QbertNoFrameskip-v4','MsPacmanNoFrameskip-v4',
        #'SeaquestNoFrameskip-v4','BowlingNoFrameskip-v4','CrazyClimberNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4']
    #for game in multiGames:
        #train_dqn(game)
    #for game in multiGames:
        #generate_dateset(game)
    

    #multiGames = ['AsterixNoFrameskip-v4', 'BoxingNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'FreewayNoFrameskip-v4',
        #'PongNoFrameskip-v4','QbertNoFrameskip-v4','MsPacmanNoFrameskip-v4',
        #'SeaquestNoFrameskip-v4','BowlingNoFrameskip-v4','CrazyClimberNoFrameskip-v4']
    #generate_dateset2(multiGames)
    game = 'MsPacmanNoFrameskip-v4'
    train_dqn(game)
    generate_dateset(game)
    #multiGames = ['PongNoFrameskip-v4', 'BowlingNoFrameskip-v4']
    #generate_dateset2(multiGames)

#AsterixNoFrameskip-v4
#BoxingNoFrameskip-v4
#BreakoutNoFrameskip-v4
#FreewayNoFrameskip-v4
#PongNoFrameskip-v4
#CrazyClimberNoFrameskip-v4
#QbertNoFrameskip-v4
#MsPacmanNoFrameskip-v4
#SpaceInvadersNoFrameskip-v4
#BowlingNoFrameskip-v4

#pong, bowling, qbert, boxing, #freeway, #pacman