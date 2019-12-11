#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import torchvision
from skimage import io
from collections import deque, defaultdict
import gym
import torch.optim
from utils import *
from tqdm import tqdm
from network import *
import os
from statistics import mean

#PREFIX = '.'
# PREFIX = '/local/data'
PREFIX = '/1TB/Datasets/Atari/data_acvp2'

class Network(nn.Module, BasicNet):
    def __init__(self, num_actions, gpu=0):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(12, 64, 8, 2, (0, 1))
        self.conv2 = nn.Conv2d(64, 128, 6, 2, (1, 1))
        self.conv3 = nn.Conv2d(128, 128, 6, 2, (1, 1))
        self.conv4 = nn.Conv2d(128, 128, 4, 2, (0, 0))

        self.hidden_units = 128 * 11 * 8

        self.fc5 = nn.Linear(self.hidden_units, 2048)
        self.fc_encode = nn.Linear(2048, 2048)
        self.fc_action = nn.Linear(num_actions, 2048)
        self.fc_decode = nn.Linear(2048, 2048)
        self.fc8 = nn.Linear(2048, self.hidden_units)
        ###
        #self.fcr = nn.Linear(self.hidden_units,1)
        ###

        self.deconv9 = nn.ConvTranspose2d(128, 128, 4, 2)
        self.deconv10 = nn.ConvTranspose2d(128, 128, 6, 2, (1, 1))
        self.deconv11 = nn.ConvTranspose2d(128, 128, 6, 2, (1, 1))#128 outchannels
        self.deconv12 = nn.ConvTranspose2d(128, 3, 8, 2, (0, 1))#128

        self.init_weights()
        self.criterion = nn.MSELoss()
        self.opt = torch.optim.Adam(self.parameters(), 1e-4)

        BasicNet.__init__(self, gpu)

    def init_weights(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_uniform(layer.weight.data)
                nn.init.constant(layer.bias.data, 0)
        nn.init.uniform(self.fc_encode.weight.data, -1, 1)
        nn.init.uniform(self.fc_decode.weight.data, -1, 1)
        nn.init.uniform(self.fc_action.weight.data, -0.1, 0.1)

    def forward(self, obs, action):
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view((-1, self.hidden_units))
        x = F.relu(self.fc5(x))
        x = self.fc_encode(x)
        action = self.fc_action(action)
        x = torch.mul(x, action)
        x = self.fc_decode(x)
        x = F.relu(self.fc8(x))
        x = x.view((-1, 128, 11, 8))
        x = F.relu(self.deconv9(x))
        x = F.relu(self.deconv10(x))
        x = F.relu(self.deconv11(x))
        x = self.deconv12(x)
        return x

    # def forward(self, obs, action):
    #     x1 = F.relu(self.conv1(obs))
    #     x2 = F.relu(self.conv2(x1))
    #     x3 = F.relu(self.conv3(x2))
    #     x4 = F.relu(self.conv4(x3))
    #     x = x4.view((-1, self.hidden_units))
    #     x = F.relu(self.fc5(x))
    #     x = self.fc_encode(x)
    #     action = self.fc_action(action)
    #     x = torch.mul(x, action)
    #     x = self.fc_decode(x)
    #     x = F.relu(self.fc8(x))
    #     ###
    #     #rew = self.fcr(x)
    #     ###
    #     x = x.view((-1, 128, 11, 8))
    #     x9 = F.relu(self.deconv9(x+x4))
    #     x10 = F.relu(self.deconv10(x9+x3))
    #     x11 = F.relu(self.deconv11(x10+x2))
    #     x12 = self.deconv12(x11+x1)
    #     return x12
    #     #return x12, rew

    def fit(self, x, a, y):
        x = self.variable(x)
        a = self.variable(a)
        y = self.variable(y)
        y_ = self.forward(x, a)
        loss = self.criterion(y_, y)
        self.opt.zero_grad()
        loss.backward()
        for param in self.parameters():
            param.grad.data.clamp_(-0.1, 0.1)
        self.opt.step()
        return np.asscalar(loss.cpu().data.numpy())

    def evaluate(self, x, a, y):
        x = self.variable(x)
        a = self.variable(a)
        y = self.variable(y)
        y_ = self.forward(x, a)
        loss = self.criterion(y_, y)
        return np.asscalar(loss.cpu().data.numpy())

    def predict(self, x, a):
        x = self.variable(x)
        a = self.variable(a)
        return self.forward(x, a).cpu().data.numpy()

def load_episode(game, ep, num_actions):
    path = '%s/dataset/%s/%05d' % (PREFIX, game, ep)
    with open('%s/action.bin' % (path), 'rb') as f:
        actions = pickle.load(f)
    num_frames = len(actions) + 1
    frames = []
    #start_frame = np.random.randint(1,num_frames-320)
    for i in range(1, num_frames):
    #for i in range(start_frame, start_frame+320):
        frame = io.imread('%s/%05d.png' % (path, i))
        frame = np.transpose(frame, (2, 0, 1))
        frames.append(frame.astype(np.uint8))

    actions = actions[1:]
    #actions = actions[(start_frame+1):(start_frame+321)]
    encoded_actions = np.zeros((len(actions), num_actions))
    encoded_actions[np.arange(len(actions)), actions] = 1

    return frames, encoded_actions

def extend_frames(frames, actions):
    buffer = deque(maxlen=4)
    extended_frames = []
    targets = []

    for i in range(len(frames) - 1):
        buffer.append(frames[i])
        if len(buffer) >= 4:
            extended_frames.append(np.vstack(buffer))
            targets.append(frames[i + 1])
    actions = actions[3:, :]

    return np.stack(extended_frames), actions, np.stack(targets)

def train(game):
    env = gym.make(game)
    num_actions = env.action_space.n

    net = Network(num_actions)

    with open('%s/dataset/%s/meta.bin' % (PREFIX, game), 'rb') as f:
        meta = pickle.load(f)
    episodes = meta['episodes']
    mean_obs = meta['mean_obs']

    def pre_process(x):
        if x.shape[1] == 12:
            return (x - np.vstack([mean_obs] * 4)) / 255.0
        elif x.shape[1] == 3:
            return (x - mean_obs) / 255.0
        else:
            assert False

    def post_process(y):
        return (y * 255 + mean_obs).astype(np.uint8)

    train_episodes = int(episodes * 0.95)
    indices_train = np.arange(train_episodes)
    iteration = 0
    while True:
        np.random.shuffle(indices_train)
        for ep in indices_train:
            frames, actions = load_episode(game, ep, num_actions)
            frames, actions, targets = extend_frames(frames, actions)
            batcher = Batcher(32, [frames, actions, targets])
            batcher.shuffle()
            while not batcher.end():
                if iteration % 10000 == 0:
                    mkdir('data/acvp-sample')
                    losses = []
                    test_indices = range(train_episodes, episodes)
                    ep_to_print = np.random.choice(test_indices)
                    for test_ep in tqdm(test_indices):
                        frames, actions = load_episode(game, test_ep, num_actions)
                        frames, actions, targets = extend_frames(frames, actions)
                        test_batcher = Batcher(32, [frames, actions, targets])
                        while not test_batcher.end():
                            x, a, y = test_batcher.next_batch()
                            losses.append(net.evaluate(pre_process(x), a, pre_process(y)))
                        if test_ep == ep_to_print:
                            test_batcher.reset()
                            x, a, y = test_batcher.next_batch()
                            y_ = post_process(net.predict(pre_process(x), a))
                            torchvision.utils.save_image(torch.from_numpy(y), 'data/acvp-sample/%s-%09d-truth.png' % (game, iteration))
                            torchvision.utils.save_image(torch.from_numpy(y_), 'data/acvp-sample/%s-%09d.png' % (game, iteration))

                    logger.info('Iteration %d, test loss %f' % (iteration, np.mean(losses)))
                    torch.save(net.state_dict(), 'data/acvp-%s.bin' % (game))

                x, a, y = batcher.next_batch()
                loss = net.fit(pre_process(x), a, pre_process(y))
                if iteration % 100 == 0:
                    logger.info('Iteration %d, loss %f' % (iteration, loss))

                iteration += 1

def trainSingleGame(game, numEpoch, batchSize, trainingSize, num_actions):
    #env = gym.make(game)
    #num_actions = env.action_space.n
    #num_actions = 18

    net = Network(num_actions)

    #Look for checkpoint
    gameDir = 'resultsNew2/'+game+str(trainingSize)+'pre'
    print(gameDir)
    sampleDir = gameDir+ '/samples'
    modelDir = gameDir+'/models'
    #pretrained = modelDir+'/acvp-'+game+'-08.bin'
    pretrained = 'resultsNew2/PongBowl2'+str(num_actions)+'/models/acvp-PongBowl2-09.bin'
    if os.path.exists(pretrained):
        net.load_state_dict(torch.load(pretrained))
        print("Grabbed Pretrained:" + pretrained)
    #net.load_state_dict(torch.load('/home/matias/Documents/fall2019/rl/DeepRL_pytorch/results/MultiPB6/models/acvp-MultiPB-19.bin'))
    with open('%s/dataset/%s/meta.bin' % (PREFIX, game), 'rb') as f:
        meta = pickle.load(f)
    episodes = meta['episodes']
    mean_obs = meta['mean_obs']

    def pre_process(x):
        if x.shape[1] == 12:
            return (x - np.vstack([mean_obs] * 4)) / 255.0
        elif x.shape[1] == 3:
            return (x - mean_obs) / 255.0
        else:
            assert False

    def post_process(y):
        return (y * 255 + mean_obs).astype(np.uint8)

    trainSplit = float(trainingSize/100)
    train_episodes = int(episodes * trainSplit)
    #test_episodes = episodes - int(episodes *(1-trainSplit))
    test_episodes = episodes - int(episodes *(1-0.89))
    indices_train = np.arange(train_episodes)
    iteration = 0
    mkdir(gameDir)
    mkdir(sampleDir)
    mkdir(modelDir)

    for epoch in range(0,numEpoch):
        print("Starting Epoch:   " + str(epoch))
        #start Training
        np.random.shuffle(indices_train)
        for ep in indices_train:
            frames, actions = load_episode(game, ep, num_actions)
            frames, actions, targets = extend_frames(frames, actions)
            batcher = Batcher(batchSize, [frames, actions, targets])
            batcher.shuffle()
            while not batcher.end():
                x, a, y = batcher.next_batch()
                loss = net.fit(pre_process(x), a, pre_process(y))
                #print(iteration)
                if iteration % 100 == 0:
                    logger.info('Iteration %d, loss %f' % (iteration, loss))
                iteration += 1

        #Save model every epoch
        losses = []
        test_indices = range(test_episodes, episodes)
        ep_to_print = np.random.choice(test_indices)
        for test_ep in tqdm(test_indices):
            frames, actions = load_episode(game, test_ep, num_actions)
            frames, actions, targets = extend_frames(frames, actions)
            test_batcher = Batcher(batchSize, [frames, actions, targets])
            while not test_batcher.end():
                x, a, y = test_batcher.next_batch()
                losses.append(net.evaluate(pre_process(x), a, pre_process(y)))
            if test_ep == ep_to_print:
                test_batcher.reset()
                x, a, y = test_batcher.next_batch()
                y_ = post_process(net.predict(pre_process(x), a))
                torchvision.utils.save_image(torch.from_numpy(x[0:32,9:12]), sampleDir+'/%s-%02d-input.png' % (game, epoch))
                torchvision.utils.save_image(torch.from_numpy(y[0:32]), sampleDir+'/%s-%02d-truth.png' % (game, epoch))
                torchvision.utils.save_image(torch.from_numpy(y_[0:32]), sampleDir+'/%s-%02d.png' % (game, epoch))

        logger.info('Epoch %d, test loss %f' % (epoch, mean(losses)))
        f = open(gameDir+ '/results.txt', 'a')
        f.write('Epoch:  ' + str(epoch)+ ',   test loss:  ' + str(mean(losses))+ '\n')
        f.close()
        torch.save(net.state_dict(), modelDir+'/acvp-%s-%02d.bin' % (game, epoch))

def trainSingleGameTorch(game, numEpoch, batchSize, trainingSize):
    env = gym.make(game)
    num_actions = env.action_space.n

    net = Network(num_actions)

    #Look for checkpoint
    gameDir = 'results/'+game+str(trainingSize)
    sampleDir = gameDir+ '/samples'
    modelDir = gameDir+'/models'
    pretrained = modelDir+'/acvp-'+game+str(trainingSize)+'.bin'
    if os.path.exists(pretrained):
        net.load_state_dict(torch.load(pretrained))

    with open('%s/dataset/%s/meta.bin' % (PREFIX, game), 'rb') as f:
        meta = pickle.load(f)
    episodes = meta['episodes']
    mean_obs = meta['mean_obs']

    def pre_process(x):
        if x.shape[1] == 12:
            return (x - np.vstack([mean_obs] * 4)) / 255.0
        elif x.shape[1] == 3:
            return (x - mean_obs) / 255.0
        else:
            assert False

    def post_process(y):
        return (y * 255 + mean_obs).astype(np.uint8)

    train_episodes = int(episodes * float(trainingSize/100))
    test_episodes = episodes - int(episodes *0.1)
    indices_train = np.arange(train_episodes)
    iteration = 0
    mkdir(gameDir)
    mkdir(sampleDir)
    mkdir(modelDir)

    first_visit = 1
    for ep in indices_train:
        frames, actions = load_episode(game, ep, num_actions)
        frames, actions, targets = extend_frames(frames, actions)
        if first_visit:
            all_frames = torch.from_numpy(frames)
            all_actions = torch.from_numpy(actions)
            all_targets = torch.from_numpy(targets)
            first_visit = 0
        else:
            all_frames = torch.cat((all_frames,torch.from_numpy(frames)))
            all_actions = torch.cat((all_actions,torch.from_numpy(actions)))
            all_targets = torch.cat((all_targets,torch.from_numpy(targets)))
    training_set = torch.utils.data.TensorDataset(all_frames, all_actions, all_targets)
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=batchSize, shuffle=True,
        num_workers=12, pin_memory=True)

    first_visit = 1
    test_indices = range(test_episodes, episodes)
    for ep in test_indices:
        frames, actions = load_episode(game, ep, num_actions)
        frames, actions, targets = extend_frames(frames, actions)
        if first_visit:
            all_frames = torch.from_numpy(frames)
            all_actions = torch.from_numpy(actions)
            all_targets = torch.from_numpy(targets)
            first_visit = 0
        else:
            all_frames = torch.cat((all_frames,torch.from_numpy(frames)))
            all_actions = torch.cat((all_actions,torch.from_numpy(actions)))
            all_targets = torch.cat((all_targets,torch.from_numpy(targets)))
    training_set = torch.utils.data.TensorDataset(all_frames, all_actions, all_targets)
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=batchSize, shuffle=True,
        num_workers=12, pin_memory=True)

    f = open(gameDir+'/results.txt', 'a')
    for epoch in range(0,numEpoch):
        print("Starting Epoch:   " + str(epoch))
        np.random.shuffle(indices_train)
        for x, a, y in training_loader:
            loss = net.fit(pre_process(x.numpy()), a.numpy(), pre_process(y.numpy()))
            if iteration % 100 == 0:
                logger.info('Iteration %d, loss %f' % (iteration, loss))

            iteration += 1
        #Save model every epoch
        losses = []
        ep_to_print = np.random.choice(test_indices)
        ep_count = 0
        for x, a, y in tqdm(training_loader):
            x = x.numpy()
            a = a.numpy()
            y = y.numpy()
            losses.append(net.evaluate(pre_process(x), a, pre_process(y)))
            if test_indices[ep_count] == ep_to_print:
                y_ = post_process(net.predict(pre_process(x), a))
                torchvision.utils.save_image(torch.from_numpy(x[0:32,9:12]), sampleDir+'/%s-%02d-input.png' % (game, epoch))
                torchvision.utils.save_image(torch.from_numpy(y[0:32]), sampleDir+'/%s-%02d-truth.png' % (game, epoch))
                torchvision.utils.save_image(torch.from_numpy(y_[0:32]), sampleDir+'/%s-%02d.png' % (game, epoch))
            ep_count+=1

        logger.info('Epoch %d, test loss %f' % (epoch, np.mean(losses)))
        f.write('Epoch:  ' + str(epoch)+ ',   test loss:  ' + str(np.mean(losses))+ '\n')
        torch.save(net.state_dict(), modelDir+'/acvp-%s-%02d.bin' % (game, epoch))
    f.close()


def trainMultiGame(game, numEpoch, batchSize, num_actions):
    #env = gym.make(game)
    #num_actions = env.action_space.n
    #num_actions = 6

    net = Network(num_actions)
        #'BowlingNoFrameskip-v4'
        #'SeaquestNoFrameskip-v4'
        #'SpaceInvadersNoFrameskip-v4'

    #Look for checkpoint
    gameDir = 'resultsNew2/'+game+str(num_actions)
    print(gameDir)
    sampleDir = gameDir+ '/samples'
    modelDir = gameDir+'/models'
    pretrained = modelDir+'/acvp-'+game+'-00.bin'
    if os.path.exists(pretrained):
        net.load_state_dict(torch.load(pretrained))
    #net.load_state_dict(torch.load('acvp-MultiPB-00.bin'))
    #with open('%s/dataset/%s/meta.bin' % (PREFIX, 'PongBowl'), 'rb') as f:
        #meta = pickle.load(f)
    #episodes = meta['episodes']
    #mean_obs = meta['mean_obs']
    with open('%s/dataset/BowlingNoFrameskip-v4/meta.bin' % (PREFIX), 'rb') as f:
        meta = pickle.load(f)
    #episodes = meta['episodes']
    mean_obsB = meta['mean_obs']

    with open('%s/dataset/PongNoFrameskip-v4/meta.bin' % (PREFIX), 'rb') as f:
        meta = pickle.load(f)
    #episodes = meta['episodes']
    mean_obsP = meta['mean_obs']

    mean_obs = (mean_obsB + mean_obsP)/2

    def pre_process(x,mean_obs):
        if x.shape[1] == 12:
            return (x - np.vstack([mean_obs] * 4)) / 255.0
        elif x.shape[1] == 3:
            return (x - mean_obs) / 255.0
        else:
            assert False

    def post_process(y,mean_obs):
        return (y * 255 + mean_obs).astype(np.uint8)

    # testNum = episodes*dataPortion*(1-trainingSize)
    # trainNum = episodes*dataPortion*trainingSize
    # e = np.arange(episodes)
    # testIdx = int(episodes/testNum)
    # test_episodes = e[testIdx::testIdx]
    # trainIdx = int((episodes-testNum)/trainNum)
    # train_episodes = np.setdiff1d(e, test_episodes)[0::trainIdx]
    train_episodes = np.concatenate((np.arange(144),np.arange(161,349)))
    test_episodes = np.concatenate((np.arange(144,161),np.arange(349,373)))
    #train_episodes = np.concatenate((np.arange(146),np.arange(163,350)))
    #test_episodes = np.concatenate((np.arange(146,163),np.arange(350,375)))
    np.random.shuffle(test_episodes)
    np.random.shuffle(train_episodes)
    iteration = 0
    mkdir(gameDir)
    mkdir(sampleDir)
    mkdir(modelDir)
    
    f = open(gameDir+'/results.txt', 'a')
    for epoch in range(0,numEpoch):
        print("Starting Epoch:   " + str(epoch))
        np.random.shuffle(train_episodes)
        for ep in train_episodes:
            if (ep > 160):
                m = mean_obs
            else:
                m = mean_obs
            frames, actions = load_episode('PongBowl2', ep, num_actions)
            frames, actions, targets = extend_frames(frames, actions)
            batcher = Batcher(batchSize, [frames, actions, targets])
            batcher.shuffle()
            while not batcher.end():
                x, a, y = batcher.next_batch()
                loss = net.fit(pre_process(x,m), a, pre_process(y,m))
                if iteration % 100 == 0:
                    logger.info('Iteration %d, loss %f' % (iteration, loss))
                iteration += 1
        
        tep = np.random.randint(161,349)
        frames, actions = load_episode('PongBowl2', tep, num_actions)
        frames, actions, targets = extend_frames(frames, actions)
        batcher = Batcher(batchSize, [frames, actions, targets])
        batcher.shuffle()
        b_count = 0
        while not batcher.end():
            x, a, y = batcher.next_batch()
            loss = net.fit(pre_process(x,m), a, pre_process(y,m))
            if b_count >20:
                break
            b_count += 1

        tep = np.random.randint(161,349)
        frames, actions = load_episode('PongBowl2', tep, num_actions)
        frames, actions, targets = extend_frames(frames, actions)
        batcher = Batcher(batchSize, [frames, actions, targets])
        batcher.shuffle()
        b_count = 0
        while not batcher.end():
            x, a, y = batcher.next_batch()
            loss = net.fit(pre_process(x,m), a, pre_process(y,m))
            if b_count >20:
                break
            b_count += 1


        #Save model every epoch
        losses = []
        ep_to_print = np.random.choice(test_episodes)
        for test_ep in tqdm(test_episodes):
            if (ep > 160):
                m = mean_obs
            else:
                m = mean_obs
            frames, actions = load_episode('PongBowl2', test_ep, num_actions)
            frames, actions, targets = extend_frames(frames, actions)
            test_batcher = Batcher(batchSize, [frames, actions, targets])
            while not test_batcher.end():
                x, a, y = test_batcher.next_batch()
                losses.append(net.evaluate(pre_process(x,m), a, pre_process(y,m)))
            if test_ep == ep_to_print:
                test_batcher.reset()
                x, a, y = test_batcher.next_batch()
                y_ = post_process(net.predict(pre_process(x,m), a),m)
                torchvision.utils.save_image(torch.from_numpy(x[0:32,9:12]), sampleDir+'/%s-%02d-input.png' % (game, epoch))
                torchvision.utils.save_image(torch.from_numpy(y[0:32]), sampleDir+'/%s-%02d-truth.png' % (game, epoch))
                torchvision.utils.save_image(torch.from_numpy(y_[0:32]), sampleDir+'/%s-%02d.png' % (game, epoch))

        logger.info('Epoch %d, test loss %f' % (epoch, mean(losses)))
        f.write('Epoch:  ' + str(epoch)+ ',   test loss:  ' + str(mean(losses))+ '\n')
        torch.save(net.state_dict(), modelDir+'/acvp-%s-%02d.bin' % (game, epoch))
    f.close()

def testMultiGame2(game, numEpoch, batchSize, num_actions):
    #env = gym.make(game)
    #num_actions = env.action_space.n
    #num_actions = 6

    net = Network(num_actions)
        #'BowlingNoFrameskip-v4'
        #'SeaquestNoFrameskip-v4'
        #'SpaceInvadersNoFrameskip-v4'

    #Look for checkpoint
    gameDir = 'resultsNew2/'+game+str(num_actions)+'Pongtest'
    print(gameDir)
    sampleDir = gameDir+ '/samples'
    modelDir = gameDir+'/models'
    pretrained = 'resultsNew2/'+game+str(num_actions)+'/models/acvp-'+game+'-09.bin'
    if os.path.exists(pretrained):
        net.load_state_dict(torch.load(pretrained))
        print("Loaded:  " + pretrained)
    #net.load_state_dict(torch.load('acvp-MultiPB-00.bin'))
    #with open('%s/dataset/%s/meta.bin' % (PREFIX, 'PongBowl'), 'rb') as f:
        #meta = pickle.load(f)
    #episodes = meta['episodes']
    #mean_obs = meta['mean_obs']
    with open('%s/dataset/BowlingNoFrameskip-v4/meta.bin' % (PREFIX), 'rb') as f:
        meta = pickle.load(f)
    #episodes = meta['episodes']
    mean_obsB = meta['mean_obs']

    with open('%s/dataset/PongNoFrameskip-v4/meta.bin' % (PREFIX), 'rb') as f:
        meta = pickle.load(f)
    #episodes = meta['episodes']
    mean_obsP = meta['mean_obs']

    mean_obs = (mean_obsB + mean_obsP)/2

    def pre_process(x,mean_obs):
        if x.shape[1] == 12:
            return (x - np.vstack([mean_obs] * 4)) / 255.0
        elif x.shape[1] == 3:
            return (x - mean_obs) / 255.0
        else:
            assert False

    def post_process(y,mean_obs):
        return (y * 255 + mean_obs).astype(np.uint8)

    # testNum = episodes*dataPortion*(1-trainingSize)
    # trainNum = episodes*dataPortion*trainingSize
    # e = np.arange(episodes)
    # testIdx = int(episodes/testNum)
    # test_episodes = e[testIdx::testIdx]
    # trainIdx = int((episodes-testNum)/trainNum)
    # train_episodes = np.setdiff1d(e, test_episodes)[0::trainIdx]
    train_episodes = np.concatenate((np.arange(144),np.arange(161,349)))
    #test_episodes = np.concatenate((np.arange(144,161),np.arange(349,373)))
    #train_episodes = np.concatenate((np.arange(146),np.arange(163,350)))
    #test_episodes = np.concatenate((np.arange(146,163),np.arange(350,375)))
    #np.random.shuffle(test_episodes)
    np.random.shuffle(train_episodes)
    iteration = 0
    mkdir(gameDir)
    mkdir(sampleDir)
    mkdir(modelDir)
    test_episodes = np.arange(349,373)
    np.random.shuffle(test_episodes)
    
    f = open(gameDir+'/results.txt', 'a')
    for epoch in range(0,numEpoch):
        # print("Starting Epoch:   " + str(epoch))
        # np.random.shuffle(train_episodes)
        # for ep in train_episodes:
        #     if (ep > 160):
        #         m = mean_obs
        #     else:
        #         m = mean_obs
        #     frames, actions = load_episode('PongBowl2', ep, num_actions)
        #     frames, actions, targets = extend_frames(frames, actions)
        #     batcher = Batcher(batchSize, [frames, actions, targets])
        #     batcher.shuffle()
        #     while not batcher.end():
        #         x, a, y = batcher.next_batch()
        #         loss = net.fit(pre_process(x,m), a, pre_process(y,m))
        #         if iteration % 100 == 0:
        #             logger.info('Iteration %d, loss %f' % (iteration, loss))
        #         iteration += 1
        
        # tep = np.random.randint(161,349)
        # frames, actions = load_episode('PongBowl2', tep, num_actions)
        # frames, actions, targets = extend_frames(frames, actions)
        # batcher = Batcher(batchSize, [frames, actions, targets])
        # batcher.shuffle()
        # b_count = 0
        # while not batcher.end():
        #     x, a, y = batcher.next_batch()
        #     loss = net.fit(pre_process(x,m), a, pre_process(y,m))
        #     if b_count >20:
        #         break
        #     b_count += 1

        # tep = np.random.randint(161,349)
        # frames, actions = load_episode('PongBowl2', tep, num_actions)
        # frames, actions, targets = extend_frames(frames, actions)
        # batcher = Batcher(batchSize, [frames, actions, targets])
        # batcher.shuffle()
        # b_count = 0
        # while not batcher.end():
        #     x, a, y = batcher.next_batch()
        #     loss = net.fit(pre_process(x,m), a, pre_process(y,m))
        #     if b_count >20:
        #         break
        #     b_count += 1


        #Save model every epoch
        losses = []
        ep_to_print = np.random.choice(test_episodes)
        for test_ep in tqdm(test_episodes):
            if (test_ep > 160):
                m = mean_obs
            else:
                m = mean_obs
            frames, actions = load_episode('PongBowl2', test_ep, num_actions)
            frames, actions, targets = extend_frames(frames, actions)
            test_batcher = Batcher(batchSize, [frames, actions, targets])
            while not test_batcher.end():
                x, a, y = test_batcher.next_batch()
                losses.append(net.evaluate(pre_process(x,m), a, pre_process(y,m)))
            if test_ep == ep_to_print:
                test_batcher.reset()
                x, a, y = test_batcher.next_batch()
                y_ = post_process(net.predict(pre_process(x,m), a),m)
                torchvision.utils.save_image(torch.from_numpy(x[0:32,9:12]), sampleDir+'/%s-%02d-input.png' % (game, epoch))
                torchvision.utils.save_image(torch.from_numpy(y[0:32]), sampleDir+'/%s-%02d-truth.png' % (game, epoch))
                torchvision.utils.save_image(torch.from_numpy(y_[0:32]), sampleDir+'/%s-%02d.png' % (game, epoch))

        logger.info('Epoch %d, test loss %f' % (epoch, mean(losses)))
        f.write('Epoch:  ' + str(epoch)+ ',   test loss:  ' + str(mean(losses))+ '\n')
        #torch.save(net.state_dict(), modelDir+'/acvp-%s-%02d.bin' % (game, epoch))
    f.close()

def trainThreeGame(game, numEpoch, batchSize, num_actions):
    #env = gym.make(game)
    #num_actions = env.action_space.n
    #num_actions = 6

    net = Network(num_actions)
        #'BowlingNoFrameskip-v4'
        #'SeaquestNoFrameskip-v4'
        #'SpaceInvadersNoFrameskip-v4'

    #Look for checkpoint
    gameDir = 'resultsNew2/'+game+str(num_actions)
    print(gameDir)
    sampleDir = gameDir+ '/samples'
    modelDir = gameDir+'/models'
    #pretrained = modelDir+'/acvp-'+game+'-00.bin'
    #if os.path.exists(pretrained):
        #net.load_state_dict(torch.load(pretrained))
    #net.load_state_dict(torch.load('acvp-MultiPB-00.bin'))
    #with open('%s/dataset/%s/meta.bin' % (PREFIX, 'PongBowl'), 'rb') as f:
        #meta = pickle.load(f)
    #episodes = meta['episodes']
    #mean_obs = meta['mean_obs']
    with open('%s/dataset/BowlingNoFrameskip-v4/meta.bin' % (PREFIX), 'rb') as f:
        meta = pickle.load(f)
    #episodes = meta['episodes']
    mean_obsB = meta['mean_obs']

    with open('%s/dataset/PongNoFrameskip-v4/meta.bin' % (PREFIX), 'rb') as f:
        meta = pickle.load(f)
    #episodes = meta['episodes']
    mean_obsP = meta['mean_obs']

    with open('%s/dataset/QbertNoFrameskip-v4/meta.bin' % (PREFIX), 'rb') as f:
        meta = pickle.load(f)
    #episodes = meta['episodes']
    mean_obsQ = meta['mean_obs']

    mean_obs = (mean_obsB + mean_obsP + mean_obsQ)/3

    def pre_process(x,mean_obs):
        if x.shape[1] == 12:
            return (x - np.vstack([mean_obs] * 4)) / 255.0
        elif x.shape[1] == 3:
            return (x - mean_obs) / 255.0
        else:
            assert False

    def post_process(y,mean_obs):
        return (y * 255 + mean_obs).astype(np.uint8)

    # testNum = episodes*dataPortion*(1-trainingSize)
    # trainNum = episodes*dataPortion*trainingSize
    # e = np.arange(episodes)
    # testIdx = int(episodes/testNum)
    # test_episodes = e[testIdx::testIdx]
    # trainIdx = int((episodes-testNum)/trainNum)
    # train_episodes = np.setdiff1d(e, test_episodes)[0::trainIdx]
    train_episodes = np.concatenate((np.arange(144),np.arange(161,349)))
    test_episodes = np.concatenate((np.arange(144,161),np.arange(349,373)))
    #train_episodes = np.concatenate((np.arange(146),np.arange(163,350)))
    #test_episodes = np.concatenate((np.arange(146,163),np.arange(350,375)))
    np.random.shuffle(test_episodes)
    np.random.shuffle(train_episodes)
    iteration = 0
    mkdir(gameDir)
    mkdir(sampleDir)
    mkdir(modelDir)
    
    f = open(gameDir+'/results.txt', 'a')
    for epoch in range(0,numEpoch):
        print("Starting Epoch:   " + str(epoch))
        np.random.shuffle(train_episodes)
        for ep in train_episodes:
            if (ep > 160):
                m = mean_obs
            else:
                m = mean_obs
            frames, actions = load_episode('PongBowl2', ep, num_actions)
            frames, actions, targets = extend_frames(frames, actions)
            batcher = Batcher(batchSize, [frames, actions, targets])
            batcher.shuffle()
            while not batcher.end():
                x, a, y = batcher.next_batch()
                loss = net.fit(pre_process(x,m), a, pre_process(y,m))
                if iteration % 100 == 0:
                    logger.info('Iteration %d, loss %f' % (iteration, loss))
                iteration += 1
        
        tep = np.random.randint(161,349)
        frames, actions = load_episode('PongBowl2', tep, num_actions)
        frames, actions, targets = extend_frames(frames, actions)
        batcher = Batcher(batchSize, [frames, actions, targets])
        batcher.shuffle()
        b_count = 0
        while not batcher.end():
            x, a, y = batcher.next_batch()
            loss = net.fit(pre_process(x,m), a, pre_process(y,m))
            if b_count >20:
                break
            b_count += 1

        tep = np.random.randint(161,349)
        frames, actions = load_episode('PongBowl2', tep, num_actions)
        frames, actions, targets = extend_frames(frames, actions)
        batcher = Batcher(batchSize, [frames, actions, targets])
        batcher.shuffle()
        b_count = 0
        while not batcher.end():
            x, a, y = batcher.next_batch()
            loss = net.fit(pre_process(x,m), a, pre_process(y,m))
            if b_count >20:
                break
            b_count += 1


        #Save model every epoch
        losses = []
        ep_to_print = np.random.choice(test_episodes)
        for test_ep in tqdm(test_episodes):
            if (ep > 160):
                m = mean_obs
            else:
                m = mean_obs
            frames, actions = load_episode('PongBowl2', test_ep, num_actions)
            frames, actions, targets = extend_frames(frames, actions)
            test_batcher = Batcher(batchSize, [frames, actions, targets])
            while not test_batcher.end():
                x, a, y = test_batcher.next_batch()
                losses.append(net.evaluate(pre_process(x,m), a, pre_process(y,m)))
            if test_ep == ep_to_print:
                test_batcher.reset()
                x, a, y = test_batcher.next_batch()
                y_ = post_process(net.predict(pre_process(x,m), a),m)
                torchvision.utils.save_image(torch.from_numpy(x[0:32,9:12]), sampleDir+'/%s-%02d-input.png' % (game, epoch))
                torchvision.utils.save_image(torch.from_numpy(y[0:32]), sampleDir+'/%s-%02d-truth.png' % (game, epoch))
                torchvision.utils.save_image(torch.from_numpy(y_[0:32]), sampleDir+'/%s-%02d.png' % (game, epoch))

        logger.info('Epoch %d, test loss %f' % (epoch, mean(losses)))
        f.write('Epoch:  ' + str(epoch)+ ',   test loss:  ' + str(mean(losses))+ '\n')
        torch.save(net.state_dict(), modelDir+'/acvp-%s-%02d.bin' % (game, epoch))
    f.close()


def testMultiGame(game):
    num_actions = 6

    net = Network(num_actions)

    #Look for checkpoint
    gameDir = 'resultsNew2/'+game+str(num_actions)
    print(gameDir)
    sampleDir = gameDir+ '/samples'
    modelDir = gameDir+'/models'
    #gameDir = 'results/SItest/'+game
    #print(gameDir)
    #pretrained = modelDir+'/acvp-'+game+str(trainingSize)+'pre.bin'
    #pretrained = '/home/matias/Documents/fall2019/rl/DeepRL_pytorch/results/SpaceInvadersNoFrameskip-v42518/models/acvp-SpaceInvadersNoFrameskip-v4-09.bin'
    #if os.path.exists(pretrained):
        #net.load_state_dict(torch.load(pretrained))
    net.load_state_dict(torch.load('/home/matias/Documents/fall2019/rl/DeepRL_pytorch/results/QbertNoFrameskip-v412small/models/acvp-QbertNoFrameskip-v4-09.bin'))
    with open('%s/dataset/%s/meta.bin' % (PREFIX, game), 'rb') as f:
        meta = pickle.load(f)
    episodes = meta['episodes']
    mean_obs = meta['mean_obs']

    def pre_process(x):
        if x.shape[1] == 12:
            return (x - np.vstack([mean_obs] * 4)) / 255.0
        elif x.shape[1] == 3:
            return (x - mean_obs) / 255.0
        else:
            assert False

    def post_process(y):
        return (y * 255 + mean_obs).astype(np.uint8)

    mkdir(gameDir)

    test_ep = 42
    frames, actions = load_episode(game, test_ep, num_actions)
    frames, actions, targets = extend_frames(frames, actions)
    test_batcher = Batcher(32, [frames, actions, targets])
    count = 0
    losses = []
    while not test_batcher.end():
        x, a, y = test_batcher.next_batch()
        y_ = post_process(net.predict(pre_process(x), a))
        torchvision.utils.save_image(torch.from_numpy(x[0:32,9:12]), gameDir+'/%s-%02d-input.png' % (game, count))
        torchvision.utils.save_image(torch.from_numpy(y[0:32]), gameDir+'/%s-%02d-truth.png' % (game, count))
        torchvision.utils.save_image(torch.from_numpy(y_[0:32]), gameDir+'/%s-%02d.png' % (game, count))
        count += 1


def testSingleGame(game, numEpoch, batchSize, trainingSize, num_actions):
    #env = gym.make(game)
    #num_actions = env.action_space.n
    #num_actions = 18

    net = Network(num_actions)

    #Look for checkpoint

    #pretrained = modelDir+'/acvp-'+game+'-08.bin'
    for x in trainingSize:
        gameDir = 'tests/'+game+str(x)
        print(gameDir)
        sampleDir = gameDir+ '/samples'
        pretrained = '/home/matias/Documents/fall2019/rl/DeepRL_pytorch/resultsNew2/MsPacmanNoFrameskip-v4'+str(x)+'/models/acvp-msPacmanNoFrameskip-v4-09.bin'
        if os.path.exists(pretrained):
            net.load_state_dict(torch.load(pretrained))
            print("Grabbed Pretrained:" + pretrained)
        #net.load_state_dict(torch.load('/home/matias/Documents/fall2019/rl/DeepRL_pytorch/results/MultiPB6/models/acvp-MultiPB-19.bin'))
        with open('%s/dataset/%s/meta.bin' % (PREFIX, game), 'rb') as f:
            meta = pickle.load(f)
        episodes = meta['episodes']
        mean_obs = meta['mean_obs']

        def pre_process(x):
            if x.shape[1] == 12:
                return (x - np.vstack([mean_obs] * 4)) / 255.0
            elif x.shape[1] == 3:
                return (x - mean_obs) / 255.0
            else:
                assert False

        def post_process(y):
            return (y * 255 + mean_obs).astype(np.uint8)

        #trainSplit = float(trainingSize/100)
        #train_episodes = int(episodes * trainSplit)
        #test_episodes = episodes - int(episodes *(1-trainSplit))
        test_episodes = episodes - int(episodes *(1-0.89))
        #indices_train = np.arange(train_episodes)
        iteration = 0
        mkdir(gameDir)
        mkdir(sampleDir)

        losses = []
        #test_indices = range(test_episodes, episodes)
        #ep_to_print = np.random.choice(test_indices)
        #for test_ep in tqdm(test_indices):
        test_ep = 39
        frames, actions = load_episode(game, test_ep, num_actions)
        frames, actions, targets = extend_frames(frames, actions)
        test_batcher = Batcher(batchSize, [frames, actions, targets])
        while not test_batcher.end():
            x, a, y = test_batcher.next_batch()
            losses.append(net.evaluate(pre_process(x), a, pre_process(y)))
            #if test_ep == ep_to_print:
            # test_batcher.reset()
            # x, a, y = test_batcher.next_batch()
            y_ = post_process(net.predict(pre_process(x), a))
            #torchvision.utils.save_image(torch.from_numpy(x[0:32,9:12]), sampleDir+'/%s-%02d-input.png' % (game, iteration))
            torchvision.utils.save_image(torch.from_numpy(y[0:32]), sampleDir+'/%s-%02d-truth.png' % (game, iteration))
            torchvision.utils.save_image(torch.from_numpy(y_[0:32]), sampleDir+'/%s-%02d.png' % (game, iteration))
            f = open(gameDir+ '/results.txt', 'a')
            f.write('Iteration:  ' + str(iteration)+ ',   iter loss:  ' + str(losses[iteration])+ '\n')
            f.close()
            iteration+=1


def testMGame():
    #env = gym.make(game)
    #num_actions = env.action_space.n
    #num_actions = 18

    

    #Look for checkpoint

    #pretrained = modelDir+'/acvp-'+game+'-08.bin'
    # with open('%s/dataset/%s/meta.bin' % (PREFIX, game), 'rb') as f:
    #     meta = pickle.load(f)
    # episodes = meta['episodes']
    # mean_obs = meta['mean_obs']

    with open('%s/dataset/BowlingNoFrameskip-v4/meta.bin' % (PREFIX), 'rb') as f:
        meta = pickle.load(f)
    #episodes = meta['episodes']
    mean_obsB = meta['mean_obs']

    with open('%s/dataset/PongNoFrameskip-v4/meta.bin' % (PREFIX), 'rb') as f:
        meta = pickle.load(f)
    #episodes = meta['episodes']
    mean_obsP = meta['mean_obs']

    mean_obs = (mean_obsB + mean_obsP)/2



    with open('%s/dataset/%s/meta.bin' % (PREFIX, 'BoxingNoFrameskip-v4'), 'rb') as f:
        meta = pickle.load(f)
    mean_obsBox = meta['mean_obs']

    with open('%s/dataset/%s/meta.bin' % (PREFIX, 'QbertNoFrameskip-v4'), 'rb') as f:
        meta = pickle.load(f)
    mean_obsQ = meta['mean_obs']

    with open('%s/dataset/%s/meta.bin' % (PREFIX, 'MsPacmanNoFrameskip-v4'), 'rb') as f:
        meta = pickle.load(f)
    mean_obsPac = meta['mean_obs']

    diffBox = np.mean((mean_obsB-mean_obsBox)**2)
    diffQ = np.mean((mean_obsB-mean_obsQ)**2)
    diffPac = np.mean((mean_obsB-mean_obsPac)**2)

    print(diffBox)
    print(diffQ)
    print(diffPac)

