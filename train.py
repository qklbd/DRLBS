import re
from util import *
from agent import HsiAgent
import numpy as np
import random
from tqdm import tqdm
from sklearn import svm
from sklearn.model_selection import train_test_split
import torch
# from tensorboardX import SummaryWriter   
# writer = SummaryWriter(path + '/log')

torch.manual_seed(42)
torch.cuda.manual_seed(42)

epsilon = 1
epsilon_decay = 0.995
num_episodes = 1000

dqn = HsiAgent(band, learning_rate, gamma, memory_size, target_replace_iter, bs)