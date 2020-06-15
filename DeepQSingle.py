import datetime
import itertools
import random
from collections import deque
import sys
import argparse

import numpy as np
import paho.mqtt.client as mqtt
from keras.layers import *
from keras.models import Sequential

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile

from matplotlib import pyplot as plt

from variable import *
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
np.random.seed(1)
DEFAULT_SIZE_ROW = 10
DEFAULT_SIZE_COLUMN = 10
MAX_SIZE = 10
GRID_SIZE = 30
PADDING = 5

obstacles_loc = []

obstacles_loc_1 = [12, 15, 16, 17, 27, 37, 30, 42, 43,
                 44, 45, 57, 58, 61, 68, 71, 72, 76, 84, 85, 88, 91]

obstacles_loc_2 = [20, 21, 22, 23, 24, 25, 49,
                 50, 55, 56, 57, 58, 59, 79, 80, 81, 82]

obstacles_loc_3 = [20, 21, 22, 23, 24, 25, 49,
                 50, 55, 56, 57, 58, 59, 79, 80, 81, 82]

obstacles_loc_4 = [12,29,30,31,32,33,37,38,39,44,51,57,65,70,71,72,73,79]

obstacles_loc_6 = [22, 25, 26, 27, 47, 67, 60, 82, 83, 84, 85, 107, 108, 121, 128, 141, 142, 146, 164, 165, 168, 181, 
                   32, 35, 36, 37, 57, 77, 70, 92, 93, 94, 95, 117, 118, 131, 138, 151, 152, 156, 174, 175, 178, 191,
                   222, 225,226,227,247,267,260,282,283,285,307,308,321,328,341,342,346,365,368,381,
                   232,235,236,237,257,277,270,292,293,294,295,317,318,331,338,351,352,356,374,375,378,391]

obstacles_loc_7 = [22, 25, 26, 31, 41, 70, 71, 85, 90, 131, 132, 142, 167, 170, 177, 
                   32, 35, 36, 37, 57, 77, 70, 92, 93, 94, 95, 117, 118, 131, 138, 151, 152, 156, 174, 179, 190,
                   211, 230, 247,267,260,310,311,312,326,329,340,343,356,357,378,371,
                   212,215,226,224,248,261,271,282,286,298,301,308,311,316,351,332,366,369,384,385,318,391]


canvas_list = []
storage_value = []

NOT_USE = -1.0
OBSTACLE = 0.0
EMPTY = 1.0
TARGET = 0.75
START = 0.5

#Run params
START_STATE = 1
MAX_EPISODES = 60
EMBTOGGLE = 2
DIMENSION = 20
TARGET_LOC = 399
EMBEDPATH = "./Embeddings/"
RESULTPATH = "./Results/LM2/"
VALSPATH = "./Results/Vals/"
STPSPATH = "./Results/STP2/"
REWPATH = "./Results/RW2/"
GRID = 20

row_num = GRID
col_num = GRID


DEBUG = False
EPSILON_REDUCE = True
RANDOM_MODE = False

ACTION_LEFT = 0
ACTION_UP = 1
ACTION_RIGHT = 2
ACTION_DOWN = 3

STATE_START = 'start'
STATE_WIN = 'win'
STATE_LOSE = 'lose'
STATE_BLOCKED = 'blocked'
STATE_VALID = 'valid'
STATE_INVALID = 'invalid'

# Hyperparameter
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
MEMORY_LEN = 1000
DISCOUNT_RATE = 0.95
BATCH_SIZE = 50



#Save params
state_index = -1
rew_arr = []
rewardAxis = np.zeros((10,60))
stepsAxis = np.zeros((10,60))
partRew = [[] for i in range(10)]
globalTotSteps = 0


class DQNAgent:
    def __init__(self, env):
        self.state_size = env.observation_space
        self.action_size = env.action_size
        self.memory = deque(maxlen=MEMORY_LEN)
        self.gamma = DISCOUNT_RATE  # discount rate
        self.num_actions = 4
        if EPSILON_REDUCE:
            self.epsilon = EPSILON  # exploration rate
            self.epsilon_min = EPSILON_MIN
            self.epsilon_decay = EPSILON_DECAY
        else:
            self.epsilon = EPSILON_MIN
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()

        if(EMBTOGGLE == 2):
            model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1),
                 activation='relu',input_shape= (GRID, GRID,1)))
            model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
            model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
            model.add(Flatten())
            model.add(Dense(50, activation='relu'))
            model.add(Dense(4, activation='tanh'))
            model.compile(loss='mse',
              optimizer='adam')
            return model

        elif(EMBTOGGLE == 0):
            model.add(Dense(64, input_shape=(
                self.state_size,), activation='relu'))
        else:
            model.add(Dense(64, input_shape=(DIMENSION,), activation='tanh'))
        model.add(Dense(32, activation='tanh'))
        model.add(Dense(self.action_size))
        model.compile(loss='mse',
                      optimizer='adam')
        return model

    def remember(self, current_state, action, reward, next_state, game_over):
        self.memory.append(
            (current_state, action, reward, next_state, game_over))

    def replay(self, batch_size):
        #print("doing")
        memory_size = len(self.memory)
        batch_size = min(memory_size, batch_size)
        minibatch = random.sample(self.memory, batch_size)
        if(EMBTOGGLE == 0):
            inputs = np.zeros((batch_size, self.state_size))
        elif(EMBTOGGLE == 2):
            inputs = np.zeros((batch_size, GRID, GRID, 1))
        else:
            inputs = np.zeros((batch_size, DIMENSION)) 
        #inputs = np.zeros((batch_size, 2*DIMENSION))
        targets = np.zeros((batch_size, self.num_actions))
        i = 0
        for state, action, reward, next_state, done in minibatch:
            
            target = reward
            if not done:
                target = reward + self.gamma * \
                    np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            inputs[i] = state
            targets[i] = target_f
            i += 1

        self.model.fit(inputs, targets, epochs=8,
                       batch_size=16, verbose=0)
        if EPSILON_REDUCE and (self.epsilon > self.epsilon_min):
            self.epsilon *= self.epsilon_decay
        return self.model.evaluate(inputs, targets, verbose=0)

    def predict(self, current_state):
        predict = self.model.predict(current_state)[0]
        sort = np.argsort(predict)[-len(predict):]
        sort = np.flipud(sort)
        return sort[0]

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


class Environment:
    def __init__(self, row_x, col_y, args):
        self.row_number = row_x
        self.col_number = col_y
        self.action_size = 4  # 1:UP,2:RIGHT,3:LEFT,4:RIGHT
        self.observation_space = row_x * col_y
        self._map = self._create_map(row_x, col_y)
        self.ready = False
        self.adj_list = []
        self.adj_dir = []
        self.num_states = row_x * col_y + 1
        self.dimension = DIMENSION
        self.cs = 0
        emb_path = EMBEDPATH + str(args.embedpath)
        self.model1 = KeyedVectors.load_word2vec_format(
            emb_path, binary=False)
        # self.model2 = KeyedVectors.load_word2vec_format(
        #     './Embeddings/maze42_dst_30.txt', binary=False)

    def _create_map(self, row_number, col_number):
        map = np.ones(shape=(row_number, col_number))
        return map

    def set_target(self, row_x, col_y):
        self.target = (row_x, col_y)
        self._map[row_x, col_y] = TARGET

    def set_collision(self, row_x, col_y):
        self._map[row_x, col_y] = OBSTACLE

    def set_start_point(self, row_x, col_y):
        self.start = (row_x, col_y)
        self.current_state = (row_x, col_y, STATE_START)
        self._map[row_x, col_y] = START
        self.cs = row_x * self.col_number + col_y

    def set_empty_point(self, row_x, col_y):
        self._map[row_x, col_y] = EMPTY

    def create_random_environment(self):
        self.ready = True
        self._map = self._create_map(self.row_number, self.col_number)
        n = min(self.row_number, self.col_number) + 1
        count = 0
        random_set = np.empty(shape=(n, 2))
        while count < n:
            x = np.random.randint(self.row_number)
            y = np.random.randint(self.col_number)
            if ([x, y] in random_set.tolist()):
                continue
            random_set[count, 0] = x
            random_set[count, 1] = y
            count += 1
        self.set_start_point(int(random_set[0, 0]), int(random_set[0, 1]))
        self.set_target(int(random_set[1, 0]), int(random_set[1, 1]))
        for i in range(2, n):
            self.set_collision(int(random_set[i, 0]), int(random_set[i, 1]))

    def reset(self):
        if RANDOM_MODE:
            self.create_random_environment()
        row_x, col_y = self.start
        self.current_state = (row_x, col_y, STATE_START)
        self.cs = row_x * self.col_number + col_y
        self.visited = set()
        self.min_reward = -5
        self.free_cells = [(r, c) for r in range(self.row_number) for c in range(self.col_number) if
                           self._map[r, c] == 1.0]
        
        self.total_reward = 0
        self.map = np.copy(self._map)
        for row, col in itertools.product(range(self.row_number), range(self.col_number)):
            storage_value[row, col] = self._map[row, col]
        

    def valid_actions(self, cell=None):
        if cell is None:
            row, col, mode = self.current_state
        else:
            row, col = cell
        
        state = self.col_number*row + col
        return self.adj_dir[state]

    def update_state(self, action):
        nrow, ncol, nmode = current_row, current_col, mode = self.current_state

        if self.map[current_row, current_col] > 0.0:
            self.visited.add((current_row, current_col))

        valid_actions = self.valid_actions()
        #print(nrow,ncol)
        if not valid_actions:
            nmode = STATE_BLOCKED
        elif action in valid_actions:
            nmode = STATE_VALID
            storage_value[nrow, ncol] = EMPTY
            if action == ACTION_LEFT:
                storage_value[nrow, ncol - 1] = START
                self.cs -= 1
                ncol -= 1
            elif action == ACTION_UP:
                storage_value[nrow - 1, ncol] = START
                self.cs -= self.col_number
                nrow -= 1
            if action == ACTION_RIGHT:
                storage_value[nrow, ncol + 1] = START
                self.cs += 1
                ncol += 1
            elif action == ACTION_DOWN:
                storage_value[nrow + 1, ncol] = START
                self.cs += self.col_number
                nrow += 1
        else:
            # invalid action
            nmode = STATE_INVALID
        
        if(self.cs in obstacles_loc):
            nmode = STATE_BLOCKED
        
   
        self.current_state = (nrow, ncol, nmode)
        self.cs = nrow*self.col_number + ncol


    # Action define:
    # 0: LEFT
    # 1: UP
    # 2: RIGHT
    # 3: DOWN

    def act(self, act):
        self.update_state(act) 
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        current_state = self.observe()
        return current_state, reward, status

    def observe(self):
        canvas = self.set_env()
        envstate = canvas.reshape((1, -1))
        return envstate

    def set_env(self):
        canvas = np.copy(self.map)
        nrows, ncols = self.map.shape
        
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r, c] > 0.0:
                    canvas[r, c] = 1.0
        row, col, valid = self.current_state
        canvas[row, col] = 0.5
        return canvas

    def game_status(self):
        if self.total_reward < self.min_reward:
            return STATE_LOSE
        current_row, current_col, mode = self.current_state
        target_row, target_col = self.target
        if current_row == target_row and current_col == target_col:
            return STATE_WIN
        return STATE_VALID


    def get_reward(self):
        current_row, current_col, mode = self.current_state
        target_row, target_col = self.target
        if current_row == target_row and current_col == target_col:
            return 1.
        if mode == STATE_BLOCKED:
            return -0.01
        if mode == STATE_INVALID:
            return -0.1
        if (current_row, current_col) in self.visited:
            return -0.01
        if mode == STATE_VALID:
            return -0.01

    def generate_embedding(self):
        state = self.cs
        embed = np.zeros(self.dimension)
        embed[:self.dimension] = self.model1[str(state)]
        #embed[self.dimension:] = self.model2[str(state)]
        return embed

    def generate_embeddings_custom(self,state):
        temp = self.cs
        state = state
        embed = np.zeros(self.dimension)
        embed[:self.dimension] = self.model1[str(state)]
        self.cs = temp
        return embed


def deepQLearning(model, env, state, args, randomMode=False, **opt):
    global  state_index, rewardAxis, state_index,globalTotSteps
    episodes = opt.get('n_epoch', MAX_EPISODES)
    #print(obstacles_loc)
    print(args)

    batch_size = opt.get('batch_size', BATCH_SIZE)
    
    win_history = []
    memory = []

    win_rate = 0.0

    history_size = env.observation_space

    cum_reward = 0

    totStps = 0
    globalTotSteps = 0
    totRew = 0
    for episode in range(episodes):

        #print("\nEpisode: ",episode)
        loss = 0.0
        env.reset()
        game_over = False

        # number of step for each episode
        n_step = 0
        list_action = []
        next_state = env.map.reshape((1, -1))
        print("Here")
        #while end state is not reached or cumulative reward doesn't reach minimum
        while not game_over:
            print(env.cs)
            valid_actions = env.valid_actions()
            if not valid_actions:
                game_over = True
                #print(env.map)
                continue

            current_state = next_state

            #Embedding for current state
            cs = (env.generate_embedding()).reshape((1, -1))

            # Get best action from current state
            if np.random.rand() < model.epsilon:
                action = random.choice(valid_actions)
            else:
                if(EMBTOGGLE == 0):
                    action = model.predict(current_state)
                elif EMBTOGGLE == 2:
                    current_state = np.reshape(current_state,(1,GRID,GRID,))
                    current_state = np.expand_dims(current_state,-1)
                    action = model.predict(current_state)
                else:
                    action = model.predict(cs)


            # Apply action, get reward and new envstate
            next_state, reward, game_status = env.act(action)
            totRew += reward
            ns = (env.generate_embedding()).reshape((1, -1))

            #print(env.cs,reward,end=' -> ')

            #print(env.cs,end=' ')
            if game_status == STATE_WIN:
                x, y, _ = env.current_state
                storage_value[x, y] = TARGET
                win_history.append(1)
                game_over = True
            elif game_status == STATE_LOSE:
                x, y, _ = env.current_state
                storage_value[x, y] = EMPTY
                win_history.append(0)
                game_over = True
            else:
                game_over = False

            if(game_over == True):
                print(env.total_reward)
                rewardAxis[state_index][episode] = env.total_reward
                stepsAxis[state_index][episode] = totStps
                #rew_arr.append(env.total_reward)

            if DEBUG:
                print("--------------------------------------")
                print(np.reshape(current_state, newshape=(4, 4)))
                print("action = {},valid_action = {},reward = {}, game_over = {}".format(action, valid_actions,
                                                                                         reward, game_over))
                #print(np.reshape(next_state, newshape=(4, 4)))
            list_action.append(action)
            #Store episode (experience)
            if(EMBTOGGLE == 1):
                model.remember(cs, action, reward, ns, game_over)
            elif(EMBTOGGLE == 2):
                current_state = np.reshape(current_state, (1,GRID,GRID,))
                next_state = np.reshape(next_state, (1,GRID,GRID,))
                current_state = np.expand_dims(current_state,-1)
                next_state = np.expand_dims(next_state,-1)
                
                model.remember(current_state, action,
                               reward, next_state, game_over)
            else:
                model.remember(current_state, action,
                               reward, next_state, game_over)
            n_step += 1
            totStps += 1

            if(totStps %50 == 0):
                partRew[state_index].append(totRew)

            loss = model.replay(batch_size)

        cum_reward += env.total_reward

        if not EPSILON_REDUCE:
            if win_rate > 0.9:
                model.epsilon = 0.05
        if len(win_history) > history_size:
            win_rate = sum(win_history[-history_size:]) / history_size

        if sum(win_history[-history_size:]) == history_size:
            env.reset()
            saved_env = env

            print("Reached 100%% win rate at episode: %d" % (episode,))
            memory.sort(key=len)
            memory = np.array(memory)
            break



def getRowCol(obs):
    return (obs//GRID, obs % GRID)


def create_environment(start_row, start_col, args):

    global obstacles_loc
    
    for obstacle in obstacles_loc:
        (row, col) = getRowCol(obstacle)
        row = obstacle//GRID
        col = obstacle % GRID
        storage_value[row, col] = OBSTACLE

    storage_value[start_row, start_col] = START
    TRow = TARGET_LOC//GRID
    TCol = TARGET_LOC%GRID
    storage_value[TRow, TCol] = TARGET

    row_num = GRID
    col_num = GRID
    env = Environment(row_num, col_num,args)

    for row, col in itertools.product(range(row_num), range(col_num)):
        if storage_value[row, col] == START:
            env.set_start_point(row, col)
        elif storage_value[row, col] == TARGET:
            env.set_target(row, col)
        elif storage_value[row, col] == OBSTACLE:
            env.set_collision(row, col)

    num_states = env.observation_space

    for i in range(GRID*GRID):
        env.adj_list.append([])
        env.adj_dir.append([])
    
    # for state in range(GRID*GRID):
    #     if state in obstacles_loc:
    #         continue
    #     if(state%GRID != 0):
    #         env.adj_list[state].append(state-1)
    #     if((state+1)%GRID != 0):
    #         env.adj_list[state].append(state+1)
    #     if(state>=GRID):
    #         env.adj_list[state].append(state-GRID)
    #     if(state+GRID<=GRID*GRID):
    #         env.adj_list[state].append(state+GRID)

    #print(env.adj_list[0]) 
    with open(args.edgelist) as f:
        for line in f:
            line = line.rstrip().split(' ')
            if(int(line[1]) not in env.adj_list[int(line[0])]):
                env.adj_list[int(line[0])].append(int(line[1]))
            if(int(line[0]) not in env.adj_list[int(line[1])]):
                env.adj_list[int(line[1])].append(int(line[0]))

    #print(env.adj_list[0])   


    for state in range(GRID*GRID):
        for next_state in env.adj_list[state]:
            if(next_state == state - 1):
                env.adj_dir[state].append(0)
            elif(next_state == state-GRID):
                env.adj_dir[state].append(1)
            elif(next_state == state + 1):
                env.adj_dir[state].append(2)
            else:
                env.adj_dir[state].append(3)

    # for i in range(400):
    #     print(i,end=' ')
    #     for j in range(len(env.adj_list[i])):
    #         print(env.adj_dir[i][j],end=' ')
    #     print()

    # for i in range(200,400):     
    #     for j in range(len(env.adj_list[i])):
    #         print(i,end=' ')
    #         print(env.adj_list[i][j])
        
    
    #print(len(env.adj_list))
    return env

def printEdgelist(args):
    f = open(args.edgelist,'w')
    for row in range(row_num):    
        for col in range(col_num):
            vertexList = []
            dirList = []
            state = col_num*row + col
            if(state in obstacles_loc):     
                continue
            
            
            if((state%row_num != 0)):
                f.write('{} {}\n'.format(state,state-1))
            if(((state+1)%row_num != 0) ):
                f.write('{} {}\n'.format(state,state+1))
            if((state > row_num) ):
                f.write('{} {}\n'.format(state,state-row_num))
            if((state+row_num < GRID*GRID)):
                f.write('{} {}\n'.format(state,state+row_num))



def trainDQN(args):

    #update state, valid actions, set collision
    start_row = 0
    start_col = 0

    env = create_environment(start_row, start_col,args)
    if env is None:
        return
    global state_index,partRew
    state_index = -1

    for _ in range(int(args.iterations)):
        state = 5
        
        if state in obstacles_loc or state == TARGET_LOC:
            continue
        state_index += 1
        model = DQNAgent(env)
        if state in obstacles_loc:
            continue
        row, col = getRowCol(state)
        env.set_start_point(row, col)
        env.reset()
        deepQLearning(model, env, state, args)

        if(_ == int(args.iterations) -1 ):
            
            partRew = np.array(partRew)
            res_path = RESULTPATH + str(args.savepath)
            vals_path = VALSPATH + str(args.savepath)
            steps_path = STPSPATH + str(args.savepath) 
            partrew_path = REWPATH + str(args.savepath)
            np.save(res_path,rewardAxis)
            np.save(steps_path,stepsAxis)
            np.save(partrew_path,partRew)
            

        #vals = np.zeros((GRID*GRID,4))

        
        #if(i not in obstacles_loc):
           # pred = model.model.predict((env.generate_embeddings_custom(i)).reshape((1, -1)))
            #for j in range(4):
             #   vals[i][j] = pred[0][j]
        
        #np.save(vals_path,vals)

    pass
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-ep", "--embedpath", help="Path for embeddings")
    parser.add_argument("-sp", "--savepath", help="Save path for npy array")
    parser.add_argument("-maze", "--maze", help="Which maze to run")
    parser.add_argument("-iter", "--iterations", help="Number of iterations")
    parser.add_argument("-target", "--target", help="Location of target")
    parser.add_argument("-el", "--edgelist", help="edgelist of the maze")
    parser.add_argument("-dim","--dimension",help="Dimension")
    args = parser.parse_args()
    
    DIMENSION = int(args.dimension)

    if  (args.maze == "1"):
        obstacles_loc = obstacles_loc_1
    elif(args.maze == "2"):
        obstacles_loc = obstacles_loc_2
    elif(args.maze == "3"):
        obstacles_loc = obstacles_loc_3
    elif(args.maze == "4"):
        obstacles_loc = obstacles_loc_4
    elif(args.maze == "6"):
        obstacles_loc = obstacles_loc_6
    elif(args.maze == "7"):
        obstacles_loc = obstacles_loc_7
    
    TARGET_LOC = int(args.target)
    for row, col in itertools.product(range(GRID), range(GRID)):
        storage_value.append(NOT_USE)
    storage_value = np.array(storage_value, dtype=np.float).reshape(GRID, GRID)
    #printEdgelist(args)
    trainDQN(args)
