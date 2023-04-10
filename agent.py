import torch
import random
import numpy as np
from collections import deque
from snakeAI import AIGame, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
from charts import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self):
        self.n_game = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def getState(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            (dir_u and game.isCollision(point_u)) or
            (dir_d and game.isCollision(point_d)) or
            (dir_l and game.isCollision(point_l)) or
            (dir_r and game.isCollision(point_r)),

            (dir_u and game.isCollision(point_r)) or
            (dir_d and game.isCollision(point_l)) or
            (dir_u and game.isCollision(point_u)) or
            (dir_d and game.isCollision(point_d)),

            (dir_u and game.isCollision(point_r)) or
            (dir_d and game.isCollision(point_l)) or
            (dir_r and game.isCollision(point_u)) or
            (dir_l and game.isCollision(point_d)),

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def longMemory(self):
        if (len(self.memory) > BATCH_SIZE):
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.trainStep(states, actions, rewards, next_states, dones)

    def shortMemory(self, state, action, reward, next_state, done):
        self.trainer.trainStep(state, action, reward, next_state, done)

    def getAction(self, state):
        self.epsilon = 80 - self.n_game
        final_move = [0, 0, 0]
        if (random.randint(0, 200) < self.epsilon):
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = AIGame()
    while True:
        state_old = agent.getState(game)

        final_move = agent.getAction(state_old)

        reward, done, score = game.playStep(final_move)
        state_new = agent.getState(game)

        agent.shortMemory(state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_game += 1
            agent.longMemory()
            record = max(record, score)
            if (score > reward):
                reward = score
                agent.model.save()
            print('Game:', agent.n_game, 'Score:', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_game
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if (__name__ == "__main__"):
    train()
