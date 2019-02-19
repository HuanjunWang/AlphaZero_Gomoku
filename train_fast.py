# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""

from __future__ import print_function
import random
import time

import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_tensorflow import PolicyValueNet
from multiprocessing import Process, Queue

CONFIG = {
    # Game configuration
    "board_width": 6,
    "board_height": 6,
    "n_in_row": 4,

    # training configuration
    "learn_rate": 2e-3,
    "lr_multiplier": 1.0,
    "temperature": 1.0,
    "buffer_size": 10000,
    "batch_size": 512,
    "epochs": 5,
    "kl_targ": 0.02,
    "max_game_num": 1500,
    "max_step_num": 100000,

    # MCTS configuration
    "n_playout": 400,
    "c_puct": 5,

    # evaluate configuration
    "check_freq": 50,
    "pure_mcts_playout_num": 1000,

    "init_model": None,

    "selfplayer_num": 5,

    "current_policy_name": "./current_policy.model",
    "best_policy_name": "./best_policy.model"
}


class SelfPlayer(Process):
    def __init__(self, config, sample_queue, model_queue):
        super(SelfPlayer, self).__init__()

        self.config = config
        self.temp = config['temperature']

        self.sample_queue = sample_queue
        self.model_queue = model_queue

        self.board = Board(width=config['board_width'],
                           height=config['board_height'],
                           n_in_row=config['n_in_row'])
        self.game = Game(self.board)

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        samples = []
        for i in range(n_games):
            _, play_data = self.game.start_self_play(self.mcts_player,
                                                     temp=self.temp)
            samples.extend(list(play_data)[:])
        return samples

    def run(self):

        self.policy_value_net = PolicyValueNet(self.config['board_width'],
                                               self.config['board_height'],
                                               model_file=self.config['init_model'])
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.config['c_puct'],
                                      n_playout=self.config['n_playout'],
                                      is_selfplay=1)

        print("running")
        while True:
            # always use the latest weight
            weights = None
            while not self.model_queue.empty():
                weights = self.model_queue.get()
            if weights:
                self.policy_value_net.set_weight(weights)

            # sample
            samples = self.collect_selfplay_data()
            # put the new sample to sample queue
            self.sample_queue.put(samples)


class Evaluator(Process):
    def __init__(self, config, weight_queue):
        super(Evaluator, self).__init__()
        self.config = config
        self.queue = weight_queue

        self.best_win_ratio = 0.0
        self.pure_mcts_playout_num = self.config['pure_mcts_playout_num']

    def run(self):
        self.policy_value_net = PolicyValueNet(self.config['board_width'],
                                               self.config['board_height'],
                                               model_file=self.config['init_model'])

        while True:
            weight = self.queue.get()
            self.policy_value_net.set_weight(weight)
            win_ratio = self.policy_evaluate()
            self.policy_value_net.save_model(self.config['current_policy_name'])

            if win_ratio > self.best_win_ratio:
                print("New best policy!!!!!!!!")
                self.best_win_ratio = win_ratio
                # update the best_policy
                self.policy_value_net.save_model(self.config['best_policy_name'])
                if (self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 10000):
                    self.pure_mcts_playout_num += 1000
                    self.best_win_ratio = 0.0

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        self.evaluate_game = Game(Board(width=self.config['board_width'],
                                        height=self.config['board_height'],
                                        n_in_row=self.config['n_in_row']))

        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.config['c_puct'],
                                         n_playout=self.config['n_playout'])

        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.config['pure_mcts_playout_num'])

        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.evaluate_game.start_play(current_mcts_player,
                                                   pure_mcts_player,
                                                   start_player=i % 2,
                                                   is_shown=0)
            win_cnt[winner] += 1

        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
            self.config['pure_mcts_playout_num'],
            win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio


class TrainPipelineFast(object):
    def __init__(self, config):
        self.config = config
        self.lr_multiplier = self.config['lr_multiplier']

        self.data_buffer = deque(maxlen=config['buffer_size'])

        # sample queue: Self player will put the samples to this queue
        # model queue: Train process will put the update model to this queue
        self.sample_queue = Queue()
        self.model_queues = []
        self.self_players = []
        self.evaluator_queue = Queue()
        self.evaluator = Evaluator(self.config, self.evaluator_queue)

        for _ in range(self.config['selfplayer_num']):
            model_queue = Queue()
            self.model_queues.append(model_queue)
            self.self_players.append(SelfPlayer(config, self.sample_queue, model_queue))

        self.policy_value_net = PolicyValueNet(config['board_width'],
                                               config['board_height'],
                                               model_file=config['init_model'])

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.config['board_height'], self.config['board_width'])), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.config['batch_size'])
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)

        for i in range(self.config['epochs']):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.config['learn_rate'] * self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                                axis=1)
                         )
            if kl > self.config['kl_targ'] * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.config['kl_targ'] * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.config['kl_targ'] / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))

        #
        # print(("kl:{:.5f},"
        #        "lr_multiplier:{:.3f},"
        #        "loss:{},"
        #        "entropy:{},"
        #        "explained_var_old:{:.3f},"
        #        "explained_var_new:{:.3f}"
        #        ).format(kl,
        #                 self.lr_multiplier,
        #                 loss,
        #                 entropy,
        #                 explained_var_old,
        #                 explained_var_new))
        return loss, entropy

    def run(self):
        # Start the self play processes
        for sp in self.self_players:
            sp.daemon = True
            sp.start()

        # start the evaluator
        self.evaluator.start()


        step_num = 0
        game_num = 0
        train_step = 0

        # put the initial weight to queue
        weights = self.policy_value_net.get_weight()
        for model_queue in self.model_queues:
            model_queue.put(weights)

        start_time = time.time()
        while True:
            try:
                while True:
                    game_sample = self.sample_queue.get(block=False)
                    game_num += 1
                    step_num += len(game_sample)
                    game_sample = self.get_equi_data(game_sample)
                    self.data_buffer.extend(game_sample)
                    print("Game:", game_num, " Stpes:", step_num, " time:", time.time() - start_time)
            except Exception:
                pass

            if len(self.data_buffer) > self.config['batch_size']:
                train_step += 1
                self.policy_update()

                weights = self.policy_value_net.get_weight()
                for model_queue in self.model_queues:
                    model_queue.put(weights)

                if game_num % self.config['check_freq'] == 0:
                    self.evaluator_queue.put(weights)

            if step_num > self.config['max_step_num']:
                break
            if game_num > self.config['max_game_num']:
                break

            time.sleep(1)


if __name__ == '__main__':
    training_pipeline = TrainPipelineFast(CONFIG)
    training_pipeline.run()
