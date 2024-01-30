import math
import os.path
import time

import torch

from DQN.dqn import DQNAgent
from Function.function import get_time_info, log_dir, model_dir, experiment_log_file_path, create_dir, \
    experiment_save_init, format_time
from Game.EscapeGameEnv import EscapeGameEnv


def dqn_train_model(model_path=None, model_dict_path=None, memory_path=None, remark='', need_train=True,
                    memory_file_path=None, need_save=True):
    # 实验保存初始化，并获取实验次数
    experiment_num = experiment_save_init()
    experiment_log_dir = os.path.join(log_dir, 'experiment_{}'.format(experiment_num))
    experiment_model_dir = os.path.join(model_dir, 'experiment_{}'.format(experiment_num))
    loss_record_path = os.path.join(experiment_log_dir, 'loss_record_{}_{}.log'.format(experiment_num, get_time_info()))
    create_dir(experiment_log_dir)
    create_dir(experiment_model_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    episodes_num = 1000 if torch.cuda.is_available() else 500

    # 初始化游戏环境
    env = EscapeGameEnv()
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size, device=device, model_path=model_path,
                     model_dict_path=model_dict_path, memory_path=memory_path)
    # 分数最大的一次记录
    best_episode_rewards = -math.inf
    best_step = 0
    best_coin = 0
    best_strength = 0

    best_log_file = None
    best_model_dict_file = None

    console_output_interval = episodes_num // 20  # 打印间隔
    save_interval = episodes_num // 10  # 保存间隔

    loss_info = 'episode; loss\n'
    start_time = time.time()
    for episode in range(episodes_num):
        one_game_log = ''
        state = env.reset()
        while not env.done:
            action = agent.select_action(torch.tensor(state, dtype=torch.float32, device=agent.device)).item()
            next_state, reward, done, info = env.step(action)
            one_game_log += info
            one_game_log += '\n'

            agent.memory.push(state, action, reward, next_state)
            agent.train()
            if agent.loss:
                loss_info += '{}; {}\n'.format(episode, agent.loss.item())
            state = next_state
        if env.rewards > best_episode_rewards:
            best_episode_rewards = env.rewards
            best_step = episode
            best_coin = env.current_coins
            best_strength = env.current_strength
            best_log_file = os.path.join(experiment_log_dir, 'log_{}_best_{}.log'.format(get_time_info(), episode))
            best_model_dict_file = os.path.join(experiment_model_dir,
                                           'model_dict_{}_best_{}.pth'.format(get_time_info(), episode))
            env.save(best_log_file)
            agent.save_dict(best_model_dict_file)

        if episode % console_output_interval == 0:
            end_time = time.time()
            # 计算每次的平均耗时
            print('Train {} rounds needs {}. Average time is {}.'.format(console_output_interval,
                                                                         format_time(end_time - start_time),
                                                                         format_time((
                                                                                             end_time - start_time) / console_output_interval)))
            start_time = time.time()
            print('Episode: {}, rewards: {}, coins: {}, strength: {}, best rewards: {}, best coins: {}, '
                  'best strength: {}'.format(episode, env.rewards, env.current_coins, env.current_strength,
                                             best_episode_rewards, best_coin, best_strength))
            if need_save:
                env.save(os.path.join(experiment_log_dir, 'log_{}_{}.log'.format(get_time_info(), episode)))
                agent.save_dict(os.path.join(experiment_model_dir, 'model_dict_{}_{}.pth'.format(get_time_info(), episode)))
    experiment_log = ('{experiment_num}; {best_step}; {best_score}; {best_coins}; {best_strength}; {model_dict_path}; '
                      '{model_path}; {log_dir}; {model_dir}; {best_model_dict_path}; {best_log_file_path}; {'
                      'train_times}; {save_interval}; {remark}; {game_setting_info}\n').format(
        experiment_num=experiment_num,
        best_step=best_step,
        best_score=best_episode_rewards,
        best_coins=best_coin,
        best_strength=best_strength,
        model_dict_path=model_dict_path,
        model_path=model_path,
        log_dir=experiment_log_dir,
        model_dir=experiment_model_dir,
        best_model_dict_path=best_model_dict_file,
        best_log_file_path=best_log_file,
        train_times=episodes_num,
        save_interval=save_interval,
        remark=remark,
        game_setting_info=env.game_setting_info
    )
    with open(experiment_log_file_path, 'a') as f:
        f.write(experiment_log)
    with open(loss_record_path, 'w') as f:
        f.write(loss_info)


if __name__ == '__main__':
    dqn_train_model()
