import math
import os.path
import time

import torch

from DQN.dqn import DQNAgent
from Function.function import get_time_info, log_dir, model_dir, experiment_log_file_path, create_dir, \
    experiment_save_init, format_time, game_map_dir
# from Game.EscapeGameEnv import EscapeGameEnv
from Game.TSPvariant import TSPvariant


def dqn_train_model(model_path=None, model_dict_path=None, memory_path=None, remark='',
                    memory_file_path=None, need_save=True, game_map_file_name=None):
    # 实验保存初始化，并获取实验次数
    experiment_num = experiment_save_init()
    experiment_log_dir = os.path.join(log_dir, 'experiment_{}'.format(experiment_num))
    experiment_model_dir = os.path.join(model_dir, 'experiment_{}'.format(experiment_num))
    loss_record_path = os.path.join(experiment_log_dir, 'loss_record_{}_{}.log'.format(experiment_num, get_time_info()))
    create_dir(experiment_log_dir)
    create_dir(experiment_model_dir)
    create_dir(game_map_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    episodes_num = 1000 if torch.cuda.is_available() else 500

    # 初始化游戏环境
    env = TSPvariant()
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size, device=device, model_path=model_path,
                     model_dict_path=model_dict_path, memory_path=memory_path)
    # 分数最大的一次记录
    best_episode_rewards = -math.inf  # 最好的rewards
    best_episode = 0  # 最好rewards对应的episode
    best_distance_score = 0  # 最好rewards对应的distance分数
    best_strength = 0  # 最好rewards对应的strength
    best_pos = (0, 0)  # 最好rewards对应的结束位置
    best_city_score = 0  # 最好rewards对应的city分数
    best_start_score = 0  # 最好rewards对应的start分数

    best_log_file = None
    best_model_dict_file = None

    console_output_interval = episodes_num // 20  # 打印间隔
    save_interval = episodes_num // 10  # 保存间隔

    loss_info = 'episode; loss\n'
    start_time = time.time()
    map_json = os.path.join(game_map_dir, game_map_file_name)
    if not os.path.exists(map_json):
        map_json = None
    for episode in range(episodes_num):
        one_game_log = ''
        if map_json:
            state = env.reset(data_map=map_json)
        else:
            map_json = os.path.join(game_map_dir, game_map_file_name)
            state = env.reset(output_map_file=map_json)
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
            best_episode = episode
            best_strength = env.strength
            best_distance_score = env.distance_score
            best_city_score = env.cities_score
            best_start_score = env.start_score
            best_pos = (env.current_position[0], env.current_position[1])

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
            print('Current: {}, rewards={:.4f}, strength={:^5d}, pos=({},{}), dis={:.4f}, cities={:.4f}, start={:.4f}\n'
                  'Best: {}, rewards={:.4f}, strength={:^5d}, pos=({},{}), dis={:.4f}, cities={:.4f}, start={:4f}\n'
                  .format(episode, env.rewards, env.strength, env.current_position[0], env.current_position[1],
                          env.distance_score,
                          env.cities_score, env.start_score, best_episode, best_episode_rewards, best_strength,
                          best_pos[0], best_pos[1],
                          best_distance_score, best_city_score, best_start_score))
            if need_save:
                env.save(os.path.join(experiment_log_dir, 'log_{}_{}.log'.format(get_time_info(), episode)))
                agent.save_dict(
                    os.path.join(experiment_model_dir, 'model_dict_{}_{}.pth'.format(get_time_info(), episode)))
            start_time = time.time()

    # experiment_log = 'experiment_times; best_episode; best_rewards; model_dict_path; model_path; '
    #                 'log_dir;model_dir; remark; game_map\n')
    experiment_log = '{}; {}; {}; {}; {}; {}; {}; {}; {}\n'.format(
        experiment_num, best_episode, best_episode_rewards, model_dict_path, model_path, experiment_log_dir,
        experiment_model_dir, remark, env.map_str)

    with open(experiment_log_file_path, 'a') as f:
        f.write(experiment_log)
    with open(loss_record_path, 'w') as f:
        f.write(loss_info)


if __name__ == '__main__':
    dqn_train_model()
