import time

import Game.TSPvariant as tspEnv
import Game.TSPvariantUI as tspUI
from DQN.dqn import ReplayMemory
from DQN.dqn_train import dqn_train_model
from Function.function import create_dir


def human_play():
    create_dir('./model/memory/')
    create_dir('./map/')
    memory_replay = ReplayMemory(1000)
    env = tspEnv.TSPvariant()
    ui = tspUI.TSPvariantUI(env)
    state = env.reset(output_map_file='./map/train_map_1.json')
    # env.reset(data_map='human_play.json')
    ui.reset()
    while True:
        action = ui.render(mode='human')
        if action is not None:
            if action < 0:
                print('Game over!')
                break
            elif action in tspEnv.actions:
                print('action = {}'.format(action))
                print(action, tspEnv.action_names[action])
                next_state, reward, done, info = env.step(action)
                memory_replay.push(state, action, reward, next_state)
                state = next_state
                print(info)
            else:
                print('reset game!')
                env.reset(reset_map=False)
                ui.reset()
                ui.render(mode='agent')
                info = None
    print('save memory, length = {}'.format(len(memory_replay)))
    memory_replay.save('./model/memory/memory_human_1.json')
    print('Game quit')
    time.sleep(2)


def dqn_train():
    remark = 'train map 1'
    game_map_file_name = 'train_map_1.json'
    model_dict_path = 'model/experiment_3/model_dict_2024_02_02_22_19_23_best_193.pth'
    for i in range(10):
        model_dict_path = dqn_train_model(model_dict_path=model_dict_path,
                    memory_path='./model/memory/memory_human_1.json', remark=remark,
                    game_map_file_name=game_map_file_name)


def replay(log_file):
    map_str, actions = tspEnv.TSPvariant.replay_log(log_file)
    env = tspEnv.TSPvariant(map_str_or_file=map_str)
    ui = tspUI.TSPvariantUI(env)
    ui.reset()
    index = 0
    while True:
        key_input = ui.render(mode='replay')
        if key_input == 'space':
            action = actions[index]
            index += 1
            if action is not None:
                if action in tspEnv.actions:
                    print('action = {}'.format(action))
                    print(action, tspEnv.action_names[action])
                    state, reward, done, info = env.step(action)
                    print(info)
                    time.sleep(1)
                elif action == -1:
                    # print('no action')
                    pass
                elif action == -100:
                    print('over')
        elif key_input == 'reset':
            env.reset(reset_map=False)
            ui.reset(mode='replay')
            index = 0
        elif key_input == 'quit':
            break
        else:
            action = actions.pop(0)
            if action is not None:
                if action in tspEnv.actions:
                    print('action = {}'.format(action))
                    print(action, tspEnv.action_names[action])
                    state, reward, done, info = env.step(action)
                    print(info)
                    time.sleep(0.2)
                elif action == -1:
                    # print('no action')
                    pass
                elif action == -100:
                    print('over')
        time.sleep(0.1)
    print('QUIT')
    time.sleep(2)


def main():
    dqn_train()
    # human_play()



if __name__ == '__main__':
    main()
