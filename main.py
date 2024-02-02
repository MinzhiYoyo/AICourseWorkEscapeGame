import time

import Game.TSPvariant as tspEnv
import Game.TSPvariantUI as tspUI

from DQN.dqn_train import dqn_train_model


def human_play():
    env = tspEnv.TSPvariant()
    ui = tspUI.TSPvariantUI(env)
    env.reset(output_map_file='human_play.json')
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
                state, reward, done, info = env.step(action)
                print(info)
            else:
                print('reset game!')
                env.reset()
                ui.reset()
                info = None
    print('Game quit')
    time.sleep(2)


def dqn_train():
    remark = 'train map 1'
    game_map_file_name = 'train_map_1.json'
    dqn_train_model(remark=remark, game_map_file_name=game_map_file_name)


def main():
    dqn_train()
    # human_play()
    pass


if __name__ == '__main__':
    main()
