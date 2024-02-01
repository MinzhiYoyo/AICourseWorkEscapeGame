import time

import Game.TSPvariant as tspEnv
import Game.TSPvariantUI as tspUI

def play():
    env = tspEnv.TSPvariant()
    ui = tspUI.TSPvariantUI(env)
    env.reset()
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
        if env.done:
            break
    print('Game quit')
    time.sleep(10)
play()
