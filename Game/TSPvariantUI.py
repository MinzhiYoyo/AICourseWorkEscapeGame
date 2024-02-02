import pygame
from Game import TSPvariant
import numpy as np

unit_size = 10
interval_size = 10
info_height = 70

COLOR_BACKGROUND = (100, 100, 100)
COLOR_CITY = (0, 255, 0)
COLOR_AGENT = (0, 0, 255)
COLOR_OBSTACLE = (255, 0, 0)
COLOR_LINE = (0, 0, 0)
COLOR_BLANK = (255, 255, 255)
COLOR_CITY_VISITED = (0, 100, 0)


class TSPvariantUI(object):
    def __init__(self, env: TSPvariant.TSPvariant):
        self.env = env

        self.reset()

        self.body_size = (self.env.map_data.shape[0] * unit_size, self.env.map_data.shape[1] * unit_size)
        self.body_pos = (interval_size, interval_size)
        self.info_size = (self.body_size[0], info_height)
        self.info_pos = (interval_size, self.body_pos[1] + self.body_size[1] + interval_size)
        self.screen_size = (self.body_size[0] + 2 * interval_size, self.body_size[1] + 2 * interval_size + info_height)

        pygame.init()
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption('TSPvariant')
        self.render(mode='agent')

    def reset(self, mode='agent'):
        pass
        # self.render(mode=mode)

    def render(self, mode='agent'):
        self.screen.fill(COLOR_BACKGROUND)
        # 绘制信息栏
        self._draw_info()
        self._draw_body()
        self._draw_grid()
        pygame.display.flip()
        return self._input(mode)

    def _input(self, mode='agent'):
        if mode == 'agent':
            pass
        elif mode == 'human':
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                return -1
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    return TSPvariant.actions[0]
                elif event.key == pygame.K_DOWN:
                    return TSPvariant.actions[1]
                elif event.key == pygame.K_LEFT:
                    return TSPvariant.actions[2]
                elif event.key == pygame.K_RIGHT:
                    return TSPvariant.actions[3]
                elif event.key == pygame.K_r:
                    return 100
        elif mode == 'replay':
            event = pygame.event.wait(1)
            # 如果是空格或者是回车均可
            if event.type == pygame.KEYDOWN and (event.key == pygame.K_SPACE or event.key == pygame.K_RETURN):
                return 'space'
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                return 'reset'
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                return 'quit'
            elif event.type == pygame.QUIT:
                return 'quit'
        return None
    def _draw_info(self):
        info1 = '({}, {}), steps = {}'.format(self.env.current_position[0], self.env.current_position[1], self.env.strength)
        info_2 = 'scores: {:.3f} \ndistance={:.2f}, strength={:.2f}\nstart={:.2f}, cities={:.2f}'.format(
           self.env.rewards if self.env.rewards else 0, self.env.distance_score, self.env.strength_score, self.env.start_score, self.env.cities_score)
        info2 = info_2.split('\n')[0]
        info3 = info_2.split('\n')[1]
        info4 = info_2.split('\n')[2]
        font = pygame.font.Font(None, 16)
        text1 = font.render(info1, True, (0, 0, 0))
        text2 = font.render(info2, True, (0, 0, 0))
        text3 = font.render(info3, True, (0, 0, 0))
        text4 = font.render(info4, True, (0, 0, 0))
        self.screen.blit(text1, (self.info_pos[0], self.info_pos[1] + 10))
        self.screen.blit(text2, (self.info_pos[0], self.info_pos[1] + 20))
        self.screen.blit(text3, (self.info_pos[0], self.info_pos[1] + 30))
        self.screen.blit(text4, (self.info_pos[0], self.info_pos[1] + 40))

    def _draw_grid(self):
        for i in range(self.env.map_data.shape[0] + 1):
            pygame.draw.line(self.screen, COLOR_LINE, (self.body_pos[0], self.body_pos[1] + i * unit_size),
                             (self.body_pos[0] + self.body_size[0], self.body_pos[1] + i * unit_size))
        for j in range(self.env.map_data.shape[1] + 1):
            pygame.draw.line(self.screen, COLOR_LINE, (self.body_pos[0] + j * unit_size, self.body_pos[1]),
                             (self.body_pos[0] + j * unit_size, self.body_pos[1] + self.body_size[1]))

    def _draw_body(self):
        # 通过self.map_data绘制地图
        for i in range(self.env.map_data.shape[0]):
            for j in range(self.env.map_data.shape[1]):
                if self.env.map_data[i, j] == TSPvariant.MAP_CITY:
                    pygame.draw.rect(self.screen, COLOR_CITY, (
                        self.body_pos[0] + i * unit_size, self.body_pos[1] + j * unit_size, unit_size, unit_size))
                elif self.env.map_data[i, j] == TSPvariant.MAP_AGENT:
                    pygame.draw.rect(self.screen, COLOR_AGENT, (
                        self.body_pos[0] + i * unit_size, self.body_pos[1] + j * unit_size, unit_size, unit_size))
                elif self.env.map_data[i, j] == TSPvariant.MAP_OBSTACLE:
                    pygame.draw.rect(self.screen, COLOR_OBSTACLE, (
                        self.body_pos[0] + i * unit_size, self.body_pos[1] + j * unit_size, unit_size, unit_size))
                elif self.env.map_data[i, j] == TSPvariant.MAP_CITY_VISITED:
                    pygame.draw.rect(self.screen, COLOR_CITY_VISITED, (
                        self.body_pos[0] + i * unit_size, self.body_pos[1] + j * unit_size, unit_size, unit_size))
                else:
                    pygame.draw.rect(self.screen, COLOR_BLANK, (
                        self.body_pos[0] + i * unit_size, self.body_pos[1] + j * unit_size, unit_size, unit_size))


    def __del__(self):
        pygame.quit()
