import numpy as np
import pygame
import Game.EscapeGameEnv as egame


# 整个板块分为两个部分，上面是游戏主体，下面是信息栏

class EscapeGameUI:
    def __init__(self, env: egame.EscapeGameEnv,map=None, unit_size=5, unit_num=egame.VISUAL_SIZE*2+1, info_high=50, interval=20):
        self.env = env
        self.map = map
        self.unit_size = unit_size
        self.unit_num = unit_num
        self.info_high = info_high
        self.interval = interval


        # 游戏主体位置与大小
        self.body_pos = (self.interval, self.interval)
        self.body_size = (self.unit_size * self.unit_num, self.unit_size * self.unit_num)
        # 信息栏位置与大小，显示：分数，距离，金币，行动力
        self.info_pos = (self.interval, self.interval + self.body_size[1])
        self.info_size = (self.body_size[0], self.info_high)
        # 整个界面大小
        self.screen_width = self.unit_size * self.unit_num + self.interval * 2
        self.screen_high = self.info_pos[1] + self.info_size[1] + self.interval
        self.screen_size = (self.screen_width, self.screen_high)
        self.background = 'Game/resource/background.jpg'

        self.info_color = (0, 0, 0)
        self.background_color = (255, 255, 255)

        pygame.init()
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption('Escape to point')
        self.reset()


    def reset(self, map=None):
        if map:
            # ({},{}); ({},{},{});({},{},{});....
            try:
                items = map.split(';')
                w = int(items[0].split(',')[0][1:])
                h = int(items[0].split(',')[1][:-1])
                self.map = np.full((w, h), egame.MAP_STATE_BLANK)
                for item in items[1:]:
                    x = int(item.split(',')[0][1:])
                    y = int(item.split(',')[1])
                    state = int(item.split(',')[2][:-1])
                    self.map[x][y] = state
            except Exception as e:
                print('Parse Map String Error!')
                print(e)
        else:
            self.map = self.env.map
        # 设置背景颜色
        self.screen.fill(self.background_color)
        self.render()



    def render(self, info: str = None):
        if info:
            # ({},{});action={};strength={:<5d};coins={:<5d};distance={:<5d};rewards={:<5.2f};{}
            try:
                info_list = info.split(';')
                pos = (int(info_list[0].split(',')[0][1:]), int(info_list[0].split(',')[1][:-1]))
                action = info_list[1].split('=')[1]
                strength = int(info_list[2].split('=')[1])
                coins = int(info_list[3].split('=')[1])
                distance = int(info_list[4].split('=')[1])
                rewards = float(info_list[5].split('=')[1])
                remark = info_list[6]
            except Exception as e:
                print('Parse Info String Error!')
                print(e)
                raise e

            self._draw_information_bar(action=action, strength=strength, coins=coins,distance=distance,rewards=rewards,pos=pos)
        self._draw_information_bar(action='None', strength=0, coins=0, distance=0, rewards=0, pos=(0, 0))
        # 刷新显示
        pygame.display.flip()

    def _draw_information_bar(self, action, strength, coins, distance, rewards, pos):
        info = '{}, strength={:<5d}, coins={:<5d}, distance={:<5d}, rewards={:3.2f}, pos=({},{})'.format(
            action, strength, coins, distance, rewards, pos[0], pos[1]
        )
        # 将文字写上去
        font = pygame.font.Font(None, 30)
        text = font.render(info, True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.centerx = self.screen.get_rect().centerx
        text_rect.y = self.info_pos[1] + self.info_size[1] // 2
        self.screen.blit(text, text_rect)

    def _draw_body(self, pos):
        # 加载背景图片
        background = pygame.image.load(self.background)



        # 绘制游戏主体
        for i in range(self.unit_num):
            for j in range(self.unit_num):
                x = i + pos[0]
                y = j + pos[1]
                if x < 0 or x >= self.map.shape[0] or y < 0 or y >= self.map.shape[1]:
                    color = (0, 0, 0)
                elif self.map[x][y] == egame.MAP_STATE_BLANK:
                    color = (255, 255, 255)
                elif self.map[x][y] == egame.MAP_STATE_WALL:
                    color = (0, 0, 0)
                elif self.map[x][y] == egame.MAP_STATE_STRENGTH:
                    color = (255, 255, 0)
                elif self.map[x][y] == egame.MAP_STATE_ESCAPE:
                    color = (0, 255, 0)
                elif egame.MAP_STATE_COIN_GAIN < self.map[x][y] <= egame.MAP_STATE_COIN_GAIN + egame.GAIN_COIN_MAX:
                    color = (0, 0, 255)
                elif egame.MAP_STATE_COIN_LOSS < self.map[x][y] <= egame.MAP_STATE_COIN_LOSS + egame.LOSE_COIN_MAX:
                    color = (255, 0, 0)
                else:
                    color = (255, 255, 255)
                pygame.draw.rect(self.screen, color,
                                 (self.body_pos[0] + i * self.unit_size, self.body_pos[1] + j * self.unit_size,
                                  self.unit_size, self.unit_size))


        # 绘制人物
        pygame.draw.rect(self.screen, (255, 0, 255),
                         (self.body_pos[0] + pos[0] * self.unit_size, self.body_pos[1] + pos[1] * self.unit_size,
                          self.unit_size, self.unit_size))

    # 析构函数
    def __del__(self):
        pygame.quit()
