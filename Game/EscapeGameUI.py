import pygame
import Game.EscapeGameEnv as egame

# 整个板块分为两个部分，上面是游戏主体，下面是信息栏

class EscapeGameUI:
    def __init__(self, env: egame.EscapeGameEnv, unit_size=5, unit_num=1000//5, info_high=50, interval=20):
        self.env = env
        self.unit_size = unit_size
        self.unit_num = unit_num
        self.info_high = info_high
        self.interval = interval

        self.screen_width = self.unit_size * self.unit_num + self.interval * 2

        # 游戏主体位置与大小
        self.body_pos = (self.interval, self.interval)
        self.body_size = (self.unit_size * self.unit_num, self.unit_size * self.unit_num)
        # 信息栏位置与大小
        self.info_pos = (self.interval, self.interval + self.body_size[1])
        self.info_size = (self.body_size[0], self.info_high)
        # 整个界面大小
        self.screen_size = (self.screen_width, self.info_pos[1] + self.info_size[1] + self.interval)

