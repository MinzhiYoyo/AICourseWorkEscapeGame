import numpy as np

STATE = tuple
ACTION = int

# 地图状态
MAP_STATE_BLANK = 0
MAP_STATE_ESCAPE = 100
MAP_STATE_WALL = 200
MAP_STATE_STRENGTH = 20
MAP_STATE_COIN_GAIN = 30
MAP_STATE_COIN_LOSS = 40

# x右为正，y下为正
# x \in [0, 2*MAP_SIZE]
# y \in [0, 2*MAP_SIZE]
actions = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
directions = np.array([[0, -1], [0, 1], [-1, 0], [1, 0], [-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=np.int32)
# 行动消耗行动力
STRENGTH_UNIT = 2
action_cost_strength = np.array(
    [STRENGTH_UNIT, STRENGTH_UNIT, STRENGTH_UNIT, STRENGTH_UNIT, STRENGTH_UNIT + 1, STRENGTH_UNIT + 1,
     STRENGTH_UNIT + 1, STRENGTH_UNIT + 1], dtype=np.int32)
action_names = (
    '    up    ', '   down   ', '   left   ', '   right   ', ' up_left  ', ' up_right ', ' down_left', 'down_right')

MAP_SIZE = 300  # 正方形地图半径，边长为：2*MAP_SIZE + 1
VISUAL_SIZE = 50  # 可视范围
START_POSITION = np.array([MAP_SIZE + 1, MAP_SIZE + 1], dtype=np.uint32)  # 初始位置

ESCAPE_DISTANCE = int(MAP_SIZE * 1.5) # 逃离点的距离
STRENGTH_START = int(ESCAPE_DISTANCE * 1.5)  # 初始行动力

# 初始金币 = 地图减益金币 < 地图金币总量
COIN_START = int((MAP_SIZE + 1) * (MAP_SIZE + 1) * 0.05)  # 初始金币，为地图金币总量的1%
MAP_SUM_STRENGTH = int((MAP_SIZE + 1) * (MAP_SIZE + 1) * 0.1)  # 地图行动力总量，不是占有的格点数
MAP_SUM_GAIN_COIN = int((MAP_SIZE + 1) * (MAP_SIZE + 1) * 0.1)  # 地图金币增益总量，不是占有的格点数
MAP_SUM_LOSS_COIN = int((MAP_SIZE + 1) * (MAP_SIZE + 1) * 0.05)  # 地图金币减少总量，不是占有的格点数
MAP_SUM_WALL = int((MAP_SIZE + 1) * (MAP_SIZE + 1) * 0.01)  # 地图墙总量，也就是占有的格点数量

POINT_COIN_GAIN = 10  # 点金币收益
POINT_COIN_LOSS = 10  # 点金币减益
POINT_STRENGTH = 10  # 点行动力

COIN_WEIGHT = 50  # 金币分数权重
STRENGTH_WEIGHT = 10  # 行动力分数权重
DISTANCE_WEIGHT = 40  # 距离分数权重
GAMEPLAY_INTRODUCTION = """
游戏玩法：
    智能体出生在坐标为(0, 0)处，智能体的目标是逃离地图，逃离点随机生成。
    智能体一开始有部分金币以及行动力，在逃离过程中，格点有如下效果：
        1. 金币加成：智能体获得金币，为[1, 10]
        2. 金币减少：智能体失去金币，为[1, 10]
        3. 行动力加成：智能体获得行动力，固定为10
        4. 不可达点：智能体无法到达
        5. 无任何效果
    智能体在游戏过程中可获得的信息有：地图最大的为 (map_size) * (map_size+1)
        1. 当前坐标
        2. 以智能体为中心，(n+1)*(n+1)的可视范围，n = 30
        3. 距离逃离点的距离 abs(x1 - x2) + abs(y1 - y2)
        4. 当前金币数：coins
        5. 当前行动力：strengths
    智能体的动作有：
        上下左右，斜着。斜着消耗1.5倍上下左右行动力，共四个方向移动
    格点的状态有：(值) 31 * 31 * 1 + 2 + 1 + 1 + 1 = 967
        1. 空地: 0
        2. 金币加成: 
        3. 金币减少: (50-max_loss, 50 , 50+max_gain)
        4. 行动力加成: ( 100-max_loss ,100, 100 + max_gain)
        5. 逃离点: 1
        6. 不可达点: -1
    智能体分数计算公式：
    0.5 0.4 0.1
        金币权重 >> 距离权重 > 行动力权重
        分数 = 标准化算法 * 金币数权重 + 标准化算法 * 行动力权重 + 标准化算法 * 距离权重
    
    到达终点与无法到达终点：
        - 到达终点：金币数翻倍
        - 未达终点：金币数清零
    
    游戏结束条件：行动力归零或者到达终点
         
        标准化算法：保证初始化为50
            1. 金币标准化算法：当前金币 / (地图增益金币 + 初始金币)
            2. 行动力标准化算法：(初始行动力 - 当前行动力 + 地图总行动力) / (初始行动力 + 地图总行动力)
            3. 距离标准化算法：(初始距离 - 当前距离) / 初始距离
"""


# 坐标范围：[0, mapsize * 2 + 1]
# 初始坐标：(mapsize+1, mapsize+1)
#


class EscapeGameEnv:
    def __init__(self):
        # 运行中的状态变量
        self.game_setting_info = ''
        self.done = None
        self.rewards = None
        self.current_position = None  # 当前位置
        self.current_strengths = None  # 当前行动力
        self.current_coins = None  # 当前金币
        self.current_map_sum = None  # 当前地图金币增益总数，地图金币减益总数，地图行动力总数
        self.distance_to_escape_point = None  # 距离逃离点的距离

        self.Once_Game_Info = None  # 游戏日志

        # 运行中的算法变量
        self.Point_Occupied_Set = set()  # 已经占用的点，存np.ndarray

        # 初始化
        self.Start_tate = self.reset()
        self.action_size = len(actions)

        print('Game Settings is :')
        print(self.game_setting_info)

    def reset(self):
        # 生成地图
        self._generate_map()

        # 初始化金币
        self.current_coins = COIN_START
        # 初始化行动力
        self.current_strengths = STRENGTH_START
        # 初始化当前分数
        self.rewards = self._get_rewards()
        # 重置游戏结束标志
        self.done = False
        # 初始化地图金币增益总数，地图金币减益总数，地图行动力总数剩余
        self.current_map_sum = [MAP_SUM_GAIN_COIN, MAP_SUM_LOSS_COIN, MAP_SUM_STRENGTH]

        # 游戏日志，先保存地图，以稀疏矩阵的形式保存
        # (width, height);(x, y, map_state);
        self.Once_Game_Info = '({},{});'.format(self.map.shape[0], self.map.shape[1])
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                if self.map[i][j] != MAP_STATE_BLANK:
                    self.Once_Game_Info += '({},{},{});'.format(i, j, self.map[i][j])
        return self._get_state()

    def step(self, action):
        info = 'GAME OVER !!!'
        if not action in actions:
            raise ValueError('action is not in actions.')

        # 判断是否结束
        if not self.done:
            s_p = self.current_position
            # 判断是否越界
            t_p = s_p + directions[action]
            # 可行性的flag
            # 没有越界
            flag = (0 <= t_p[0] < self.map.shape[0] and 0 <= t_p[1] < self.map.shape[1])
            # 没有越界的情况下，判断是否是不可达点
            flag = flag and (not self.map[t_p[0]][t_p[1]] == MAP_STATE_WALL)
            # 行动力是否够
            flag = flag and (self.current_strengths >= action_cost_strength[action])
            # 判断是否到达终点
            # 如果说明可行动
            g = ''  # 记录收益
            if flag:
                # 更新行动力
                self.current_strengths -= action_cost_strength[action]
                # 更新位置
                self.current_position = t_p
                # 判断是否到达终点
                if (self.current_position == self.escape_point_distance).all():
                    # 到达终点
                    self.current_coins *= 2
                    g = 'Arrive'
                    self.done = True
                else:
                    # 没有到达终点
                    # 判断是否是金币加成
                    if self.map[self.current_position[0]][self.current_position[1]] == MAP_STATE_COIN_GAIN:
                        # 是金币加成
                        self.current_coins += POINT_COIN_GAIN
                        self.current_map_sum[0] -= POINT_COIN_GAIN
                        g = 'GainCoin_{:<3d}'.format(POINT_COIN_GAIN)
                    # 判断是否是金币减少
                    elif self.map[self.current_position[0]][self.current_position[1]] == MAP_STATE_COIN_LOSS:
                        # 是金币加成
                        self.current_coins += POINT_COIN_LOSS
                        self.current_map_sum[1] -= POINT_COIN_LOSS
                        g = 'GainCoin_{:<3d}'.format(POINT_COIN_LOSS)
                    # 判断是否是行动力加成
                    elif self.map[self.current_position[0]][self.current_position[1]] == MAP_STATE_STRENGTH:
                        # 是行动力加成
                        self.current_strengths += POINT_STRENGTH
                        self.current_map_sum[2] -= POINT_STRENGTH
                        self.map[self.current_position[0]][self.current_position[1]] = MAP_STATE_BLANK
                        g = 'GainStrength_{}'.format(POINT_STRENGTH)
            # 行动力为0，游戏结束
            if self.current_strengths < STRENGTH_UNIT:
                self.done = True

            self.rewards = self._get_rewards()

            # 保存的info
            info = '({},{});action={};strength={:<5d};coins={:<5d};distance={:<5d};rewards={:<5.2f};{}'.format(
                self.current_position[0], self.current_position[1], action_names[action], self.current_strengths,
                self.current_coins, self.distance_to_escape_point, self.rewards, g)
            self.Once_Game_Info += info
            self.Once_Game_Info += '\n'
        return self._get_state(), self.rewards, self.done, info

    def save(self, path):
        with open(path, 'w') as f:
            f.write(self.Once_Game_Info)

    # 下面是关于AI玩游戏的一些东西
    def _get_state(self):
        # 状态包括，当前可视范围的地图状态，当前坐标，当前行动力，当前金币，当前距离
        # 可视范围可能超过地图边界，超过地图边界的全填充为不可达点

        image_state = np.full((2 * VISUAL_SIZE + 1, 2 * VISUAL_SIZE + 1), MAP_STATE_WALL, dtype=np.uint8)
        for i in range(-VISUAL_SIZE, VISUAL_SIZE + 1):
            for j in range(-VISUAL_SIZE, VISUAL_SIZE + 1):
                if 0 <= self.current_position[0] + i < self.map.shape[0] and 0 <= self.current_position[1] + j < \
                        self.map.shape[1]:
                    image_state[i + VISUAL_SIZE][j + VISUAL_SIZE] = self.map[self.current_position[0] + i][
                        self.current_position[1] + j]

        # 游戏参数状态
        info_state = np.array([self.current_position[0], self.current_position[1], self.current_strengths,
                               self.current_coins, self.distance_to_escape_point], dtype=np.int32)

        return (image_state, info_state)
        # # 将ans一维化
        # ans = ans.reshape(-1)
        # # 增加当前坐标
        # ans = np.append(ans, self.current_position)
        # # 增加当前行动力
        # ans = np.append(ans, self.current_strengths)
        # # 增加当前金币
        # ans = np.append(ans, self.current_coins)
        # # 增加当前距离
        # ans = np.append(ans, self.distance_to_escape_point)
        # return ans

    def _get_rewards(self):
        # 计算金币标准化分数
        coin_rewards = self.current_coins / (self.current_map_sum[0] - self.current_map_sum[1] + COIN_START)
        # 计算行动力标准化分数
        strengths_rewards = (STRENGTH_START - self.current_strengths + self.current_map_sum[2]) / (
                STRENGTH_START + self.current_map_sum[2])
        # 计算距离标准化分数
        c_d = EscapeGameEnv._get_two_points_distance(self.current_position, self.escape_point_distance)
        distance_rewards = (ESCAPE_DISTANCE - c_d) / ESCAPE_DISTANCE
        return coin_rewards * COIN_WEIGHT + strengths_rewards * STRENGTH_WEIGHT + distance_rewards * DISTANCE_WEIGHT

    ## 下面是关于游戏的算法与功能函数
    # 地图生成
    def _generate_map(self):
        self.Point_Occupied_Set.clear()
        # 初始化地图
        self.map = np.zeros((2 * MAP_SIZE + 1, 2 * MAP_SIZE + 1), dtype=np.uint8)
        # 把地图所有边界都加到self.Point_Occupied_Set中
        self.Point_Occupied_Set.update([(0, i) for i in range(2 * MAP_SIZE + 1)])  # 左边界
        self.Point_Occupied_Set.update([(i, 0) for i in range(2 * MAP_SIZE + 1)])  # 上边界
        self.Point_Occupied_Set.update([(2 * MAP_SIZE, i) for i in range(2 * MAP_SIZE + 1)])  # 右边界
        self.Point_Occupied_Set.update([(i, 2 * MAP_SIZE) for i in range(2 * MAP_SIZE + 1)])  # 下边界

        # 初始化智能体位置，智能体初始位置：(MAP_SIZE+1, MAP_SIZE+1)
        self.current_position = START_POSITION
        self.Point_Occupied_Set.add((self.current_position[0], self.current_position[1]))

        # 初始化逃离点
        self._generate_escape_point()
        # 初始化不可达点
        self._generate_wall()
        # 初始化地图行动力
        self._random_choice_points_from_map('strength')
        # self._generate_strength()
        # 初始化地图金币增益
        self._random_choice_points_from_map('gain_coin')
        # 初始化地图金币减益
        # self._random_choice_points_from_map('loss_coin')
        # self._generate_coin()

    @staticmethod
    def _get_two_points_distance(point1: np.ndarray, point2: np.ndarray):
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

    # 生成逃离点的算法
    def _generate_escape_point(self):
        # 生成逃离点，先随机生成距离，再在该距离上随机生成点
        while True:
            self.escape_point_distance = np.random.randint(0, self.map.shape[0], size=2)
            if EscapeGameEnv._get_two_points_distance(self.escape_point_distance, self.current_position):
                break
        self.distance_to_escape_point = EscapeGameEnv._get_two_points_distance(self.current_position,
                                                                               self.escape_point_distance)
        self.Point_Occupied_Set.add((self.escape_point_distance[0], self.escape_point_distance[1]))

    # 生成墙的算法
    def _generate_wall(self):
        # 使用微信抢红包的算法
        wall_point_num = MAP_SUM_WALL
        wall_num = np.random.randint(1, 10) - 1
        while wall_point_num > 0 or wall_num > 0:
            the_wall_start_point = np.random.randint([0, 0], [self.map.shape[0], self.map.shape[1]], dtype=np.uint32)
            # 开始点及开始点周围八个点都没有被占用
            while (the_wall_start_point[0], the_wall_start_point[1]) in self.Point_Occupied_Set or np.array(
                    [(d[0] + the_wall_start_point[0], d[1] + the_wall_start_point[1]) in self.Point_Occupied_Set for d
                     in directions]).any():
                the_wall_start_point = np.random.randint([0, 0], [self.map.shape[0], self.map.shape[1]],
                                                         dtype=np.uint32)
            # 生成墙，使用栈来保存可用点
            the_wall_point_stack = [the_wall_start_point]
            # 这堵墙的最大长度为
            the_wall_max_length = np.random.randint(1, wall_point_num // 2, dtype=np.uint32) if wall_num > 1 else wall_point_num
            the_wall_direction = directions[np.random.randint(len(directions), dtype=np.uint32)]
            while len(the_wall_point_stack) < the_wall_max_length:
                the_wall_next_point = the_wall_point_stack[-1] + the_wall_direction
                if (the_wall_next_point[0], the_wall_next_point[1]) in self.Point_Occupied_Set or np.array(
                        [(d[0] + the_wall_next_point[0], d[1] + the_wall_next_point[1]) in self.Point_Occupied_Set for d
                         in directions]).any():
                    if len(the_wall_point_stack) <= 1:
                        break
                    the_wall_point_stack.pop()
                    if len(the_wall_point_stack) <= 1:
                        break
                    the_wall_point_stack.pop()
                    break
                the_wall_point_stack.append(the_wall_next_point)
            # 生成墙
            for a_point_stack in the_wall_point_stack:
                self.map[a_point_stack[0]][a_point_stack[1]] = MAP_STATE_WALL
            self.Point_Occupied_Set.update(tuple(map(tuple, the_wall_point_stack)))
            wall_point_num -= len(the_wall_point_stack)
            wall_num -= 1

    # 定义一个在地图中随机取若干个点的函数
    def _random_choice_points_from_map(self, point_type):
        if point_type == 0:
            sum_val = MAP_SUM_GAIN_COIN
            map_flag = MAP_STATE_COIN_GAIN
            point_value = POINT_COIN_GAIN
        elif point_type == 1:
            sum_val = MAP_SUM_LOSS_COIN
            map_flag = MAP_STATE_COIN_LOSS
            point_value = POINT_COIN_LOSS
        elif point_type == 2:
            sum_val = MAP_SUM_STRENGTH
            map_flag = MAP_STATE_STRENGTH
            point_value = POINT_STRENGTH
        else:
            return
        point_num = sum_val // point_value
        while point_num > 0:
            the_point = np.random.randint([0, 0], [self.map.shape[0], self.map.shape[1]], dtype=np.uint32)
            while (the_point[0], the_point[1]) in self.Point_Occupied_Set:
                the_point = np.random.randint([0, 0], [self.map.shape[0], self.map.shape[1]], dtype=np.uint32)
            self.map[the_point[0]][the_point[1]] = map_flag
            self.Point_Occupied_Set.add((the_point[0], the_point[1]))
            point_num -= 1


if __name__ == '__main__':
    env = EscapeGameEnv()
    out = np.zeros((2 * MAP_SIZE + 1, 2 * MAP_SIZE + 1), dtype=str)
    out[env.map == MAP_STATE_BLANK] = ' '
    out[env.map == MAP_STATE_STRENGTH] = 'S'
    out[env.map == MAP_STATE_COIN_GAIN] = 'G'
    out[env.map == MAP_STATE_COIN_LOSS] = 'L'
    out[env.map == MAP_STATE_WALL] = 'X'
    out[env.map == MAP_STATE_ESCAPE] = 'E'
    out[env.current_position[0], env.current_position[1]] = 'A'
    out[env.escape_point_distance[0], env.escape_point_distance[1]] = 'B'
    # 保存到文件
    np.savetxt('map.txt', out, fmt='%s')
