import numpy as np

STATE = tuple
ACTION = int

# 地图状态
MAP_STATE_BLANK = 0
MAP_STATE_ESCAPE_POINT = 1
MAP_STATE_UNREACHABLE_POINT = 2
MAP_STATE_STRENGTH = 3
MAP_STATE_COIN_GAIN_MIN = 10  # 金币增益最小值 范围：[11, 10 + GAIN_COIN_MAX]
MAP_STATE_COIN_LOSS_MIN = 20  # 金币减少最小值 范围：[21, 20 + LOSS_COIN_MAX]

# x右为正，y下为正
# x \in [0, 2*MAP_SIZE]
# y \in [0, 2*MAP_SIZE]
actions = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
directions = np.array([[0, -1], [0, 1], [-1, 0], [1, 0], [-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=np.int32)
# 行动消耗行动力
STRENGTH_UNIT = 2
action_cost_strength = np.array([STRENGTH_UNIT, STRENGTH_UNIT, STRENGTH_UNIT, STRENGTH_UNIT, STRENGTH_UNIT+1, STRENGTH_UNIT+1, STRENGTH_UNIT+1, STRENGTH_UNIT+1], dtype=np.int32)
action_names = (
    '    up    ', '   down   ', '   left   ', '   right   ', ' up_left  ', ' up_right ', ' down_left', 'down_right')

MAP_SIZE = 300  # 正方形地图半径，边长为：2*MAP_SIZE + 1
VISUAL_SIZE = 30  # 可视范围
START_POSITION = np.array([MAP_SIZE + 1, MAP_SIZE + 1], dtype=np.uint32)  # 初始位置
COIN_WEIGHT = 70  # 金币分数权重
STRENGTH_WEIGHT = 10  # 行动力分数权重
DISTANCE_WEIGHT = 20  # 距离分数权重
ESCAPE_DISTANCE_MAX = MAP_SIZE * 3  # 逃离点最大距离
ESCAPE_DISTANCE_MIN = MAP_SIZE  # 逃离点最小距离
STRENGTH_START = int(ESCAPE_DISTANCE_MAX * 1.3)  # 初始行动力
COIN_START = int((MAP_SIZE + 1) * (MAP_SIZE + 1) * 0.01)  # 初始金币

STRENGTH_MAP_SUM = int((MAP_SIZE + 1) * (MAP_SIZE + 1) * 0.1)  # 地图行动力总量，不是占有的格点数
GAIN_COIN_MAP_SUM = int((MAP_SIZE + 1) * (MAP_SIZE + 1) * 0.1)  # 地图金币增益总量，不是占有的格点数
LOSS_COIN_MAP_SUM = int((MAP_SIZE + 1) * (MAP_SIZE + 1) * 0.1)  # 地图金币减少总量，不是占有的格点数

GAIN_COIN_MAX = 10  # 金币增益最大值
LOSS_COIN_MAX = 10  # 金币减少最大值
POINT_STRENGTH = 10  # 点行动力

WALL_POINT_NUM = 2 * MAP_SIZE + 1

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
    0.8 0.1 0.1
        金币权重 >> 距离权重 > 行动力权重
        分数 = 标准化算法 * 金币数权重 + 标准化算法 * 行动力权重 + 标准化算法 * 距离权重
    
    到达终点与无法到达终点：
        - 到达终点：金币数翻倍
        - 未达终点：金币数清零
    
    游戏结束条件：行动力归零或者到达终点
         
        标准化算法：
            1. 金币标准化算法：(当前 - (初始金币数 - 地图减益金币数)) / (地图增益金币数 + 地图减益金币数)
            2. 行动力标准化算法：1 - (初始行动力 - 当前行动力 + 地图总行动力) / (初始行动力 + 地图总行动力)
            3. 距离标准化算法：(初始距离 - 当前距离) / 初始距离
"""


# 坐标范围：[0, mapsize * 2 + 1]
# 初始坐标：(mapsize+1, mapsize+1)
#


class EscapeGameEnv:
    def __init__(self):
        # 运行中的状态变量
        self.game_setting_info = ('map_size={},visual_size={},coin_wight={},strength_weight={},distance_weight={},'
                                  'escape_distance_max={},escape_distance_min={},strength_start={},coin_start={},'
                                  'strength_map_sum={},gain_coin_map_sum={},loss_coin_map_sum={},gain_coin_max={},'
                                  'loss_coin_max={},point_strength={},wall_point_num={}').format(
            MAP_SIZE, VISUAL_SIZE, COIN_WEIGHT, STRENGTH_WEIGHT, DISTANCE_WEIGHT, ESCAPE_DISTANCE_MAX,
            ESCAPE_DISTANCE_MIN,
            STRENGTH_START, COIN_START, STRENGTH_MAP_SUM, GAIN_COIN_MAP_SUM, LOSS_COIN_MAP_SUM, GAIN_COIN_MAX,
            LOSS_COIN_MAX, POINT_STRENGTH, WALL_POINT_NUM)
        self.current_strength = None
        self.done = None
        self.rewards = None
        self.current_position = None  # 当前位置
        self.current_strengths = None  # 当前行动力
        self.current_coins = None  # 当前金币
        self.map_gain_coin_sum = 0  # 地图增益金币总数
        self.map_loss_coin_sum = 0  # 地图减益金币总数
        self.map_strength_sum = 0  # 地图行动力总数
        self.distance_to_escape_point = None  # 距离逃离点的距离

        self.Once_Game_Info = None # 游戏日志

        # 运行中的算法变量
        self.Point_Occupied_Set = set()  # 已经占用的点，存np.ndarray

        # 初始化
        current_state = self.reset()
        self.state_size = current_state.shape[0]
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

        # 游戏日志，先保存地图，以稀疏矩阵的形式保存
        # (width, height);(x, y, map_state);
        self.Once_Game_Info = '({},{})\n'.format(self.map.shape[0], self.map.shape[1])
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
            flag = flag and (not self.map[t_p[0]][t_p[1]] == MAP_STATE_UNREACHABLE_POINT)
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
                    if MAP_STATE_COIN_GAIN_MIN < self.map[self.current_position[0]][
                        self.current_position[1]] <= MAP_STATE_COIN_GAIN_MIN + GAIN_COIN_MAX:
                        # 是金币加成
                        self.current_coins += self.map[self.current_position[0]][
                                                  self.current_position[1]] - MAP_STATE_COIN_GAIN_MIN
                        self.map_gain_coin_sum -= (self.map[self.current_position[0]][
                                                       self.current_position[1]] - MAP_STATE_COIN_GAIN_MIN)
                        self.map[self.current_position[0]][self.current_position[1]] = MAP_STATE_BLANK
                        g = 'GainCoin_{:<3d}'.format(
                            self.map[self.current_position[0]][self.current_position[1]] - MAP_STATE_COIN_GAIN_MIN)
                    # 判断是否是金币减少
                    elif MAP_STATE_COIN_LOSS_MIN < self.map[self.current_position[0]][
                        self.current_position[1]] <= MAP_STATE_COIN_LOSS_MIN + LOSS_COIN_MAX:
                        # 是金币减少
                        self.current_coins -= self.map[self.current_position[0]][
                                                  self.current_position[1]] - MAP_STATE_COIN_LOSS_MIN
                        self.map_loss_coin_sum -= (self.map[self.current_position[0]][
                                                       self.current_position[1]] - MAP_STATE_COIN_LOSS_MIN)
                        self.map[self.current_position[0]][self.current_position[1]] = MAP_STATE_BLANK
                        g = 'LossCoin_{:<3d}'.format(
                            self.map[self.current_position[0]][self.current_position[1]] - MAP_STATE_COIN_LOSS_MIN)
                    # 判断是否是行动力加成
                    elif self.map[self.current_position[0]][self.current_position[1]] == MAP_STATE_STRENGTH:
                        # 是行动力加成
                        self.current_strengths += POINT_STRENGTH
                        self.map_strength_sum -= POINT_STRENGTH
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
        ans = self.map[
              self.current_position[0] - VISUAL_SIZE:self.current_position[0] + VISUAL_SIZE + 1,
              self.current_position[1] - VISUAL_SIZE:self.current_position[1] + VISUAL_SIZE + 1
              ].copy()

        # 将ans一维化
        ans = ans.reshape(-1)
        # 增加当前坐标
        ans = np.append(ans, self.current_position)
        # 增加当前行动力
        ans = np.append(ans, self.current_strengths)
        # 增加当前金币
        ans = np.append(ans, self.current_coins)
        # 增加当前距离
        ans = np.append(ans, self.distance_to_escape_point)
        return ans

    def _get_rewards(self):
        # 计算金币标准化分数
        coin_rewards = (self.current_coins - (COIN_START - self.map_loss_coin_sum)) / (
                self.map_gain_coin_sum + self.map_loss_coin_sum)
        # 计算行动力标准化分数
        strengths_rewards = 1 - (STRENGTH_START - self.current_strengths + self.map_strength_sum) / (
                STRENGTH_START + self.map_strength_sum)
        # 计算距离标准化分数
        s_d = EscapeGameEnv._get_two_points_distance(START_POSITION, self.escape_point_distance)
        c_d = EscapeGameEnv._get_two_points_distance(self.current_position, self.escape_point_distance)
        distance_rewards = (s_d - c_d) / s_d
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
        self._random_choice_points_from_map('loss_coin')
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
        wall_point_num = np.random.randint(WALL_POINT_NUM // 2, WALL_POINT_NUM + 1, dtype=np.uint32)
        wall_num = np.random.randint(1, 10) - 1
        while wall_point_num > 0 and wall_num > 0:
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
            the_wall_max_length = np.random.randint(1, wall_point_num // 2, dtype=np.uint32)
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
            pass
            # 生成墙
            for a_point_stack in the_wall_point_stack:
                self.map[a_point_stack[0]][a_point_stack[1]] = MAP_STATE_UNREACHABLE_POINT
            self.Point_Occupied_Set.update(tuple(map(tuple, the_wall_point_stack)))
            wall_point_num -= len(the_wall_point_stack)
            wall_num -= 1

    def _generate_strength(self):
        strength_sum = np.random.randint(STRENGTH_MAP_SUM // 2, STRENGTH_MAP_SUM)
        strength_point_num = strength_sum // MAP_STATE_STRENGTH
        while strength_point_num > 0:
            the_strength_point = np.random.randint([0, 0], [self.map.shape[0], self.map.shape[1]], dtype=np.uint32)
            while (the_strength_point[0], the_strength_point[1]) in self.Point_Occupied_Set:
                the_strength_point = np.random.randint([0, 0], [self.map.shape[0], self.map.shape[1]], dtype=np.uint32)
            self.map[the_strength_point[0]][the_strength_point[1]] = MAP_STATE_STRENGTH
            self.map_strength_sum += POINT_STRENGTH
            self.Point_Occupied_Set.add((the_strength_point[0], the_strength_point[1]))
            strength_point_num -= 1

    def _generate_coin(self):
        # 金币增益
        gain_coin_sum = np.random.randint(GAIN_COIN_MAP_SUM // 2, GAIN_COIN_MAP_SUM)
        gain_coin_point_num = gain_coin_sum // MAP_STATE_COIN_GAIN_MIN
        while gain_coin_point_num > 0:
            the_gain_coin_point = np.random.randint([0, 0], [self.map.shape[0], self.map.shape[1]], dtype=np.uint32)
            while (the_gain_coin_point[0], the_gain_coin_point[1]) in self.Point_Occupied_Set:
                the_gain_coin_point = np.random.randint([0, 0], [self.map.shape[0], self.map.shape[1]], dtype=np.uint32)
            the_gain_coin_is_ = np.random.randint(1, GAIN_COIN_MAX + 1) + MAP_STATE_COIN_GAIN_MIN
            self.map[the_gain_coin_point[0]][the_gain_coin_point[1]] = the_gain_coin_is_
            self.map_gain_coin_sum += the_gain_coin_is_
            self.Point_Occupied_Set.add((the_gain_coin_point[0], the_gain_coin_point[1]))
            gain_coin_point_num -= 1
        # 金币减少
        loss_coin_sum = np.random.randint(LOSS_COIN_MAP_SUM // 2, LOSS_COIN_MAP_SUM)
        loss_coin_point_num = loss_coin_sum // MAP_STATE_COIN_LOSS_MIN
        while loss_coin_point_num > 0:
            the_loss_coin_point = np.random.randint([0, 0], [self.map.shape[0], self.map.shape[1]], dtype=np.uint32)
            while (the_loss_coin_point[0], the_loss_coin_point[1]) in self.Point_Occupied_Set:
                the_loss_coin_point = np.random.randint([0, 0], [self.map.shape[0], self.map.shape[1]], dtype=np.uint32)
            the_loss_coin_is_ = np.random.randint(1, LOSS_COIN_MAX + 1) + MAP_STATE_COIN_LOSS_MIN
            self.map[the_loss_coin_point[0]][the_loss_coin_point[1]] = the_loss_coin_is_
            self.map_loss_coin_sum += the_loss_coin_is_
            self.Point_Occupied_Set.add((the_loss_coin_point[0], the_loss_coin_point[1]))
            loss_coin_point_num -= 1

    # 定义一个在地图中随机取若干个点的函数
    def _random_choice_points_from_map(self, type):
        # data = {
        #     'gain_coin': {'sum': GAIN_COIN_MAP_SUM, 'min': MAP_STATE_COIN_GAIN_MIN, 'max': GAIN_COIN_MAX},
        #     'loss_coin': {'sum': LOSS_COIN_MAP_SUM, 'min': MAP_STATE_COIN_LOSS_MIN, 'max': LOSS_COIN_MAX},
        #     'strength': {'sum': STRENGTH_MAP_SUM, 'min': MAP_STATE_STRENGTH}
        # }
        if type == 'gain_coin':
            sum_val = GAIN_COIN_MAP_SUM
            min_val = MAP_STATE_COIN_GAIN_MIN
            max_val = GAIN_COIN_MAX
        elif type == 'loss_coin':
            sum_val = LOSS_COIN_MAP_SUM
            min_val = MAP_STATE_COIN_LOSS_MIN
            max_val = LOSS_COIN_MAX
        elif type == 'strength':
            sum_val = STRENGTH_MAP_SUM
            min_val = MAP_STATE_STRENGTH
        else:
            return
        sum_ = np.random.randint(sum_val // 2, sum_val)
        point_num = sum_ // min_val
        while point_num > 0:
            the_point = np.random.randint([0, 0], [self.map.shape[0], self.map.shape[1]], dtype=np.uint32)
            while (the_point[0], the_point[1]) in self.Point_Occupied_Set:
                the_point = np.random.randint([0, 0], [self.map.shape[0], self.map.shape[1]], dtype=np.uint32)
            if type == 'gain_coin':
                the_point_is_ = np.random.randint(1, max_val + 1) + min_val
                self.map[the_point[0]][the_point[1]] = the_point_is_
                self.map_gain_coin_sum += the_point_is_
            elif type == 'loss_coin':
                the_point_is_ = np.random.randint(1, max_val + 1) + min_val
                self.map[the_point[0]][the_point[1]] = the_point_is_
                self.map_loss_coin_sum += the_point_is_
            elif type == 'strength':
                self.map[the_point[0]][the_point[1]] = MAP_STATE_STRENGTH
                self.map_strength_sum += POINT_STRENGTH
            self.Point_Occupied_Set.add((the_point[0], the_point[1]))
            point_num -= 1


if __name__ == '__main__':
    env = EscapeGameEnv()
    out = np.zeros((2 * MAP_SIZE + 1, 2 * MAP_SIZE + 1), dtype=str)
    out[env.map == MAP_STATE_BLANK] = ' '
    out[env.map == MAP_STATE_STRENGTH] = 'S'
    for i in range(MAP_STATE_COIN_GAIN_MIN, MAP_STATE_COIN_GAIN_MIN + GAIN_COIN_MAX):
        out[env.map == i] = 'G'
    for i in range(MAP_STATE_COIN_LOSS_MIN, MAP_STATE_COIN_LOSS_MIN + LOSS_COIN_MAX):
        out[env.map == i] = 'L'
    # out[env.map >= MAP_STATE_COIN_GAIN_MIN and env.map <= (MAP_STATE_COIN_GAIN_MIN + GAIN_COIN_MAX)] = 'G'
    # out[env.map >= MAP_STATE_COIN_LOSS_MIN and env.map <= (MAP_STATE_COIN_LOSS_MIN + LOSS_COIN_MAX)] = 'L'
    out[env.map == MAP_STATE_UNREACHABLE_POINT] = 'X'
    out[env.map == MAP_STATE_ESCAPE_POINT] = 'E'
    out[env.current_position[0], env.current_position[1]] = 'A'
    out[env.escape_point_distance[0], env.escape_point_distance[1]] = 'B'
    # 保存到文件
    np.savetxt('map.txt', out, fmt='%s')
