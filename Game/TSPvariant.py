"""
游戏玩法说明：
    旅行商问题变种，在一片 map_size * map_size 的地图上，有一些城市，玩家需要从一个城市出发，经过所有的城市，最后回到出发的城市。
    但是，城市之间的距离不是直线距离，而是通过地图上的路径，玩家需要在地图上行走，而不是直线飞行。
    城市之间有一定的障碍物，玩家需要绕过障碍物，才能到达目的地。
    分数计算：到达一座城市，+100分，每走一步，-1分，最后达到终点城市，+1000分。
    rewards机制计算：
        距离得分：
        所有城市的 曼哈顿距离数值微分 与 欧几里得距离数值微分差值 的加权，作为reward
        在地图上存在一些障碍物，玩家需要绕过障碍物，才能到达目的地。

        城市得分：
        每到达一个城市，+100分
        最后到达终点城市，+1000分
        每走一步，-1分

    state: 整张图，卷积神经网络，q值，dqn算法
    action: 上下左右，dqn算法
    rewards: 曼哈顿权重 0.7 > 欧拉权重 0.3
"""
import math
import os.path
import json

"""

曼哈顿：abs(x1-x2) + abs(y1-y2)
欧几里得：(x1-x2)^2 + (y1-y2)^2

. * * * * *
* * * * * *
* * * * * *
* * * * * *
* * * * * *
* * * * * *

"""

import numpy as np

# action集合
actions = np.array([0, 1, 2, 3])
action_names = np.array(['up', 'down', 'left', 'right'])
directions = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]])
ACTION_NUM = len(actions)

# 地图状态
MAP_BLANK = 0
MAP_OBSTACLE = 1  # 后面再设置
MAP_AGENT = 2
MAP_CITY = 3
MAP_CITY_VISITED = 4
MAP_SIZE = 20

# 障碍物的格点数量
NUM_OBSTACLE = int(MAP_SIZE // 2)
# 城市的数量
NUM_CITY = int(MAP_SIZE * 0.2)
# 墙的数量
NUM_WALL = 3

# START_CITY = (0, 0)
START_POINT = np.array([0, 0])

# 距离分数权重
WEIGHT_MH = 0.8
WEIGH_E = 0.2

# 城市分数
SCORE_CITY = 100

# 起点分数
SCORE_START = SCORE_CITY // NUM_CITY

# 行动力分数
SCORE_STRENGTH = -1


class TSPvariant(object):
    def __init__(self, map_str_or_file=None):
        self.game_info = None
        self.last_map_state = None
        self.done = None
        self.map_data = None

        # 定义运行中的参数
        self.current_position = None
        self.cities = list()
        self.occupied = set()
        self.wall_point = set()
        self.strength = None

        self.start_score = None
        self.cities_score = None
        self.strength_score = None
        self.distance_score = None

        self.rewards = None

        self.map_str = None
        self.map_path = None
        state = self.reset(data_map=map_str_or_file)
        self.state_size = len(state)
        self.action_size = len(actions)

    def reset(self, data_map=None, output_map_file=None, reset_map=True):
        self.done = False
        self.occupied.clear()
        self.occupied.add((START_POINT[0], START_POINT[1]))
        self.current_position = np.array([0, 0])  # 当前位置
        self.strength = 0
        self.distance_score = 0
        self.strength_score = self.strength * SCORE_STRENGTH
        self.cities_score = 0
        self.start_score = 0
        self.last_map_state = None

        self.game_info = None
        self.map_str = None
        self.map_path = None

        # 将所有城市置 False
        if self.cities:
            for i, city in enumerate(self.cities):
                city[2] = False
        if reset_map:
            self.wall_point.clear()
            self.cities.clear()
            self._generate_map(data_map)
        else:
            self.map_data[self.current_position[0], self.current_position[1]] = MAP_BLANK
            for city in self.cities:
                self.map_data[city[0], city[1]] = MAP_CITY
        self.map_data[START_POINT[0], START_POINT[1]] = MAP_AGENT
        self._save_map(output_map_file)
        self.game_info = self.map_str + '\n'
        return self._state()

    # 游戏结束情况
    def step(self, action):
        if action not in actions:
            raise ValueError('Invalid action')
        if not self.done:
            s_p = self.current_position.copy()
            t_p = self.current_position + directions[action]

            if ((t_p[0], t_p[1]) in self.wall_point) or t_p[0] < 0 or t_p[0] >= self.map_data.shape[0] or t_p[1] < 0 or \
                    t_p[1] >= self.map_data.shape[1]:
                # 该步移动无效
                # print('Invalid action')
                pass
            else:
                self.current_position = t_p
                self.map_data[s_p[0], s_p[
                    1]] = self.last_map_state if self.last_map_state is not None and not self.last_map_state == MAP_AGENT else MAP_BLANK
                self.map_data[self.current_position[0], self.current_position[1]] = MAP_AGENT
                self.strength += 1
                for i, city in enumerate(self.cities):
                    if city[0] == t_p[0] and city[1] == t_p[1]:
                        self.map_data[t_p[0], t_p[1]] = MAP_CITY_VISITED
                        self.cities[i][2] = True
                if self.current_position[0] == START_POINT[0] and self.current_position[1] == START_POINT[
                    1] and self.strength > 1:
                    self.done = True
                self.last_map_state = self.map_data[self.current_position[0], self.current_position[1]]
        self.rewards = self._rewards()
        if self.rewards < 0 or self.strength > MAP_SIZE * MAP_SIZE:
            self.done = True
        info = '{},{},action={},score={:.2f},distance={:.2f},strength={:.2f},cities={:.2f},start={:.2f},done={}'.format(
            self.current_position[0], self.current_position[1],action, self.rewards, self.distance_score, self.strength_score,
            self.cities_score, self.start_score, self.done
        )
        self.game_info += info
        self.game_info += '\n'
        return self._state(), self.rewards, self.done, info

    def _state(self):
        # 返回map即可
        ans: np.ndarray = self.map_data.copy()
        # 将ans 一维化
        ans = ans.reshape(-1)
        # 将strength添加
        ans = np.append(ans, self.strength)
        ans = ans.astype(np.float32)
        return ans

    def _rewards(self):
        """
        思路：遍历城市之前，城市distance分数影响最大；遍历城市之后，起始点分数影响最大。
        分数构成：距离分数+行动力分数+城市分数+起始点分数的加权和
        权重为：距离 > 起始点 >> 行动力

        s是地图大小，曼d哈顿与欧式距离，需要分别经过一个函数之后，用加权求和得到的 -> 距离分数

           strength  行动力分数：行动力 * -0.5
           city 城市分数：每到达一个城市 +50
           distance 当前位置与所有城市的距离分数之和：
           start 当前位置与起始点的距离 * 已经遍历城市的数量
        :return:
        """
        # 遍历城市
        self.distance_score = 0
        self.strength_score = self.strength * SCORE_STRENGTH
        self.cities_score = 0
        self.start_score = 0

        visited_cities = 0
        remain_cities = 0
        dis_to_cities = []
        for city in self.cities:
            if city[2]:
                self.cities_score += SCORE_CITY
                visited_cities += 1
            else:
                remain_cities += 1
                md = TSPvariant._y_manhattan(TSPvariant._get_distance_manhattan(self.current_position, city))
                ed = TSPvariant._y_euclidean(TSPvariant._get_distance_euclidean(self.current_position, city))
                dis_to_cities.append((md, ed))
        if remain_cities > 0:
            # 降序排序
            dis_to_cities.sort(key=lambda x: WEIGHT_MH * x[0] + WEIGH_E * x[1], reverse=True)
            # 复制最后一项visited_cities次
            dis_to_cities += [dis_to_cities[-1]] * visited_cities
            weight = 0.5
            index = 0
            while index < remain_cities:
                self.distance_score += weight * (
                        WEIGHT_MH * dis_to_cities[index][0] + WEIGH_E * dis_to_cities[index][1])
                index += 1
                weight /= 2
        s_md = TSPvariant._y_manhattan(TSPvariant._get_distance_manhattan(self.current_position, START_POINT))
        s_ed = TSPvariant._y_euclidean(TSPvariant._get_distance_euclidean(self.current_position, START_POINT))
        self.start_score = (WEIGHT_MH * s_md + WEIGH_E * s_ed) * visited_cities
        self.cities_score = SCORE_CITY * visited_cities
        DISTANCE_WEIGHT = 100
        STRENGTH_WEIGHT = 1
        CITIES = 1
        START = 50
        return DISTANCE_WEIGHT * self.distance_score + STRENGTH_WEIGHT * self.strength_score + CITIES * self.cities_score + START * self.start_score

    def _generate_map(self, data_map=None):

        if data_map is not None:
            self._load_map(data_map)
            return

        # 创新生成地图
        self.map_data = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.int64)
        # 把边界加入occupied中
        for i in range(MAP_SIZE):
            self.occupied.add((i, 0))
            self.occupied.add((i, MAP_SIZE - 1))
            self.occupied.add((0, i))
            self.occupied.add((MAP_SIZE - 1, i))

        # 生成障碍
        self._generate_obstacle()

        # 生成城市
        self._generate_cities()

    def _generate_cities(self):
        self.cities = list()

        for i in range(NUM_CITY):
            city = np.random.randint([0, 0], [MAP_SIZE, MAP_SIZE])
            while (city[0], city[1]) in self.occupied:
                city = np.random.randint([0, 0], [MAP_SIZE, MAP_SIZE])
            self.cities.append([city[0], city[1], False])
            self.occupied.add((city[0], city[1]))
            self.map_data[city[0], city[1]] = MAP_CITY

    # 生成障碍
    def _generate_obstacle(self):
        wall_point_num = NUM_OBSTACLE  # 墙点的数量
        wall_num = NUM_WALL  # 墙的数量
        self.wall_point = set()

        while wall_point_num > 0:
            the_wall_start_point = np.random.randint([0, 0], [MAP_SIZE, MAP_SIZE])
            if (the_wall_start_point[0], the_wall_start_point[1]) in self.occupied or np.array([
                (d[0] + the_wall_start_point[0], d[1] + the_wall_start_point[1]) in self.occupied
                for d in directions
            ]).any():
                continue
            the_wall_max_length = np.random.randint(1,
                                                    wall_point_num // 2) if wall_num > 0 and wall_point_num > 2 else wall_point_num
            the_wall_direction = directions[np.random.randint(0, ACTION_NUM)]
            the_wall_point_stack = [the_wall_start_point]
            while len(the_wall_point_stack) < the_wall_max_length:
                the_wall_next_point = the_wall_point_stack[-1] + the_wall_direction
                if (the_wall_next_point[0], the_wall_next_point[1]) in self.occupied or np.array([
                    (d[0] + the_wall_next_point[0], d[1] + the_wall_next_point[1]) in self.occupied
                    for d in directions
                ]).any():
                    break
                the_wall_point_stack.append(the_wall_next_point)

            # 更新临时墙到总墙
            self.occupied.update([(p[0], p[1]) for p in the_wall_point_stack])
            self.wall_point.update([(d[0], d[1]) for d in the_wall_point_stack])
            for p in the_wall_point_stack:
                self.map_data[p[0], p[1]] = MAP_OBSTACLE
            wall_point_num -= len(the_wall_point_stack)
            wall_num -= 1
        return

    def _save_map(self, file_path):
        # 将map以三元组的形式保存成str
        data = {
            'map_size': int(MAP_SIZE),
            'cities': [],
            'obstacles': [],
        }
        for city in self.cities:
            data['cities'].append([int(city[0]), int(city[1])])
        for obstacle in self.wall_point:
            data['obstacles'].append([int(obstacle[0]), int(obstacle[1])])
        # 以json的形式存储
        if file_path:
            json.dump(data, open(file_path, 'w'))
        self.map_str = json.dumps(data)

    def _load_map(self, map_data: str):
        # 把json字符串解析
        # map['map_size']
        # 将map转成json
        if os.path.exists(map_data):
            map_data = json.load(open(map_data, 'r'))
        else:
            map_data = json.loads(map_data)
        self.map_data = np.zeros((map_data['map_size'], map_data['map_size']), dtype=np.int64)
        self.cities = list()  # map_data['cities']

        for city in map_data['cities']:
            self.cities.append([city[0], city[1], False])
            self.map_data[city[0], city[1]] = MAP_CITY
        self.wall_point = set()
        for (i, obstacle) in enumerate(map_data['obstacles']):
            self.wall_point.add((obstacle[0], obstacle[1]))
            self.map_data[obstacle[0]][obstacle[1]] = MAP_OBSTACLE

    @staticmethod
    def _y_manhattan(x):
        return 1. / math.sqrt((x + 0.01))

    @staticmethod
    def _y_euclidean(x):
        return 1. / math.sqrt((x + 0.01))

    @staticmethod
    def _get_distance_manhattan(pos1, pos2):
        # 计算曼哈顿距离
        return abs(pos1[0] - pos2[0]) + abs(pos2[0] - pos2[1])

    @staticmethod
    def _get_distance_euclidean(pos1, pos2):
        # 计算欧几里得距离
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos2[0] - pos2[1]) ** 2)

    def save(self, best_log_file):
        with open(best_log_file, 'w') as f:
            f.write(self.game_info)

    @staticmethod
    def replay_log(log_file):
        """
        :param log_file:
        :return: map_str and action list
        """

        def get_action(last_pos, content_str: str):
            s = content_str.split(',')
            if len(s) < 2:
                return -100, last_pos
            pos = (int(s[0]), int(s[1]))
            diff_pos = (pos[0] - last_pos[0], pos[1] - last_pos[1])
            for i, direction in enumerate(directions):
                if diff_pos[0] == direction[0] and diff_pos[1] == direction[1]:
                    return i, pos
            return -1, pos

        with open(log_file, 'r') as f:
            content = f.read()
            content_list = content.split('\n')
            tmp_ = content_list[0].split('}')
            if len(tmp_) >= 1:
                map_str = tmp_[0] + '}'
            else:
                map_str = content_list[0]
            if len(tmp_) >= 2:
                content_list = tmp_[1:] + content_list[1:]
            else:
                content_list = content_list[1:]
            action_list = []
            pos = START_POINT
            for c in content_list:
                a, pos = get_action(pos, c)
                action_list.append(a)
        return map_str, action_list


if __name__ == '__main__':
    env = TSPvariant()
    env.reset(output_map_file='./test.json')


