import os.path
import sqlite3

"""
需要保存的内容有：
    1. 一步游戏的日志 -> 一盘游戏的日志 -> 若干盘游戏的日志
        日志表：游戏日志保存表头：游戏盘的id（用来标识一盘游戏），坐标（包含两个整数），动作（字符串），strength（整数），coins（整数），rewards（浮点数），distance（整数），done（布尔值），info（字符串）
        地图表：地图保存表头：游戏盘的id（用来标识一盘游戏），横坐标（整数），纵坐标（整数），地图状态（uint8）
    2. 整个实验的日志表
        实验日志保存表头：实验的id（用来标识一次实验），最佳盘的id（用来标识一盘游戏），最佳盘的得分（浮点数），最佳盘的coins（整数），最佳盘的strength（整数），模型字典路径（字符串），模型路径（字符串），日志路径（字符串），模型保存路径（字符串），最佳模型字典路径（字符串），最佳日志路径（字符串），训练次数（整数），保存间隔（整数），备注（字符串），游戏设置信息（字符串）
    3. 训练中的loss值，用于画图
"""

class DataSave:
    def __init__(self, file_path):
        # 检测数据库是否存在
        self.conn = sqlite3.connect('DataBase/DataBase.db')
        self.cur = self.conn.cursor()
        # 检测数据库是否有表
        self.cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        table_list = self.cur.fetchall()
        if ('Data',) not in table_list:
            self.cur.execute('CREATE TABLE Data (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, data TEXT)')
            self.conn.commit()

    def save(self, name, data):
        self.cur.execute("INSERT INTO Data (name, data) VALUES ('{}', '{}')".format(name, data))
        self.conn.commit()

