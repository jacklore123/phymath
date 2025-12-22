"""
智能自适应学习系统 - 纯标准库版本
===================================
本系统模拟一个基于理想轨迹对齐的智能教育平台，包含：
1. 多学生状态管理
2. 理想专家轨迹对比
3. 实时摄像头监控模拟
4. 自适应学习路径规划
5. 学习策略智能推荐
6. 数据持久化和基础分析

注意：此版本仅使用Python标准库，无需安装任何额外依赖
作者: AI助手
版本: 2.2（纯标准库版本）
日期: 2024
"""

import random
import json
import datetime
import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import sqlite3
from enum import Enum
import os


# ======================
# 数据类型定义
# ======================

class LearningStrategy(Enum):
    """学习策略枚举类
    定义系统中可用的各种教学策略
    """
    LECTURE = "讲解"  # 教师讲解知识点
    EXAMPLE = "例题"  # 例题分析与练习
    REFLECTION = "反思"  # 学生反思总结
    REST = "休息"  # 休息恢复精力
    INTERACTIVE = "互动学习"  # 互动式学习
    CONTINUE = "继续学习"  # 持续深入学习
    REVIEW = "复习"  # 复习巩固知识


@dataclass
class LearningGoal:
    """学习目标数据类
    描述一个具体的学习目标及其属性
    """
    module: str  # 学习模块（如：力学、词汇）
    topic: str  # 具体知识点（如：牛顿第二定律）
    target_level: float  # 目标掌握程度（0-5）
    current_difficulty: float = 4.5  # 当前难度设置

    def to_dict(self) -> Dict:
        """将对象转换为字典格式，便于序列化"""
        return {
            "module": self.module,
            "topic": self.topic,
            "level": self.target_level,
            "difficulty": self.current_difficulty
        }


@dataclass
class StudentState:
    """学生状态数据类
    记录学生的所有学习状态信息
    """
    name: str  # 学生姓名
    age: int  # 学生年龄
    subject: str  # 当前学习科目
    module: str  # 当前学习模块
    topic: str  # 当前学习知识点
    level: float  # 当前掌握程度（0-5）
    attention: float  # 专注度（0-1）
    fatigue: float  # 疲劳度（0-1）
    learning_history: List[Dict] = None  # 学习历史记录
    last_updated: str = None  # 最后更新时间

    def __post_init__(self):
        """初始化后处理
        确保所有字段都有合理的默认值
        """
        if self.learning_history is None:
            self.learning_history = []
        if self.last_updated is None:
            self.last_updated = datetime.datetime.now().isoformat()

    def to_dict(self) -> Dict:
        """将学生状态转换为字典，便于序列化和存储"""
        return {
            "name": self.name,
            "age": self.age,
            "subject": self.subject,
            "module": self.module,
            "topic": self.topic,
            "level": self.level,
            "attention": self.attention,
            "fatigue": self.fatigue,
            "learning_history": self.learning_history,
            "last_updated": self.last_updated
        }

    def validate(self) -> Tuple[bool, str]:
        """验证学生状态的合法性

        Returns:
            Tuple[bool, str]: (验证是否通过, 错误信息或成功消息)
        """
        # 检查学习水平是否在合理范围内
        if not 0 <= self.level <= 5:
            return False, f"学习水平 {self.level} 超出范围 [0, 5]"

        # 检查专注度是否在合理范围内
        if not 0 <= self.attention <= 1:
            return False, f"专注度 {self.attention} 超出范围 [0, 1]"

        # 检查疲劳度是否在合理范围内
        if not 0 <= self.fatigue <= 1:
            return False, f"疲劳度 {self.fatigue} 超出范围 [0, 1]"

        return True, "状态验证通过"


# ======================
# 数据库管理模块
# ======================

class LearningDatabase:
    """数据库管理类
    负责所有与数据库相关的操作，包括创建表、插入、更新和查询数据
    """

    def __init__(self, db_path: str = "learning_system.db"):
        """初始化数据库连接

        Args:
            db_path: 数据库文件路径
        """
        self.conn = sqlite3.connect(db_path)
        self.create_tables()

    def create_tables(self):
        """创建数据库表结构
        包括学生表、学习目标表和学习记录表
        """
        cursor = self.conn.cursor()

        # 学生状态表 - 存储学生的基本信息和当前状态
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS students
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           name
                           TEXT
                           NOT
                           NULL,
                           age
                           INTEGER,
                           subject
                           TEXT,
                           module
                           TEXT,
                           topic
                           TEXT,
                           level
                           REAL,
                           attention
                           REAL,
                           fatigue
                           REAL,
                           learning_history
                           TEXT,
                           last_updated
                           TIMESTAMP,
                           created_at
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP
                       )
                       ''')

        # 学习目标表 - 存储各学科的学习目标和难度设置
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS learning_goals
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           subject
                           TEXT,
                           module
                           TEXT,
                           topic
                           TEXT,
                           target_level
                           REAL,
                           current_difficulty
                           REAL,
                           created_at
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP
                       )
                       ''')

        # 学习记录表 - 存储每次学习会话的详细记录
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS learning_records
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           student_id
                           INTEGER,
                           session_id
                           TEXT,
                           strategy
                           TEXT,
                           level_before
                           REAL,
                           level_after
                           REAL,
                           attention_before
                           REAL,
                           attention_after
                           REAL,
                           fatigue_before
                           REAL,
                           fatigue_after
                           REAL,
                           efficiency_score
                           REAL,
                           timestamp
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP,
                           FOREIGN
                           KEY
                       (
                           student_id
                       ) REFERENCES students
                       (
                           id
                       )
                           )
                       ''')

        self.conn.commit()
        print("✅ 数据库表创建完成")

    def save_student_state(self, student: StudentState):
        """保存学生状态到数据库

        Args:
            student: 学生状态对象
        """
        cursor = self.conn.cursor()

        # 检查学生是否已存在
        cursor.execute('SELECT id FROM students WHERE name = ?', (student.name,))
        result = cursor.fetchone()

        if result:
            # 更新现有记录
            cursor.execute('''
                           UPDATE students
                           SET age              = ?,
                               subject          = ?,
                               module           = ?,
                               topic            = ?,
                               level            = ?,
                               attention        = ?,
                               fatigue          = ?,
                               learning_history = ?,
                               last_updated     = ?
                           WHERE name = ?
                           ''', (student.age, student.subject, student.module, student.topic,
                                 student.level, student.attention, student.fatigue,
                                 json.dumps(student.learning_history),
                                 datetime.datetime.now().isoformat(),
                                 student.name))
            print(f"📝 更新学生 {student.name} 的状态到数据库")
        else:
            # 插入新记录
            cursor.execute('''
                           INSERT INTO students
                           (name, age, subject, module, topic, level, attention, fatigue, learning_history,
                            last_updated)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                           ''', (student.name, student.age, student.subject, student.module, student.topic,
                                 student.level, student.attention, student.fatigue,
                                 json.dumps(student.learning_history),
                                 datetime.datetime.now().isoformat()))
            print(f"📝 新增学生 {student.name} 到数据库")

        self.conn.commit()

    def save_learning_record(self, student_name: str, session_id: str,
                             strategy: str, level_before: float, level_after: float,
                             attention_before: float, attention_after: float,
                             fatigue_before: float, fatigue_after: float,
                             efficiency_score: float):
        """保存学习记录到数据库

        Args:
            student_name: 学生姓名
            session_id: 学习会话ID
            strategy: 使用的学习策略
            level_before: 学习前的水平
            level_after: 学习后的水平
            attention_before: 学习前的专注度
            attention_after: 学习后的专注度
            fatigue_before: 学习前的疲劳度
            fatigue_after: 学习后的疲劳度
            efficiency_score: 学习效率评分
        """
        cursor = self.conn.cursor()

        # 获取学生ID
        cursor.execute('SELECT id FROM students WHERE name = ?', (student_name,))
        result = cursor.fetchone()

        if result:
            student_id = result[0]
            cursor.execute('''
                           INSERT INTO learning_records
                           (student_id, session_id, strategy, level_before, level_after,
                            attention_before, attention_after, fatigue_before, fatigue_after,
                            efficiency_score)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                           ''', (student_id, session_id, strategy, level_before, level_after,
                                 attention_before, attention_after, fatigue_before, fatigue_after,
                                 efficiency_score))

            self.conn.commit()
            print(f"📊 保存学习记录：{student_name} - {strategy}")

    def get_student_history(self, student_name: str) -> List[Dict]:
        """获取学生的学习历史记录

        Args:
            student_name: 学生姓名

        Returns:
            List[Dict]: 学习历史记录列表
        """
        cursor = self.conn.cursor()
        cursor.execute('''
                       SELECT strategy,
                              level_before,
                              level_after,
                              attention_before,
                              attention_after,
                              fatigue_before,
                              fatigue_after,
                              efficiency_score, timestamp
                       FROM learning_records
                       WHERE student_id = (SELECT id FROM students WHERE name = ?)
                       ORDER BY timestamp
                       ''', (student_name,))

        records = []
        for row in cursor.fetchall():
            records.append({
                "strategy": row[0],
                "level_before": row[1],
                "level_after": row[2],
                "attention_before": row[3],
                "attention_after": row[4],
                "fatigue_before": row[5],
                "fatigue_after": row[6],
                "efficiency_score": row[7],
                "timestamp": row[8]
            })

        print(f"📖 获取到 {student_name} 的 {len(records)} 条学习记录")
        return records

    def close(self):
        """关闭数据库连接"""
        self.conn.close()
        print("🔒 数据库连接已关闭")


# ======================
# 数学工具函数（替代numpy）
# ======================

class MathUtils:
    """数学工具类
    提供基本的数学运算功能，替代numpy的部分功能
    """

    @staticmethod
    def mean(values: List[float]) -> float:
        """计算平均值

        Args:
            values: 数值列表

        Returns:
            float: 平均值
        """
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def min_max_normalize(values: List[float]) -> List[float]:
        """最小-最大归一化

        Args:
            values: 数值列表

        Returns:
            List[float]: 归一化后的数值列表
        """
        if not values:
            return []

        min_val = min(values)
        max_val = max(values)

        if max_val == min_val:
            return [0.5] * len(values)  # 所有值相等时返回中间值

        return [(v - min_val) / (max_val - min_val) for v in values]

    @staticmethod
    def linspace(start: float, stop: float, num: int = 50) -> List[float]:
        """生成等差数列

        Args:
            start: 起始值
            stop: 结束值
            num: 元素个数

        Returns:
            List[float]: 等差数列
        """
        if num < 2:
            return [start]

        step = (stop - start) / (num - 1)
        return [start + step * i for i in range(num)]


# ======================
# 文本可视化工具类
# ======================

class TextVisualizer:
    """文本可视化工具类
    提供基于文本和ASCII字符的图表显示功能
    """

    def __init__(self):
        """初始化可视化工具"""
        self.math_utils = MathUtils()

    def create_progress_bar(self, value: float, max_value: float = 5.0,
                            bar_length: int = 20, show_percentage: bool = True) -> str:
        """创建文本进度条

        Args:
            value: 当前值
            max_value: 最大值
            bar_length: 进度条长度（字符数）
            show_percentage: 是否显示百分比

        Returns:
            str: 进度条字符串
        """
        # 确保值在合理范围内
        normalized_value = max(0, min(value, max_value))

        # 计算填充长度
        filled_length = int(normalized_value / max_value * bar_length)
        empty_length = bar_length - filled_length

        # 选择进度条字符
        filled_char = "█"
        empty_char = "░"

        # 构建进度条
        bar = filled_char * filled_length + empty_char * empty_length

        # 添加百分比显示
        if show_percentage:
            percentage = (normalized_value / max_value) * 100
            return f"[{bar}] {normalized_value:.2f}/{max_value} ({percentage:.1f}%)"
        else:
            return f"[{bar}] {normalized_value:.2f}/{max_value}"

    def create_sparkline(self, values: List[float], height: int = 5) -> str:
        """创建Sparkline迷你图表

        Args:
            values: 数值列表
            height: 图表高度（行数）

        Returns:
            str: Sparkline图表字符串
        """
        if not values:
            return "无数据"

        # 归一化数据
        normalized = self.math_utils.min_max_normalize(values)

        # 创建字符映射
        chars = " ▁▂▃▄▅▆▇█"
        sparkline = ""

        # 为每个值选择合适的字符
        for norm_val in normalized:
            char_index = int(norm_val * (len(chars) - 1))
            sparkline += chars[char_index]

        return sparkline

    def create_bar_chart(self, data: Dict[str, float], bar_length: int = 20) -> str:
        """创建垂直柱状图

        Args:
            data: 数据字典 {标签: 值}
            bar_length: 最大柱状长度

        Returns:
            str: 柱状图字符串
        """
        if not data:
            return "无数据"

        chart_lines = []
        max_value = max(data.values()) if data.values() else 1

        for label, value in data.items():
            # 计算柱状长度
            bar_len = int(value / max_value * bar_length) if max_value > 0 else 0

            # 创建柱状
            bar = "█" * bar_len + " " * (bar_length - bar_len)

            # 添加标签和数值
            chart_lines.append(f"{label:10} |{bar}| {value:.2f}")

        return "\n".join(chart_lines)

    def create_line_chart(self, values: List[float], width: int = 50, height: int = 10) -> str:
        """创建ASCII线图

        Args:
            values: 数值列表
            width: 图表宽度
            height: 图表高度

        Returns:
            str: 线图字符串
        """
        if len(values) < 2:
            return "数据点不足"

        # 归一化数据
        normalized = self.math_utils.min_max_normalize(values)

        # 创建坐标网格
        grid = [[' ' for _ in range(width)] for _ in range(height)]

        # 在网格上绘制点
        for i, norm_val in enumerate(normalized):
            x = int(i / (len(values) - 1) * (width - 1))
            y = int((1 - norm_val) * (height - 1))  # 翻转Y轴

            if 0 <= x < width and 0 <= y < height:
                grid[y][x] = '●'

        # 添加连接线（简单版本）
        for i in range(len(values) - 1):
            x1 = int(i / (len(values) - 1) * (width - 1))
            y1 = int((1 - normalized[i]) * (height - 1))
            x2 = int((i + 1) / (len(values) - 1) * (width - 1))
            y2 = int((1 - normalized[i + 1]) * (height - 1))

            # 简单线性插值
            steps = max(abs(x2 - x1), abs(y2 - y1))
            if steps > 0:
                for s in range(steps + 1):
                    x = int(x1 + (x2 - x1) * s / steps)
                    y = int(y1 + (y2 - y1) * s / steps)
                    if 0 <= x < width and 0 <= y < height and grid[y][x] == ' ':
                        grid[y][x] = '·'

        # 构建图表字符串
        chart_lines = []
        for row in grid:
            chart_lines.append(''.join(row))

        # 添加坐标轴信息
        min_val = min(values)
        max_val = max(values)
        chart_lines.append(f"最小值: {min_val:.2f}  最大值: {max_val:.2f}")

        return "\n".join(chart_lines)

    def create_simple_table(self, data: List[Dict], headers: List[str] = None) -> str:
        """创建简单表格

        Args:
            data: 数据列表（字典列表）
            headers: 表头列表

        Returns:
            str: 表格字符串
        """
        if not data:
            return "无数据"

        # 确定列宽
        if headers:
            col_names = headers
        else:
            col_names = list(data[0].keys())

        # 计算每列的最大宽度
        col_widths = []
        for col in col_names:
            # 考虑表头长度
            max_width = len(str(col))

            # 考虑数据长度
            for row in data:
                if col in row:
                    max_width = max(max_width, len(str(row[col])))

            col_widths.append(max_width + 2)  # 添加一些边距

        # 构建表格
        table_lines = []

        # 表头
        header_line = "┌"
        for width in col_widths:
            header_line += "─" * width + "┬"
        header_line = header_line[:-1] + "┐"
        table_lines.append(header_line)

        # 表头内容
        header_content = "│"
        for i, col in enumerate(col_names):
            header_content += f" {col:<{col_widths[i] - 2}} │"
        table_lines.append(header_content)

        # 分隔线
        separator_line = "├"
        for width in col_widths:
            separator_line += "─" * width + "┼"
        separator_line = separator_line[:-1] + "┤"
        table_lines.append(separator_line)

        # 数据行
        for row in data:
            data_line = "│"
            for i, col in enumerate(col_names):
                value = row.get(col, "")
                data_line += f" {str(value):<{col_widths[i] - 2}} │"
            table_lines.append(data_line)

        # 底部边框
        bottom_line = "└"
        for width in col_widths:
            bottom_line += "─" * width + "┴"
        bottom_line = bottom_line[:-1] + "┘"
        table_lines.append(bottom_line)

        return "\n".join(table_lines)


# ======================
# 核心学习系统类
# ======================

class AdaptiveLearningSystem:
    """自适应学习系统核心类
    管理整个学习系统的运行，包括学生管理、教学策略、路径规划等
    """

    def __init__(self, use_database: bool = True):
        """初始化学习系统

        Args:
            use_database: 是否使用数据库存储数据
        """
        print("🤖 初始化智能自适应学习系统...")

        # 初始化数据库（如果需要）
        self.use_database = use_database
        if use_database:
            self.db = LearningDatabase()
        else:
            self.db = None
            print("⚠️  数据库功能已禁用")

        # 初始化数学工具
        self.math_utils = MathUtils()

        # 初始化学生状态
        self.students = self._initialize_students()

        # 理想专家轨迹（牛顿的物理学习轨迹）
        self.ideal_state = {
            "name": "牛顿",
            "subject": "物理",
            "module": "力学",
            "topic": "牛顿第二定律",
            "level": 4.5
        }

        # 学科名称映射（中文到英文）
        self.subject_mapping = {
            "物理": "physics",
            "英语": "english",
            "生物": "biology",
            "数学": "math",
            "化学": "chemistry",
            "语文": "chinese"
        }

        # 初始化学习目标
        self.learning_goals = self._initialize_learning_goals()

        # 学习策略配置
        self.strategy_weights = {
            "讲解": {"base_gain": 0.2, "fatigue_impact": 0.05},
            "例题": {"base_gain": 0.3, "fatigue_impact": 0.08},
            "反思": {"base_gain": 0.4, "fatigue_impact": -0.1},
            "休息": {"base_gain": 0, "fatigue_impact": -0.15},
            "互动学习": {"base_gain": 0.5, "fatigue_impact": 0.05},
            "继续学习": {"base_gain": 0.4, "fatigue_impact": 0.1},
            "复习": {"base_gain": 0.25, "fatigue_impact": 0.03}
        }

        # 初始化文本可视化工具
        self.viz = TextVisualizer()

        print("✅ 系统初始化完成")

    def _initialize_students(self) -> List[StudentState]:
        """初始化学生状态列表

        Returns:
            List[StudentState]: 学生状态列表
        """
        students = [
            StudentState(
                name="学生A", age=17, subject="物理",
                module="力学", topic="牛顿第二定律",
                level=2.6, attention=0.8, fatigue=0.2
            ),
            StudentState(
                name="学生B", age=16, subject="英语",
                module="词汇", topic="常见单词",
                level=3.0, attention=0.9, fatigue=0.1
            ),
            StudentState(
                name="学生C", age=18, subject="生物",
                module="细胞学", topic="细胞分裂",
                level=2.2, attention=0.7, fatigue=0.3
            )
        ]
        print(f"👨‍🎓 已初始化 {len(students)} 名学生")
        return students

    def _initialize_learning_goals(self) -> Dict[str, List[LearningGoal]]:
        """初始化学习目标

        Returns:
            Dict[str, List[LearningGoal]]: 按学科分类的学习目标字典
        """
        goals = {
            "physics": [
                LearningGoal("力学", "牛顿第二定律", 4.5),
                LearningGoal("力学", "动量守恒", 4.5),
                LearningGoal("电学", "欧姆定律", 4.5),
                LearningGoal("电学", "电容器", 4.5)
            ],
            "math": [
                LearningGoal("代数", "二次方程", 4.5),
                LearningGoal("几何", "平面几何", 4.5),
                LearningGoal("微积分", "极限与连续", 4.5)
            ],
            "chemistry": [
                LearningGoal("无机化学", "化学反应速率", 4.5),
                LearningGoal("有机化学", "烯烃", 4.5)
            ],
            "english": [
                LearningGoal("词汇", "常见单词", 4.5),
                LearningGoal("语法", "时态", 4.5),
                LearningGoal("阅读", "文章理解", 4.5)
            ],
            "chinese": [
                LearningGoal("文学", "唐诗宋词", 4.5),
                LearningGoal("语文基础", "汉字结构", 4.5),
                LearningGoal("写作", "作文技巧", 4.5)
            ],
            "biology": [
                LearningGoal("细胞学", "细胞分裂", 4.5),
                LearningGoal("遗传学", "孟德尔遗传定律", 4.5),
                LearningGoal("生态学", "物种关系", 4.5)
            ]
        }
        total_goals = sum(len(v) for v in goals.values())
        print(f"🎯 已初始化 {total_goals} 个学习目标")
        return goals

    # ======================
    # 显示功能模块
    # ======================

    def show_learning_position(self, state: StudentState):
        """显示当前学习位置

        Args:
            state: 学生状态对象
        """
        print(f"\n📚 学生：{state.name}")
        print(f"   学科：{state.subject} | 模块：{state.module}")
        print(f"   知识点：{state.topic}")
        print(f"   当前水平：Level {state.level:.2f}")
        print(f"   专注度：{state.attention:.2f} | 疲劳度：{state.fatigue:.2f}")

    def show_progress_bar(self, state: StudentState, max_level: float = 5.0, bar_length: int = 20):
        """显示学习进度条

        Args:
            state: 学生状态对象
            max_level: 最大学习水平
            bar_length: 进度条长度
        """
        print(self.viz.create_progress_bar(state.level, max_level, bar_length))

    def compare_with_ideal(self, student: StudentState, ideal: Dict):
        """比较学生与理想轨迹

        Args:
            student: 学生状态对象
            ideal: 理想轨迹字典
        """
        print("\n" + "=" * 60)
        print("🎯 学习轨迹对齐对比")
        print("=" * 60)

        print(f"\n🌟 【理想轨迹 - {ideal['name']}】")
        print(f"   学科：{ideal['subject']} | 知识点：{ideal['topic']}")

        # 创建理想状态的临时对象用于显示进度条
        ideal_state_obj = StudentState(
            name=ideal['name'], age=0, subject=ideal['subject'],
            module=ideal['module'], topic=ideal['topic'],
            level=ideal['level'], attention=1.0, fatigue=0.0
        )
        self.show_progress_bar(ideal_state_obj)

        print(f"\n👨‍🎓 【学生当前轨迹 - {student.name}】")
        self.show_learning_position(student)
        self.show_progress_bar(student)

        # 计算差距
        gap = ideal["level"] - student.level

        # 根据差距大小提供不同的反馈
        if gap > 1.5:
            print(f"\n⚠️  学习差距：{gap:.2f}（需加大学习力度）")
        elif gap > 0.5:
            print(f"\n📈  学习差距：{gap:.2f}（稳步前进中）")
        elif gap > 0:
            print(f"\n✨  学习差距：{gap:.2f}（接近理想水平）")
        else:
            print(f"\n🎉  已达到或超过理想轨迹！")

    # ======================
    # 传感器模拟模块
    # ======================

    def simulate_camera_signal(self) -> Tuple[float, float]:
        """模拟摄像头信号
        生成专注度和情绪波动信号

        Returns:
            Tuple[float, float]: (专注度信号, 情绪波动信号)
        """
        # 生成随机但合理的专注度信号（0.6-0.95）
        attention_signal = random.uniform(0.6, 0.95)

        # 生成随机情绪波动信号（-0.2到0.2）
        emotion_signal = random.uniform(-0.2, 0.2)

        return attention_signal, emotion_signal

    def apply_camera_signal(self, state: StudentState):
        """应用摄像头信号到学生状态

        Args:
            state: 学生状态对象
        """
        attention_signal, emotion_signal = self.simulate_camera_signal()

        # 使用加权平均更新专注度（70%历史值 + 30%新信号）
        state.attention = 0.7 * state.attention + 0.3 * attention_signal

        # 情绪波动影响疲劳度
        state.fatigue += emotion_signal

        # 边界检查
        state.attention = max(0, min(state.attention, 1))
        state.fatigue = max(0, min(state.fatigue, 1))

        print(f"📷 摄像头监测 -> 专注度: {state.attention:.2f} | 情绪波动: {emotion_signal:.2f}")

    # ======================
    # 教学行为模块
    # ======================

    def apply_teaching_action(self, state: StudentState, action: str, session_id: str) -> Dict:
        """应用教学行为到学生

        Args:
            state: 学生状态对象
            action: 教学行为名称
            session_id: 学习会话ID

        Returns:
            Dict: 学习记录信息
        """
        # 记录学习前的状态
        level_before = state.level
        attention_before = state.attention
        fatigue_before = state.fatigue

        print(f"\n🎯 执行教学行为：{action}")

        # 计算当前学习效率（专注度越高、疲劳度越低，效率越高）
        efficiency = state.attention * (1 - state.fatigue)

        # 检查是否为有效的教学行为
        if action in self.strategy_weights:
            config = self.strategy_weights[action]
            base_gain = config["base_gain"]

            # 特殊行为：休息
            if action == "休息":
                # 休息时恢复专注度和降低疲劳度
                state.attention = min(state.attention + 0.15, 1)
                state.fatigue = max(state.fatigue - 0.2, 0)
                print("💤 休息中... 专注度恢复，疲劳度降低")
            else:
                # 正常学习行为：根据效率计算实际增益
                real_gain = base_gain * efficiency
                state.level += real_gain

                # 更新疲劳度
                state.fatigue += config["fatigue_impact"]
                state.fatigue = max(0, min(state.fatigue, 1))

                print(f"📈 学习增益：{real_gain:.3f} (效率系数：{efficiency:.2f})")

        # 确保学习水平在合理范围内
        state.level = max(0, min(state.level, 5))

        # 计算学习效率分数
        efficiency_score = self._calculate_efficiency_score(
            level_before, state.level,
            fatigue_before, state.fatigue
        )

        # 记录学习历史
        state.learning_history.append({
            "session_id": session_id,
            "strategy": action,
            "level_change": state.level - level_before,
            "timestamp": datetime.datetime.now().isoformat()
        })

        # 返回学习记录
        return {
            "level_before": level_before,
            "level_after": state.level,
            "attention_before": attention_before,
            "attention_after": state.attention,
            "fatigue_before": fatigue_before,
            "fatigue_after": state.fatigue,
            "efficiency_score": efficiency_score
        }

    def _calculate_efficiency_score(self, level_before: float, level_after: float,
                                    fatigue_before: float, fatigue_after: float) -> float:
        """计算学习效率分数
        综合考虑水平提升和疲劳度变化

        Args:
            level_before: 学习前水平
            level_after: 学习后水平
            fatigue_before: 学习前疲劳度
            fatigue_after: 学习后疲劳度

        Returns:
            float: 效率分数
        """
        level_gain = level_after - level_before
        fatigue_change = fatigue_after - fatigue_before

        if fatigue_change <= 0:
            # 疲劳度降低，效率更高（乘以1.2奖励）
            efficiency = level_gain * (1 - fatigue_after) * 1.2
        else:
            # 疲劳度增加，效率降低（乘以0.8惩罚）
            efficiency = level_gain * (1 - fatigue_after) * 0.8

        return max(0, efficiency)  # 确保效率分数非负

    # ======================
    # 学习路径管理模块
    # ======================

    def update_learning_path(self, student: StudentState) -> bool:
        """更新学习路径
        当学生掌握当前目标时，跳转到下一个学习目标

        Args:
            student: 学生状态对象

        Returns:
            bool: 是否成功更新路径
        """
        # 获取对应学科的学习目标
        subject_key = self.subject_mapping.get(student.subject, "physics")
        goals = self.learning_goals.get(subject_key, [])

        if not goals:
            print("⚠️  没有设定学习目标！")
            return False

        # 查找当前学习目标
        current_goal = None
        for goal in goals:
            if student.module == goal.module and student.topic == goal.topic:
                current_goal = goal
                break

        # 检查是否已掌握当前目标
        if current_goal and student.level >= current_goal.current_difficulty:
            print(f"🎉  已掌握 {student.topic}，准备跳转到下一个目标...")

            # 查找下一个目标
            next_goal = None
            for i, goal in enumerate(goals):
                if goal == current_goal and i + 1 < len(goals):
                    next_goal = goals[i + 1]
                    break

            if next_goal:
                # 更新学生状态到新目标
                student.module = next_goal.module
                student.topic = next_goal.topic
                student.level = 2.0  # 重置学习水平（新目标从2.0开始）

                print(f"🚀  新目标：{next_goal.module} - {next_goal.topic}")
                return True
            else:
                print("🏆  恭喜！已完成所有学习目标！")
                return False
        else:
            print(f"📖  继续学习当前目标：{student.topic}")
            return False

    # ======================
    # 自适应学习引擎模块
    # ======================

    def adaptive_learning(self, student: StudentState):
        """自适应调整学习难度
        根据学生的学习进度动态调整目标难度

        Args:
            student: 学生状态对象
        """
        # 获取对应学科的学习目标
        subject_key = self.subject_mapping.get(student.subject, "physics")
        goals = self.learning_goals.get(subject_key, [])

        if not goals:
            return

        # 查找当前学习目标
        current_goal = None
        for goal in goals:
            if student.module == goal.module and student.topic == goal.topic:
                current_goal = goal
                break

        if current_goal:
            # 根据学习水平调整难度
            if student.level < 2.5:
                # 掌握较慢，降低难度（最低2.0）
                current_goal.current_difficulty = max(
                    2.0, current_goal.current_difficulty - 0.2
                )
                print(f"📉  {student.name} 掌握较慢，降低目标难度至 {current_goal.current_difficulty:.1f}")
            elif student.level > 4.0:
                # 掌握较快，提高难度（最高5.0）
                current_goal.current_difficulty = min(
                    5.0, current_goal.current_difficulty + 0.2
                )
                print(f"📈  {student.name} 掌握较快，提高目标难度至 {current_goal.current_difficulty:.1f}")

    # ======================
    # 学习策略推荐模块
    # ======================

    def recommend_learning_strategy(self, state: StudentState) -> str:
        """推荐学习策略（基础版）
        基于当前状态推荐最合适的策略

        Args:
            state: 学生状态对象

        Returns:
            str: 推荐的学习策略名称
        """
        # 基于状态的推荐规则
        if state.fatigue > 0.7:
            return "休息"  # 疲劳度过高，建议休息
        elif state.attention < 0.5:
            return "互动学习"  # 专注度过低，建议互动学习
        elif state.attention > 0.85 and state.fatigue < 0.3:
            return "继续学习"  # 状态很好，可以继续深入学习
        elif state.level < 3.0:
            return "讲解"  # 初学者，需要详细讲解
        elif 3.0 <= state.level <= 4.0:
            return "例题"  # 中级学习者，适合例题练习
        else:
            return "反思"  # 高级学习者，适合反思总结

    def enhanced_strategy_recommendation(self, state: StudentState) -> str:
        """增强版策略推荐
        基于历史数据避免策略疲劳

        Args:
            state: 学生状态对象

        Returns:
            str: 推荐的学习策略名称
        """
        # 获取最近的学习历史
        recent_history = state.learning_history[-3:] if state.learning_history else []

        # 分析历史效果
        if recent_history:
            # 检查最近是否频繁使用同一策略
            strategies_used = [record.get("strategy", "未知") for record in recent_history]
            if len(set(strategies_used)) == 1 and len(strategies_used) >= 2:
                # 避免策略疲劳，推荐不同策略
                current_strategy = strategies_used[0]
                all_strategies = list(self.strategy_weights.keys())
                if current_strategy in all_strategies:
                    all_strategies.remove(current_strategy)

                # 随机选择一个不同的策略
                new_strategy = random.choice(all_strategies) if all_strategies else current_strategy
                print(f"🔄  检测到策略疲劳，更换策略：{current_strategy} → {new_strategy}")
                return new_strategy

        # 如果没有策略疲劳问题，使用基础推荐
        return self.recommend_learning_strategy(state)

    # ======================
    # 学习效果评估模块
    # ======================

    def evaluate_learning_effect(self, before: StudentState, after: StudentState,
                                 duration_hours: float = 1.0) -> Dict:
        """评估学习效果

        Args:
            before: 学习前的状态
            after: 学习后的状态
            duration_hours: 学习时长（小时）

        Returns:
            Dict: 学习效果评估报告
        """
        # 计算各项变化
        level_improvement = after.level - before.level
        attention_change = after.attention - before.attention
        fatigue_change = after.fatigue - before.fatigue

        # 计算学习效率
        if duration_hours > 0:
            hourly_gain = level_improvement / duration_hours
        else:
            hourly_gain = level_improvement

        # 计算疲劳效率比（单位疲劳度带来的水平提升）
        if fatigue_change > 0 and fatigue_change != 0:
            fatigue_efficiency = level_improvement / fatigue_change
        else:
            fatigue_efficiency = level_improvement * 2  # 疲劳度降低或不变，效率加倍

        # 构建评估报告
        return {
            "student_name": before.name,
            "subject": before.subject,
            "learning_time_hours": duration_hours,
            "level_improvement": round(level_improvement, 3),
            "hourly_learning_rate": round(hourly_gain, 3),
            "attention_change": round(attention_change, 3),
            "fatigue_change": round(fatigue_change, 3),
            "fatigue_efficiency": round(fatigue_efficiency, 3),
            "final_level": round(after.level, 2),
            "final_attention": round(after.attention, 2),
            "final_fatigue": round(after.fatigue, 2)
        }

    # ======================
    # 完整学习流程模块
    # ======================

    def enhanced_learning_process(self, student: StudentState, num_sessions: int = 4) -> StudentState:
        """增强版学习过程
        完整的个性化学习流程

        Args:
            student: 学生状态对象
            num_sessions: 学习会话数量

        Returns:
            StudentState: 学习后的学生状态
        """
        print(f"\n{'=' * 60}")
        print(f"🚀 开始 {student.name} 的个性化学习旅程")
        print(f"📚 学科：{student.subject} | 初始水平：{student.level:.2f}")
        print(f"{'=' * 60}")

        # 保存初始状态用于后续评估
        initial_state = StudentState(
            name=student.name, age=student.age, subject=student.subject,
            module=student.module, topic=student.topic,
            level=student.level, attention=student.attention,
            fatigue=student.fatigue
        )

        # 生成唯一的会话ID
        session_id = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 执行多个学习会话
        for session in range(1, num_sessions + 1):
            print(f"\n{'─' * 40}")
            print(f"📖 学习会话 {session}/{num_sessions}")
            print(f"{'─' * 40}")

            # 验证学生状态
            is_valid, message = student.validate()
            if not is_valid:
                print(f"⚠️  状态异常：{message}")
                break

            # 显示当前状态
            self.show_learning_position(student)
            self.show_progress_bar(student)

            # 智能推荐学习策略
            strategy = self.enhanced_strategy_recommendation(student)
            print(f"\n🤖 智能推荐策略：{strategy}")

            # 应用教学行为
            record = self.apply_teaching_action(student, strategy, f"{session_id}_{session}")

            # 应用摄像头信号（模拟实时监控）
            self.apply_camera_signal(student)

            # 显示更新后的进度
            self.show_progress_bar(student)

            # 保存学习记录到数据库（如果启用）
            if self.use_database and self.db:
                self.db.save_learning_record(
                    student.name, f"{session_id}_{session}", strategy,
                    record["level_before"], record["level_after"],
                    record["attention_before"], record["attention_after"],
                    record["fatigue_before"], record["fatigue_after"],
                    record["efficiency_score"]
                )

            # 每2次会话显示一次理想对比（仅限物理学科）
            if session % 2 == 0 and student.subject == self.ideal_state["subject"]:
                self.compare_with_ideal(student, self.ideal_state)

            # 最后一次会话时更新学习路径和调整难度
            if session == num_sessions:
                self.update_learning_path(student)
                self.adaptive_learning(student)

        # 保存最终状态到数据库（如果启用）
        if self.use_database and self.db:
            self.db.save_student_state(student)

        # 生成学习效果评估报告
        report = self.evaluate_learning_effect(initial_state, student, num_sessions * 0.5)

        # 显示详细报告
        print(f"\n{'=' * 60}")
        print("📊 学习效果详细报告")
        print(f"{'=' * 60}")

        # 使用表格形式显示报告
        report_table = [
            {"项目": "学生姓名", "值": report["student_name"]},
            {"项目": "学习科目", "值": report["subject"]},
            {"项目": "学习时长(小时)", "值": report["learning_time_hours"]},
            {"项目": "水平提升", "值": f"{report['level_improvement']:+.3f}"},
            {"项目": "每小时学习率", "值": f"{report['hourly_learning_rate']:.3f}"},
            {"项目": "专注度变化", "值": f"{report['attention_change']:+.3f}"},
            {"项目": "疲劳度变化", "值": f"{report['fatigue_change']:+.3f}"},
            {"项目": "疲劳效率比", "值": f"{report['fatigue_efficiency']:.3f}"},
            {"项目": "最终水平", "值": report["final_level"]},
            {"项目": "最终专注度", "值": report["final_attention"]},
            {"项目": "最终疲劳度", "值": report["final_fatigue"]},
        ]

        print(self.viz.create_simple_table(report_table))

        return student

    # ======================
    # 文本可视化模块
    # ======================

    def visualize_learning_progress(self, student_name: str):
        """文本可视化学习进度

        Args:
            student_name: 学生姓名
        """
        # 获取学习历史（如果启用数据库）
        if self.use_database and self.db:
            history = self.db.get_student_history(student_name)
        else:
            # 从学生对象中获取历史
            history = []
            for student in self.students:
                if student.name == student_name:
                    history = student.learning_history
                    break

        if not history:
            print(f"⚠️  没有找到 {student_name} 的学习历史")
            return

        print(f"\n{'=' * 60}")
        print(f"📈 {student_name} 学习进度分析")
        print(f"{'=' * 60}")

        # 提取数据用于可视化
        sessions = list(range(1, len(history) + 1))
        levels_before = [record.get("level_before", 0) for record in history]
        levels_after = [record.get("level_after", 0) for record in history]

        # 计算每次学习的增益
        level_changes = []
        for i in range(len(history)):
            if i < len(levels_after) and i < len(levels_before):
                level_changes.append(levels_after[i] - levels_before[i])

        strategies = [record.get("strategy", "未知") for record in history]

        # 1. 显示水平变化趋势
        print("\n1️⃣ 学习水平变化趋势:")
        if len(levels_after) > 1:
            print(self.viz.create_line_chart(levels_after, width=40, height=8))
        else:
            print("  数据不足生成趋势图")

        # 2. 显示策略使用统计
        print("\n2️⃣ 学习策略使用统计:")
        strategy_counts = {}
        for strategy in strategies:
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        if strategy_counts:
            strategy_table = []
            for strategy, count in strategy_counts.items():
                percentage = count / len(strategies) * 100
                strategy_table.append({
                    "策略": strategy,
                    "使用次数": count,
                    "占比(%)": f"{percentage:.1f}"
                })

            print(self.viz.create_simple_table(strategy_table, ["策略", "使用次数", "占比(%)"]))

        # 3. 显示累计学习增益
        print("\n3️⃣ 累计学习效果:")
        cumulative_gain = []
        total = 0
        for i, gain in enumerate(level_changes):
            total += gain
            cumulative_gain.append(total)

            # 显示前10次学习的详细信息
            if i < 10:
                sparkline = self.viz.create_sparkline([gain]) if gain != 0 else "   "
                gain_str = f"{gain:+.3f}" if gain != 0 else " 0.000"
                print(f"   会话{i + 1:2}: 增益{gain_str} {sparkline}")

        if len(history) > 10:
            print(f"   ... 还有 {len(history) - 10} 条记录")

        # 显示统计信息
        print(f"\n   📊 统计信息:")
        if level_changes:
            avg_gain = sum(level_changes) / len(level_changes)
            max_gain = max(level_changes) if level_changes else 0
            min_gain = min(level_changes) if level_changes else 0

            print(f"       总学习增益: {total:.3f}")
            print(f"       平均每次增益: {avg_gain:.3f}")
            print(f"       最大单次增益: {max_gain:.3f}")
            print(f"       最小单次增益: {min_gain:.3f}")

        # 4. 显示学习效率分布
        print("\n4️⃣ 学习效率分布:")
        efficiency_scores = [record.get("efficiency_score", 0) for record in history]
        if efficiency_scores:
            avg_efficiency = sum(efficiency_scores) / len(efficiency_scores)
            max_efficiency = max(efficiency_scores)

            print(f"   平均学习效率: {avg_efficiency:.3f}")
            print(f"   最高学习效率: {max_efficiency:.3f}")

            # 显示效率条形图
            if max_efficiency > 0:
                efficiency_data = {
                    "平均效率": avg_efficiency,
                    "最高效率": max_efficiency,
                    "当前效率": efficiency_scores[-1] if efficiency_scores else 0
                }
                print(self.viz.create_bar_chart(efficiency_data, bar_length=20))

    def generate_comprehensive_report(self):
        """生成综合学习报告"""
        print(f"\n{'=' * 60}")
        print("📋 生成综合学习报告")
        print(f"{'=' * 60}")

        # 构建报告数据结构
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_students": len(self.students),
            "students": [],
            "summary": {}
        }

        total_improvement = 0
        total_sessions = 0

        # 为每个学生生成报告
        for student in self.students:
            student_history = student.learning_history

            student_report = {
                "name": student.name,
                "subject": student.subject,
                "current_level": student.level,
                "current_attention": student.attention,
                "current_fatigue": student.fatigue,
                "learning_sessions": len(student_history),
                "strategies_used": {},
                "total_improvement": 0
            }

            if student_history:
                # 计算总提升（如果有历史记录）
                if len(student_history) > 1:
                    # 使用历史记录计算
                    initial_level = student_history[0].get("level_before", 0)
                    final_level = student.level
                    total_improvement += final_level - initial_level
                    student_report["total_improvement"] = final_level - initial_level
                else:
                    # 单个记录的情况
                    student_report["total_improvement"] = student_history[0].get("level_change", 0)

                total_sessions += len(student_history)

                # 统计策略使用情况
                for record in student_history:
                    strategy = record.get("strategy", "未知")
                    student_report["strategies_used"][strategy] = \
                        student_report["strategies_used"].get(strategy, 0) + 1

            report["students"].append(student_report)

        # 生成摘要统计
        if len(self.students) > 0:
            avg_improvement = total_improvement / len(self.students) if total_improvement > 0 else 0
            avg_sessions = total_sessions / len(self.students) if total_sessions > 0 else 0

            # 统计最受欢迎的学科
            subjects = [s.subject for s in self.students]
            if subjects:
                # 找到出现次数最多的学科
                subject_count = {}
                for subject in subjects:
                    subject_count[subject] = subject_count.get(subject, 0) + 1

                most_popular_subject = max(subject_count.items(), key=lambda x: x[1])[0]
            else:
                most_popular_subject = "无数据"

            report["summary"] = {
                "average_improvement_per_student": round(avg_improvement, 3),
                "average_sessions_per_student": round(avg_sessions, 1),
                "most_popular_subject": most_popular_subject
            }

        # 保存报告到JSON文件
        report_file = f"learning_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n✅ 综合学习报告已保存为 '{report_file}'")

        # 在控制台显示摘要
        print(f"\n{'=' * 60}")
        print("📊 学习系统摘要")
        print(f"{'=' * 60}")
        print(f"总学生数：{report['total_students']}")

        if report['summary']:
            print(f"平均每个学生提升：{report['summary']['average_improvement_per_student']:.3f}")
            print(f"平均每个学生会话：{report['summary']['average_sessions_per_student']:.1f}")
            print(f"最受欢迎的学科：{report['summary']['most_popular_subject']}")

        # 显示每个学生的简要信息
        print(f"\n👨‍🎓 学生详情：")
        student_table = []
        for student_report in report['students']:
            student_table.append({
                "姓名": student_report['name'],
                "学科": student_report['subject'],
                "当前水平": f"{student_report['current_level']:.2f}",
                "会话数": student_report['learning_sessions'],
                "提升": f"{student_report['total_improvement']:+.3f}"
            })

        print(self.viz.create_simple_table(student_table, ["姓名", "学科", "当前水平", "会话数", "提升"]))

        return report

    # ======================
    # 系统管理模块
    # ======================

    def run_demo(self):
        """运行系统演示"""
        print("\n" + "=" * 70)
        print("🤖 智能自适应学习系统 - 演示模式")
        print("=" * 70)
        print(f"📅 系统时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"👨‍🎓 学生数量：{len(self.students)}")
        print(f"🎯 学习策略：{len(self.strategy_weights)} 种")
        print("=" * 70)

        # 为每个学生运行学习过程
        for i, student in enumerate(self.students):
            print(f"\n{'#' * 70}")
            print(f"👨‍🎓 学生 {i + 1}/{len(self.students)}: {student.name}")
            print(f"{'#' * 70}")

            # 运行个性化学习过程
            num_sessions = random.randint(3, 6)  # 随机选择3-6个学习会话
            self.enhanced_learning_process(student, num_sessions)

            # 询问是否查看学习进度可视化
            if input(f"\n是否查看 {student.name} 的学习进度图表？(y/n): ").lower() == 'y':
                self.visualize_learning_progress(student.name)

        # 询问是否生成综合报告
        if input("\n是否生成综合学习报告？(y/n): ").lower() == 'y':
            self.generate_comprehensive_report()

        # 保存所有数据
        self.save_all_data()

        # 关闭数据库连接
        if self.use_database and self.db:
            self.db.close()

        print(f"\n{'=' * 70}")
        print("🎉 学习系统运行完成！")
        print("📁 数据已保存到以下文件：")
        print("   - students_final_state.json")
        print("   - learning_goals_state.json")
        print("   - learning_report_*.json")
        print(f"{'=' * 70}")

    def save_all_data(self):
        """保存所有系统数据到文件"""
        print("\n💾 正在保存系统数据...")

        try:
            # 保存学生状态到JSON文件
            students_dict = [s.to_dict() for s in self.students]
            with open("students_final_state.json", "w", encoding='utf-8') as f:
                json.dump(students_dict, f, ensure_ascii=False, indent=2)
            print("✅ 学生状态已保存到 students_final_state.json")

            # 保存学习目标状态
            goals_dict = {}
            for subject, goals in self.learning_goals.items():
                goals_dict[subject] = [goal.to_dict() for goal in goals]

            with open("learning_goals_state.json", "w", encoding='utf-8') as f:
                json.dump(goals_dict, f, ensure_ascii=False, indent=2)
            print("✅ 学习目标已保存到 learning_goals_state.json")

        except Exception as e:
            print(f"❌ 保存数据时出错: {e}")

    def run_single_student_demo(self, student_index: int = 0, num_sessions: int = 5):
        """运行单个学生的演示

        Args:
            student_index: 学生索引（0-based）
            num_sessions: 学习会话数量
        """
        if student_index >= len(self.students):
            print(f"❌ 错误：学生索引 {student_index} 超出范围（0-{len(self.students) - 1}）")
            return

        student = self.students[student_index]
        print(f"\n🎯 运行 {student.name} 的单人演示模式")
        print(f"   学科：{student.subject} | 初始水平：{student.level:.2f}")
        print(f"   学习会话：{num_sessions} 次")

        self.enhanced_learning_process(student, num_sessions)
        self.visualize_learning_progress(student.name)

    def show_system_info(self):
        """显示系统信息"""
        print("\n" + "=" * 60)
        print("📋 系统信息")
        print("=" * 60)
        print(f"系统版本: 2.2 (纯标准库版本)")
        print(f"学生数量: {len(self.students)}")
        print(f"学习策略: {len(self.strategy_weights)} 种")
        print(f"数据库状态: {'已启用' if self.use_database else '已禁用'}")

        total_goals = sum(len(v) for v in self.learning_goals.values())
        print(f"学习目标总数: {total_goals}")
        print("=" * 60)

        # 显示学生列表
        print("\n👨‍🎓 学生列表:")
        student_table = []
        for i, student in enumerate(self.students):
            student_table.append({
                "序号": i + 1,
                "姓名": student.name,
                "学科": student.subject,
                "水平": f"{student.level:.2f}",
                "专注度": f"{student.attention:.2f}",
                "疲劳度": f"{student.fatigue:.2f}"
            })

        print(self.viz.create_simple_table(student_table, ["序号", "姓名", "学科", "水平", "专注度", "疲劳度"]))

        # 显示学习策略
        print("\n🎯 可用学习策略:")
        strategy_table = []
        for i, (strategy, config) in enumerate(self.strategy_weights.items()):
            strategy_table.append({
                "序号": i + 1,
                "策略": strategy,
                "基础增益": f"{config['base_gain']:.2f}",
                "疲劳影响": f"{config['fatigue_impact']:+.2f}"
            })

        print(self.viz.create_simple_table(strategy_table, ["序号", "策略", "基础增益", "疲劳影响"]))


# ======================
# 主程序入口
# ======================

def main():
    """主函数 - 程序入口点"""
    print("🎓 欢迎使用智能自适应学习系统")
    print("版本: 2.2 (纯标准库版本)")
    print("作者: AI助手")
    print("=" * 50)

    # 创建学习系统实例
    # 参数说明：use_database=True 启用数据库，False 禁用数据库
    learning_system = AdaptiveLearningSystem(use_database=True)

    # 显示系统信息
    learning_system.show_system_info()

    # 选择运行模式
    print("\n请选择运行模式:")
    print("1. 完整演示模式（所有学生）")
    print("2. 单人演示模式")
    print("3. 仅显示系统信息")
    print("4. 退出系统")

    choice = input("\n请输入选择 (1-4): ").strip()

    if choice == "1":
        # 运行完整演示
        learning_system.run_demo()
    elif choice == "2":
        # 运行单人演示
        print("\n请选择学生:")
        for i, student in enumerate(learning_system.students):
            print(f"{i + 1}. {student.name} ({student.subject})")

        try:
            student_choice = int(input("\n请输入学生编号 (1-3): ")) - 1
            if 0 <= student_choice < len(learning_system.students):
                sessions = input("请输入学习会话数量 (默认5): ").strip()
                num_sessions = int(sessions) if sessions.isdigit() else 5
                learning_system.run_single_student_demo(student_choice, num_sessions)
            else:
                print("❌ 无效的学生编号")
        except ValueError:
            print("❌ 请输入有效的数字")
    elif choice == "3":
        # 仅显示系统信息
        learning_system.show_system_info()
        print("\nℹ️  系统信息显示完成")
    elif choice == "4":
        print("👋 感谢使用，再见！")
        return
    else:
        print("❌ 无效的选择，请重新运行程序")

    # 结束程序
    input("\n按回车键退出程序...")


# ======================
# 程序启动
# ======================

if __name__ == "__main__":
    """程序启动点"""
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 程序被用户中断")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        import traceback

        traceback.print_exc()
        print("\n请检查错误信息并确保所有文件权限正确")
    finally:
        print("\n🎓 智能学习系统已关闭")