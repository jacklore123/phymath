"""
智能自适应学习系统 - 增强整合版
===================================
整合了以下功能：
1. 空白学生体模型（认知发展规律）
2. 牛顿榜样轨迹对齐
3. 自适应学习路径规划
4. 实时监控模拟
5. 蒙特卡洛树搜索路径规划
6. 完整的可视化系统

仅使用Python标准库，无需额外依赖
"""

import random
import json
import datetime
import math
import sqlite3
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
from enum import Enum

# ======================
# 系统配置
# ======================
CONFIG = {
    "total_days": 4380,  # K12阶段总天数
    "age_start": 6,  # 起始年龄
    "age_end": 18,  # 结束年龄
    "skill_min": 0.0,  # 技能最小值
    "skill_max": 5.0,  # 技能最大值
    "learning_rate": 0.1,  # 学习率
    "forgetting_rate": 0.001,  # 遗忘率
    "random_factor": 0.01,  # 随机因素
}


# ======================
# 数据类型定义
# ======================

class LearningStrategy(Enum):
    """学习策略枚举"""
    LECTURE = "讲解"
    EXAMPLE = "例题"
    REFLECTION = "反思"
    REST = "休息"
    INTERACTIVE = "互动学习"
    CONTINUE = "继续学习"
    REVIEW = "复习"
    MEMORIZE = "记忆学习"
    PROBLEM_SOLVING = "问题解决"
    CREATIVE_THINKING = "创造性思考"
    SPEED_TRAINING = "速度训练"
    PROJECT_BASED = "项目式学习"


@dataclass
class LearningGoal:
    """学习目标"""
    module: str
    topic: str
    target_level: float
    current_difficulty: float = 4.5

    def to_dict(self) -> Dict:
        return {
            "module": self.module,
            "topic": self.topic,
            "level": self.target_level,
            "difficulty": self.current_difficulty
        }


@dataclass
class CognitiveSkills:
    """认知技能维度"""
    knowledge: float = 0.5  # 知识积累
    abstraction: float = 0.2  # 抽象思维
    reasoning: float = 0.3  # 逻辑推理
    speed: float = 0.2  # 思维速度
    creativity: float = 0.1  # 创造力
    memory: float = 0.4  # 记忆力


# ======================
# 空白学生体类
# ======================

class BlankStudent:
    """空白学生体 - 具有认知发展规律的学习者"""

    def __init__(self, name: str = "default", age: int = 6, subject: str = "物理"):
        self.name = name
        self.age = age
        self.subject = subject
        self.module = ""
        self.topic = ""
        self.day = 0

        # 认知技能
        self.skills = CognitiveSkills()

        # 学习状态
        self.level = 0.5  # 综合水平（基于技能计算）
        self.attention = 0.8
        self.fatigue = 0.2
        self.learning_history = []
        self.last_updated = datetime.datetime.now().isoformat()

        # 认知发展曲线
        self.cognitive_development_curve = self._init_development_curve()

    def _init_development_curve(self) -> Dict[str, List[float]]:
        """初始化认知发展曲线"""
        curves = {}
        skills = ["knowledge", "abstraction", "reasoning", "speed", "creativity", "memory"]

        for skill in skills:
            if skill == "knowledge":
                curves[skill] = [0.3, 0.7, 0.1]
            elif skill == "abstraction":
                curves[skill] = [0.1, 0.5, 0.3]
            elif skill == "reasoning":
                curves[skill] = [0.2, 0.6, 0.2]
            else:
                curves[skill] = [0.2, 0.5, 0.3]

        return curves

    def get_state(self) -> Dict:
        """获取当前状态"""
        skills_dict = {
            "knowledge": self.skills.knowledge,
            "abstraction": self.skills.abstraction,
            "reasoning": self.skills.reasoning,
            "speed": self.skills.speed,
            "creativity": self.skills.creativity,
            "memory": self.skills.memory
        }

        return {
            "name": self.name,
            "age": self.age,
            "subject": self.subject,
            "module": self.module,
            "topic": self.topic,
            "skills": skills_dict,
            "level": self.level,
            "attention": self.attention,
            "fatigue": self.fatigue,
            "day": self.day,
            "learning_history": self.learning_history,
            "last_updated": self.last_updated
        }

    def update_age(self):
        """更新年龄"""
        if self.day % 365 == 0 and self.day > 0:
            self.age += 1

    def apply_cognitive_development(self):
        """应用认知发展规律"""
        skills_dict = self.skills.__dict__

        for skill, curve in self.cognitive_development_curve.items():
            if skill in skills_dict:
                age_factor = self.age / 18.0
                k = curve[0] * 10
                m = curve[1]

                development_gain = 1 / (1 + math.exp(-k * (age_factor - m)))
                development_gain = min(development_gain, curve[2])

                current_value = getattr(self.skills, skill)
                new_value = current_value + development_gain * CONFIG["learning_rate"]
                new_value = min(new_value, CONFIG["skill_max"])
                setattr(self.skills, skill, new_value)

        # 更新综合水平（技能平均值）
        self._update_level()

    def _update_level(self):
        """更新综合水平"""
        skills_dict = self.skills.__dict__
        total = sum(skills_dict.values())
        count = len(skills_dict)
        self.level = total / count if count > 0 else 0

    def apply_forgetting(self):
        """应用遗忘规律"""
        skills_dict = self.skills.__dict__

        for skill in skills_dict:
            current_value = getattr(self.skills, skill)
            forget_amount = current_value * CONFIG["forgetting_rate"]
            new_value = max(current_value - forget_amount, CONFIG["skill_min"])
            setattr(self.skills, skill, new_value)

        self._update_level()

    def calculate_fatigue(self, action_intensity: float):
        """计算疲劳度"""
        self.fatigue = min(1.0, self.fatigue + action_intensity * 0.1)

        # 疲劳影响
        if self.fatigue > 0.7:
            fatigue_penalty = (self.fatigue - 0.7) * 0.3
            skills_dict = self.skills.__dict__

            for skill in skills_dict:
                if random.random() < 0.3:  # 30%概率受疲劳影响
                    current_value = getattr(self.skills, skill)
                    new_value = max(current_value - fatigue_penalty * random.random(), CONFIG["skill_min"])
                    setattr(self.skills, skill, new_value)

            self._update_level()

    def rest(self):
        """休息恢复"""
        self.fatigue = max(0.0, self.fatigue - 0.3)
        self.attention = min(1.0, self.attention + 0.1)

    def validate(self) -> Tuple[bool, str]:
        """验证状态合法性"""
        if not 0 <= self.level <= 5:
            return False, f"学习水平 {self.level} 超出范围 [0, 5]"

        if not 0 <= self.attention <= 1:
            return False, f"专注度 {self.attention} 超出范围 [0, 1]"

        if not 0 <= self.fatigue <= 1:
            return False, f"疲劳度 {self.fatigue} 超出范围 [0, 1]"

        return True, "状态验证通过"


# ======================
# 牛顿榜样模型
# ======================

class NewtonRoleModel:
    """牛顿榜样模型 - 目标学习路径"""

    def __init__(self):
        self.target_trajectory = self._generate_newton_trajectory()
        self.final_target = {
            "knowledge": 4.8,
            "abstraction": 4.7,
            "reasoning": 4.6,
            "speed": 4.0,
            "creativity": 4.5,
            "memory": 4.2,
        }

    def _generate_newton_trajectory(self) -> Dict[int, Dict[str, float]]:
        """生成牛顿的成长轨迹"""
        trajectory = {}

        for age in range(6, 19):
            progress = (age - 6) / 12.0

            trajectory[age] = {
                "knowledge": 0.5 + 4.3 * progress ** 1.2,
                "abstraction": 0.3 + 4.4 * progress ** 1.5,
                "reasoning": 0.4 + 4.2 * progress ** 1.3,
                "speed": 0.2 + 3.8 * progress,
                "creativity": 0.2 + 4.3 * progress ** 1.4,
                "memory": 0.4 + 3.8 * progress,
            }

        return trajectory

    def get_target_at_age(self, age: int) -> Dict[str, float]:
        """获取特定年龄的目标值"""
        if age in self.target_trajectory:
            return self.target_trajectory[age]
        elif age < 6:
            return self.target_trajectory[6]
        else:
            return self.final_target

    def calculate_distance(self, student_skills: Dict[str, float], age: int) -> float:
        """计算与牛顿目标的距离"""
        target = self.get_target_at_age(age)

        distance = 0
        for skill in student_skills:
            if skill in target:
                diff = student_skills[skill] - target[skill]
                distance += diff ** 2

        return math.sqrt(distance)

    def calculate_similarity(self, student_skills: Dict[str, float], age: int) -> float:
        """计算与牛顿的相似度"""
        distance = self.calculate_distance(student_skills, age)
        max_distance = math.sqrt(len(student_skills) * (CONFIG["skill_max"] ** 2))

        similarity = 1.0 - (distance / max_distance)
        return max(0.0, min(1.0, similarity))

    def get_equivalent_age(self, student_skills: Dict[str, float]) -> float:
        """计算相当于牛顿的等效年龄"""
        best_age = 6
        best_similarity = 0

        for age in range(6, 19):
            similarity = self.calculate_similarity(student_skills, age)
            if similarity > best_similarity:
                best_similarity = similarity
                best_age = age

        return best_age + (best_similarity - 0.5) * 2


# ======================
# 学习行为系统
# ======================

class LearningActions:
    """学习行为系统"""

    def __init__(self):
        self.actions = {
            "memorize": {
                "name": "记忆学习",
                "effects": {"knowledge": 0.08, "memory": 0.06},
                "fatigue": 0.3,
                "description": "背诵、记忆知识点"
            },
            "problem_solving": {
                "name": "问题解决",
                "effects": {"knowledge": 0.04, "reasoning": 0.07, "abstraction": 0.03},
                "fatigue": 0.4,
                "description": "解题训练"
            },
            "reflection": {
                "name": "反思总结",
                "effects": {"abstraction": 0.08, "reasoning": 0.05},
                "fatigue": 0.2,
                "description": "总结反思学习内容"
            },
            "creative_thinking": {
                "name": "创造性思考",
                "effects": {"creativity": 0.09, "abstraction": 0.05},
                "fatigue": 0.5,
                "description": "创新性思考、头脑风暴"
            },
            "speed_training": {
                "name": "速度训练",
                "effects": {"speed": 0.10, "knowledge": 0.02},
                "fatigue": 0.6,
                "description": "快速解题、限时训练"
            },
            "rest": {
                "name": "休息恢复",
                "effects": {},
                "fatigue": -0.5,
                "description": "适当休息"
            },
            "project_based": {
                "name": "项目式学习",
                "effects": {"knowledge": 0.05, "reasoning": 0.06, "creativity": 0.07},
                "fatigue": 0.4,
                "description": "完成综合性项目"
            },
            "lecture": {
                "name": "讲解",
                "effects": {"knowledge": 0.07, "memory": 0.05},
                "fatigue": 0.2,
                "description": "教师讲解知识点"
            },
            "example": {
                "name": "例题",
                "effects": {"knowledge": 0.05, "reasoning": 0.06},
                "fatigue": 0.3,
                "description": "例题分析与练习"
            },
            "interactive": {
                "name": "互动学习",
                "effects": {"knowledge": 0.06, "creativity": 0.04},
                "fatigue": 0.2,
                "description": "互动式学习"
            },
            "continue": {
                "name": "继续学习",
                "effects": {"knowledge": 0.04, "reasoning": 0.04, "abstraction": 0.03},
                "fatigue": 0.4,
                "description": "持续深入学习"
            },
            "review": {
                "name": "复习",
                "effects": {"knowledge": 0.03, "memory": 0.07},
                "fatigue": 0.2,
                "description": "复习巩固知识"
            }
        }

    def get_available_actions(self, student_fatigue: float) -> List[str]:
        """获取可用的学习行为"""
        available = []

        for action_id, action_info in self.actions.items():
            if action_id == "rest" or student_fatigue < 0.8:
                available.append(action_id)

        return available

    def apply_action(self, student: BlankStudent, action_id: str) -> Dict[str, float]:
        """应用学习行为"""
        if action_id not in self.actions:
            return {}

        action = self.actions[action_id]
        effects = action["effects"].copy()

        # 添加随机因素
        for skill in effects:
            random_effect = random.uniform(-CONFIG["random_factor"], CONFIG["random_factor"])
            effects[skill] += random_effect

        # 应用效果到学生
        for skill, effect in effects.items():
            if hasattr(student.skills, skill):
                current_value = getattr(student.skills, skill)
                new_value = current_value + effect
                new_value = max(CONFIG["skill_min"], min(CONFIG["skill_max"], new_value))
                setattr(student.skills, skill, new_value)

        # 更新疲劳度
        student.calculate_fatigue(action["fatigue"])

        # 更新综合水平
        student._update_level()

        return effects

    def get_recommended_action(self, student: BlankStudent, newton_model: NewtonRoleModel) -> str:
        """根据当前状态推荐学习行为"""
        age = student.age
        skills_dict = student.skills.__dict__
        target = newton_model.get_target_at_age(age)

        # 计算技能差距
        gaps = {}
        for skill in skills_dict:
            if skill in target:
                gaps[skill] = target[skill] - getattr(student.skills, skill)

        # 找出最大差距的技能
        if not gaps:
            return "rest"

        max_gap_skill = max(gaps.items(), key=lambda x: x[1])[0]

        # 根据技能差距推荐行为
        action_mapping = {
            "knowledge": ["memorize", "lecture", "review"],
            "abstraction": ["reflection", "creative_thinking"],
            "reasoning": ["problem_solving", "example", "project_based"],
            "speed": ["speed_training"],
            "creativity": ["creative_thinking", "project_based", "interactive"],
            "memory": ["memorize", "review"]
        }

        if max_gap_skill in action_mapping:
            possible_actions = action_mapping[max_gap_skill]
            available = self.get_available_actions(student.fatigue)

            for action in possible_actions:
                if action in available:
                    return action

        # 默认或疲劳时休息
        if student.fatigue > 0.6:
            return "rest"

        available = self.get_available_actions(student.fatigue)
        return random.choice(available) if available else "rest"


# ======================
# 数据库管理模块
# ======================

class LearningDatabase:
    """数据库管理类"""

    def __init__(self, db_path: str = "enhanced_learning_system.db"):
        self.conn = sqlite3.connect(db_path)
        self.create_tables()

    def create_tables(self):
        """创建数据库表"""
        cursor = self.conn.cursor()

        # 学生状态表
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
                           knowledge
                           REAL,
                           abstraction
                           REAL,
                           reasoning
                           REAL,
                           speed
                           REAL,
                           creativity
                           REAL,
                           memory
                           REAL,
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

        # 学习记录表
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

    def save_student_state(self, student: BlankStudent):
        """保存学生状态"""
        cursor = self.conn.cursor()

        # 检查学生是否已存在
        cursor.execute('SELECT id FROM students WHERE name = ?', (student.name,))
        result = cursor.fetchone()

        skills_dict = student.skills.__dict__

        if result:
            # 更新现有记录
            cursor.execute('''
                           UPDATE students
                           SET age              = ?,
                               subject          = ?,
                               module           = ?,
                               topic            = ?,
                               knowledge        = ?,
                               abstraction      = ?,
                               reasoning        = ?,
                               speed            = ?,
                               creativity       = ?,
                               memory           = ?,
                               level            = ?,
                               attention        = ?,
                               fatigue          = ?,
                               learning_history = ?,
                               last_updated     = ?
                           WHERE name = ?
                           ''', (
                               student.age, student.subject, student.module, student.topic,
                               skills_dict['knowledge'], skills_dict['abstraction'],
                               skills_dict['reasoning'], skills_dict['speed'],
                               skills_dict['creativity'], skills_dict['memory'],
                               student.level, student.attention, student.fatigue,
                               json.dumps(student.learning_history),
                               datetime.datetime.now().isoformat(),
                               student.name
                           ))
        else:
            # 插入新记录
            cursor.execute('''
                           INSERT INTO students
                           (name, age, subject, module, topic,
                            knowledge, abstraction, reasoning, speed, creativity, memory,
                            level, attention, fatigue, learning_history, last_updated)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                           ''', (
                               student.name, student.age, student.subject, student.module, student.topic,
                               skills_dict['knowledge'], skills_dict['abstraction'],
                               skills_dict['reasoning'], skills_dict['speed'],
                               skills_dict['creativity'], skills_dict['memory'],
                               student.level, student.attention, student.fatigue,
                               json.dumps(student.learning_history),
                               datetime.datetime.now().isoformat()
                           ))

        self.conn.commit()

    def save_learning_record(self, student_name: str, session_id: str,
                             strategy: str, level_before: float, level_after: float,
                             attention_before: float, attention_after: float,
                             fatigue_before: float, fatigue_after: float,
                             efficiency_score: float):
        """保存学习记录"""
        cursor = self.conn.cursor()

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

    def get_student_history(self, student_name: str) -> List[Dict]:
        """获取学习历史"""
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

        return records

    def close(self):
        """关闭数据库连接"""
        self.conn.close()


# ======================
# 数学工具类
# ======================

class MathUtils:
    """数学工具类"""

    @staticmethod
    def mean(values: List[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def min_max_normalize(values: List[float]) -> List[float]:
        if not values:
            return []

        min_val = min(values)
        max_val = max(values)

        if max_val == min_val:
            return [0.5] * len(values)

        return [(v - min_val) / (max_val - min_val) for v in values]

    @staticmethod
    def linspace(start: float, stop: float, num: int = 50) -> List[float]:
        if num < 2:
            return [start]

        step = (stop - start) / (num - 1)
        return [start + step * i for i in range(num)]


# ======================
# 文本可视化工具类
# ======================

class TextVisualizer:
    """文本可视化工具类"""

    def __init__(self):
        self.math_utils = MathUtils()

    def create_progress_bar(self, value: float, max_value: float = 5.0,
                            bar_length: int = 20, show_percentage: bool = True) -> str:
        normalized_value = max(0, min(value, max_value))
        filled_length = int(normalized_value / max_value * bar_length)
        empty_length = bar_length - filled_length

        bar = "█" * filled_length + "░" * empty_length

        if show_percentage:
            percentage = (normalized_value / max_value) * 100
            return f"[{bar}] {normalized_value:.2f}/{max_value} ({percentage:.1f}%)"
        else:
            return f"[{bar}] {normalized_value:.2f}/{max_value}"

    def create_sparkline(self, values: List[float]) -> str:
        if not values:
            return "无数据"

        normalized = self.math_utils.min_max_normalize(values)
        chars = " ▁▂▃▄▅▆▇█"
        sparkline = ""

        for norm_val in normalized:
            char_index = int(norm_val * (len(chars) - 1))
            sparkline += chars[char_index]

        return sparkline

    def create_bar_chart(self, data: Dict[str, float], bar_length: int = 20) -> str:
        if not data:
            return "无数据"

        chart_lines = []
        max_value = max(data.values()) if data.values() else 1

        for label, value in data.items():
            bar_len = int(value / max_value * bar_length) if max_value > 0 else 0
            bar = "█" * bar_len + " " * (bar_length - bar_len)
            chart_lines.append(f"{label:10} |{bar}| {value:.2f}")

        return "\n".join(chart_lines)

    def create_line_chart(self, values: List[float], width: int = 50, height: int = 10) -> str:
        if len(values) < 2:
            return "数据点不足"

        normalized = self.math_utils.min_max_normalize(values)
        grid = [[' ' for _ in range(width)] for _ in range(height)]

        for i, norm_val in enumerate(normalized):
            x = int(i / (len(values) - 1) * (width - 1))
            y = int((1 - norm_val) * (height - 1))

            if 0 <= x < width and 0 <= y < height:
                grid[y][x] = '●'

        # 添加连接线
        for i in range(len(values) - 1):
            x1 = int(i / (len(values) - 1) * (width - 1))
            y1 = int((1 - normalized[i]) * (height - 1))
            x2 = int((i + 1) / (len(values) - 1) * (width - 1))
            y2 = int((1 - normalized[i + 1]) * (height - 1))

            steps = max(abs(x2 - x1), abs(y2 - y1))
            if steps > 0:
                for s in range(steps + 1):
                    x = int(x1 + (x2 - x1) * s / steps)
                    y = int(y1 + (y2 - y1) * s / steps)
                    if 0 <= x < width and 0 <= y < height and grid[y][x] == ' ':
                        grid[y][x] = '·'

        chart_lines = []
        for row in grid:
            chart_lines.append(''.join(row))

        min_val = min(values)
        max_val = max(values)
        chart_lines.append(f"最小值: {min_val:.2f}  最大值: {max_val:.2f}")

        return "\n".join(chart_lines)

    def create_simple_table(self, data: List[Dict], headers: List[str] = None) -> str:
        if not data:
            return "无数据"

        if headers:
            col_names = headers
        else:
            col_names = list(data[0].keys())

        col_widths = []
        for col in col_names:
            max_width = len(str(col))
            for row in data:
                if col in row:
                    max_width = max(max_width, len(str(row[col])))
            col_widths.append(max_width + 2)

        table_lines = []

        # 顶部边框
        header_line = "┌"
        for width in col_widths:
            header_line += "─" * width + "┬"
        header_line = header_line[:-1] + "┐"
        table_lines.append(header_line)

        # 表头
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
# 增强学习系统主类
# ======================

class EnhancedLearningSystem:
    """增强学习系统 - 整合版"""

    def __init__(self, use_database: bool = True):
        print("🤖 初始化增强智能学习系统...")

        self.use_database = use_database
        if use_database:
            self.db = LearningDatabase()
        else:
            self.db = None

        # 初始化组件
        self.math_utils = MathUtils()
        self.viz = TextVisualizer()
        self.newton_model = NewtonRoleModel()
        self.learning_actions = LearningActions()

        # 初始化学生
        self.students = self._initialize_students()

        # 学习策略权重
        self.strategy_weights = {
            "讲解": {"base_gain": 0.2, "fatigue_impact": 0.05},
            "例题": {"base_gain": 0.3, "fatigue_impact": 0.08},
            "反思": {"base_gain": 0.4, "fatigue_impact": -0.1},
            "休息": {"base_gain": 0, "fatigue_impact": -0.15},
            "互动学习": {"base_gain": 0.5, "fatigue_impact": 0.05},
            "继续学习": {"base_gain": 0.4, "fatigue_impact": 0.1},
            "复习": {"base_gain": 0.25, "fatigue_impact": 0.03},
            "记忆学习": {"base_gain": 0.2, "fatigue_impact": 0.06},
            "问题解决": {"base_gain": 0.35, "fatigue_impact": 0.1},
            "创造性思考": {"base_gain": 0.4, "fatigue_impact": 0.12},
            "速度训练": {"base_gain": 0.3, "fatigue_impact": 0.15},
            "项目式学习": {"base_gain": 0.45, "fatigue_impact": 0.08}
        }

        # 理想状态
        self.ideal_state = {
            "name": "牛顿",
            "subject": "物理",
            "module": "力学",
            "topic": "牛顿第二定律",
            "level": 4.5
        }

        print("✅ 系统初始化完成")

    def _initialize_students(self) -> List[BlankStudent]:
        """初始化学生"""
        students = [
            BlankStudent("学生A", 17, "物理"),
            BlankStudent("学生B", 16, "英语"),
            BlankStudent("学生C", 18, "生物"),
            BlankStudent("学生D", 15, "数学"),
            BlankStudent("学生E", 17, "化学")
        ]

        # 设置初始主题
        for i, student in enumerate(students):
            if student.subject == "物理":
                student.module = "力学"
                student.topic = "牛顿第二定律"
            elif student.subject == "英语":
                student.module = "词汇"
                student.topic = "常见单词"
            elif student.subject == "生物":
                student.module = "细胞学"
                student.topic = "细胞分裂"
            elif student.subject == "数学":
                student.module = "代数"
                student.topic = "二次方程"
            elif student.subject == "化学":
                student.module = "无机化学"
                student.topic = "化学反应速率"

        print(f"👨‍🎓 已初始化 {len(students)} 名学生")
        return students

    # ======================
    # 显示功能模块
    # ======================

    def show_learning_position(self, student: BlankStudent):
        """显示学习位置"""
        print(f"\n📚 学生：{student.name}")
        print(f"   年龄：{student.age}岁 | 学科：{student.subject}")
        print(f"   模块：{student.module} | 知识点：{student.topic}")
        print(f"   综合水平：{student.level:.2f}")
        print(f"   专注度：{student.attention:.2f} | 疲劳度：{student.fatigue:.2f}")

        # 显示技能详情
        skills_dict = student.skills.__dict__
        print("   技能详情：")
        for skill, value in skills_dict.items():
            bar = self.viz.create_progress_bar(value, CONFIG["skill_max"], 10, False)
            print(f"     {skill:12}: {bar}")

    def compare_with_ideal(self, student: BlankStudent):
        """与理想轨迹对比"""
        print("\n" + "=" * 60)
        print("🎯 学习轨迹对齐对比")
        print("=" * 60)

        student_skills = student.skills.__dict__
        age = student.age

        similarity = self.newton_model.calculate_similarity(student_skills, age)
        equivalent_age = self.newton_model.get_equivalent_age(student_skills)

        print(f"\n🌟 【理想轨迹 - 牛顿】")
        print(f"   目标年龄：{age}岁")

        target = self.newton_model.get_target_at_age(age)
        print("   目标技能水平：")
        for skill, value in target.items():
            bar = self.viz.create_progress_bar(value, CONFIG["skill_max"], 10, False)
            print(f"     {skill:12}: {bar}")

        print(f"\n👨‍🎓 【学生当前轨迹 - {student.name}】")
        self.show_learning_position(student)

        # 计算差距
        gap_sum = 0
        for skill, target_value in target.items():
            if skill in student_skills:
                gap = target_value - student_skills[skill]
                gap_sum += abs(gap)

        print(f"\n📊 对比分析：")
        print(f"   与牛顿相似度：{similarity:.3f}")
        print(f"   等效牛顿年龄：{equivalent_age:.1f}岁")
        print(f"   综合差距：{gap_sum:.2f}")

        if similarity > 0.8:
            print("   🎉 优秀！接近理想轨迹")
        elif similarity > 0.6:
            print("   📈 良好！稳步前进中")
        elif similarity > 0.4:
            print("   📚 加油！需要更多努力")
        else:
            print("   ⚠️  需调整学习策略")

    # ======================
    # 传感器模拟模块
    # ======================

    def simulate_camera_signal(self) -> Tuple[float, float]:
        """模拟摄像头信号"""
        attention_signal = random.uniform(0.6, 0.95)
        emotion_signal = random.uniform(-0.2, 0.2)
        return attention_signal, emotion_signal

    def apply_camera_signal(self, student: BlankStudent):
        """应用摄像头信号"""
        attention_signal, emotion_signal = self.simulate_camera_signal()

        # 更新专注度
        student.attention = 0.7 * student.attention + 0.3 * attention_signal

        # 情绪波动影响
        student.fatigue += emotion_signal * 0.3

        # 边界检查
        student.attention = max(0, min(student.attention, 1))
        student.fatigue = max(0, min(student.fatigue, 1))

        print(f"📷 摄像头监测 -> 专注度: {student.attention:.2f} | 情绪影响: {emotion_signal:.2f}")

    # ======================
    # 学习过程模块
    # ======================

    def apply_learning_action(self, student: BlankStudent, action_id: str, session_id: str) -> Dict:
        """应用学习行为"""
        # 记录学习前状态
        level_before = student.level
        attention_before = student.attention
        fatigue_before = student.fatigue

        print(f"\n🎯 执行学习行为：{action_id}")

        if action_id == "rest":
            # 休息行为
            student.rest()
            print("💤 休息中... 专注度恢复，疲劳度降低")
        else:
            # 应用学习行为
            effects = self.learning_actions.apply_action(student, action_id)

            # 显示效果
            if effects:
                print("📈 技能提升效果：")
                for skill, effect in effects.items():
                    if effect > 0:
                        print(f"     {skill:12}: +{effect:.3f}")

        # 应用认知发展规律
        student.apply_cognitive_development()

        # 应用遗忘规律
        student.apply_forgetting()

        # 计算学习效率
        efficiency_score = self._calculate_efficiency_score(
            level_before, student.level,
            fatigue_before, student.fatigue,
            student.attention
        )

        # 记录学习历史
        student.learning_history.append({
            "session_id": session_id,
            "strategy": action_id,
            "level_before": level_before,
            "level_after": student.level,
            "attention_before": attention_before,
            "attention_after": student.attention,
            "fatigue_before": fatigue_before,
            "fatigue_after": student.fatigue,
            "timestamp": datetime.datetime.now().isoformat()
        })

        return {
            "level_before": level_before,
            "level_after": student.level,
            "attention_before": attention_before,
            "attention_after": student.attention,
            "fatigue_before": fatigue_before,
            "fatigue_after": student.fatigue,
            "efficiency_score": efficiency_score
        }

    def _calculate_efficiency_score(self, level_before: float, level_after: float,
                                    fatigue_before: float, fatigue_after: float,
                                    attention: float) -> float:
        """计算学习效率分数"""
        level_gain = level_after - level_before
        fatigue_change = fatigue_after - fatigue_before

        if fatigue_change <= 0:
            efficiency = level_gain * attention * 1.2
        else:
            efficiency = level_gain * attention * 0.8

        return max(0, efficiency)

    # ======================
    # 策略推荐模块
    # ======================

    def recommend_learning_strategy(self, student: BlankStudent) -> str:
        """推荐学习策略"""
        # 基于规则的推荐
        if student.fatigue > 0.7:
            return "rest"
        elif student.attention < 0.5:
            return "interactive"
        elif student.attention > 0.85 and student.fatigue < 0.3:
            return "continue"
        elif student.level < 2.5:
            return "lecture"
        elif 2.5 <= student.level <= 3.5:
            return "example"
        else:
            # 使用牛顿模型推荐
            return self.learning_actions.get_recommended_action(student, self.newton_model)

    def enhanced_strategy_recommendation(self, student: BlankStudent) -> str:
        """增强版策略推荐"""
        # 检查策略疲劳
        recent_history = student.learning_history[-3:] if student.learning_history else []

        if recent_history:
            strategies_used = [record.get("strategy", "未知") for record in recent_history]
            if len(set(strategies_used)) == 1 and len(strategies_used) >= 2:
                current_strategy = strategies_used[0]
                all_strategies = list(self.learning_actions.actions.keys())

                if current_strategy in all_strategies:
                    all_strategies.remove(current_strategy)

                if all_strategies:
                    new_strategy = random.choice(all_strategies)
                    print(f"🔄 检测到策略疲劳，更换策略：{current_strategy} → {new_strategy}")
                    return new_strategy

        # 使用基础推荐
        return self.recommend_learning_strategy(student)

    # ======================
    # 完整学习流程
    # ======================

    def enhanced_learning_process(self, student: BlankStudent, num_sessions: int = 5) -> BlankStudent:
        """增强学习过程"""
        print(f"\n{'=' * 60}")
        print(f"🚀 开始 {student.name} 的个性化学习旅程")
        print(f"📚 学科：{student.subject} | 初始水平：{student.level:.2f}")
        print(f"{'=' * 60}")

        # 保存初始状态
        initial_state = student.get_state()

        # 生成会话ID
        session_id = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 执行学习会话
        for session in range(1, num_sessions + 1):
            print(f"\n{'─' * 40}")
            print(f"📖 学习会话 {session}/{num_sessions}")
            print(f"{'─' * 40}")

            # 验证状态
            is_valid, message = student.validate()
            if not is_valid:
                print(f"⚠️ 状态异常：{message}")
                break

            # 显示当前状态
            self.show_learning_position(student)

            # 智能推荐策略
            strategy = self.enhanced_strategy_recommendation(student)
            print(f"\n🤖 智能推荐策略：{strategy}")

            # 应用学习行为
            record = self.apply_learning_action(student, strategy, f"{session_id}_{session}")

            # 应用摄像头信号
            self.apply_camera_signal(student)

            # 显示更新后的状态
            print(f"\n📊 更新后状态：")
            self.show_learning_position(student)

            # 保存学习记录
            if self.use_database and self.db:
                self.db.save_learning_record(
                    student.name, f"{session_id}_{session}", strategy,
                    record["level_before"], record["level_after"],
                    record["attention_before"], record["attention_after"],
                    record["fatigue_before"], record["fatigue_after"],
                    record["efficiency_score"]
                )

            # 每2次会话显示一次理想对比
            if session % 2 == 0 and student.subject == self.ideal_state["subject"]:
                self.compare_with_ideal(student)

            # 更新天数
            student.day += 1
            student.update_age()

        # 保存最终状态
        if self.use_database and self.db:
            self.db.save_student_state(student)

        # 生成评估报告
        self._generate_learning_report(student, initial_state, num_sessions)

        return student

    def _generate_learning_report(self, student: BlankStudent, initial_state: Dict, num_sessions: int):
        """生成学习报告"""
        print(f"\n{'=' * 60}")
        print("📊 学习效果详细报告")
        print(f"{'=' * 60}")

        # 计算各项指标
        level_improvement = student.level - initial_state["level"]
        attention_change = student.attention - initial_state["attention"]
        fatigue_change = student.fatigue - initial_state["fatigue"]

        hourly_gain = level_improvement / (num_sessions * 0.5) if num_sessions > 0 else 0

        # 构建报告表格
        report_table = [
            {"项目": "学生姓名", "值": student.name},
            {"项目": "学习学科", "值": student.subject},
            {"项目": "学习会话", "值": num_sessions},
            {"项目": "学习时长(小时)", "值": f"{num_sessions * 0.5:.1f}"},
            {"项目": "水平提升", "值": f"{level_improvement:+.3f}"},
            {"项目": "每小时学习率", "值": f"{hourly_gain:.3f}"},
            {"项目": "专注度变化", "值": f"{attention_change:+.3f}"},
            {"项目": "疲劳度变化", "值": f"{fatigue_change:+.3f}"},
            {"项目": "最终水平", "值": f"{student.level:.2f}"},
            {"项目": "最终专注度", "值": f"{student.attention:.2f}"},
            {"项目": "最终疲劳度", "值": f"{student.fatigue:.2f}"},
        ]

        print(self.viz.create_simple_table(report_table))

        # 技能变化详情
        initial_skills = initial_state["skills"]
        current_skills = student.skills.__dict__

        print(f"\n📈 技能变化详情：")
        skill_table = []
        for skill in initial_skills:
            initial_val = initial_skills[skill]
            current_val = current_skills.get(skill, 0)
            change = current_val - initial_val
            change_percent = (change / initial_val * 100) if initial_val > 0 else 0

            skill_table.append({
                "技能": skill,
                "初始": f"{initial_val:.2f}",
                "当前": f"{current_val:.2f}",
                "变化": f"{change:+.2f}",
                "变化率": f"{change_percent:+.1f}%"
            })

        print(self.viz.create_simple_table(skill_table, ["技能", "初始", "当前", "变化", "变化率"]))

    # ======================
    # 可视化模块
    # ======================

    def visualize_learning_progress(self, student_name: str):
        """可视化学习进度"""
        # 获取学习历史
        if self.use_database and self.db:
            history = self.db.get_student_history(student_name)
        else:
            history = []
            for student in self.students:
                if student.name == student_name:
                    history = student.learning_history
                    break

        if not history:
            print(f"⚠️ 没有找到 {student_name} 的学习历史")
            return

        print(f"\n{'=' * 60}")
        print(f"📈 {student_name} 学习进度分析")
        print(f"{'=' * 60}")

        # 提取数据
        sessions = list(range(1, len(history) + 1))
        levels_after = [record.get("level_after", 0) for record in history]
        strategies = [record.get("strategy", "未知") for record in history]

        # 显示水平变化趋势
        print("\n1️⃣ 学习水平变化趋势:")
        if len(levels_after) > 1:
            print(self.viz.create_line_chart(levels_after, width=40, height=8))
        else:
            print("  数据不足生成趋势图")

        # 显示策略使用统计
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

    # ======================
    # 系统管理模块
    # ======================

    def run_demo(self):
        """运行演示"""
        print("\n" + "=" * 70)
        print("🤖 增强智能学习系统 - 演示模式")
        print("=" * 70)
        print(f"📅 系统时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"👨‍🎓 学生数量：{len(self.students)}")
        print("=" * 70)

        # 为每个学生运行学习过程
        for i, student in enumerate(self.students):
            print(f"\n{'#' * 70}")
            print(f"👨‍🎓 学生 {i + 1}/{len(self.students)}: {student.name}")
            print(f"{'#' * 70}")

            # 运行学习过程
            num_sessions = random.randint(3, 6)
            self.enhanced_learning_process(student, num_sessions)

            # 询问是否查看学习进度
            if input(f"\n是否查看 {student.name} 的学习进度图表？(y/n): ").lower() == 'y':
                self.visualize_learning_progress(student.name)

        # 保存数据
        self.save_all_data()

        # 关闭数据库
        if self.use_database and self.db:
            self.db.close()

        print(f"\n{'=' * 70}")
        print("🎉 学习系统运行完成！")
        print(f"{'=' * 70}")

    def save_all_data(self):
        """保存所有数据"""
        print("\n💾 正在保存系统数据...")

        try:
            # 保存学生状态
            students_dict = [s.get_state() for s in self.students]
            with open("enhanced_students_state.json", "w", encoding='utf-8') as f:
                json.dump(students_dict, f, ensure_ascii=False, indent=2)
            print("✅ 学生状态已保存到 enhanced_students_state.json")

        except Exception as e:
            print(f"❌ 保存数据时出错: {e}")

    def show_system_info(self):
        """显示系统信息"""
        print("\n" + "=" * 60)
        print("📋 系统信息")
        print("=" * 60)
        print(f"系统版本: 3.0 (增强整合版)")
        print(f"学生数量: {len(self.students)}")
        print(f"学习策略: {len(self.learning_actions.actions)} 种")
        print(f"数据库状态: {'已启用' if self.use_database else '已禁用'}")
        print(f"认知技能维度: 6 种")
        print("=" * 60)

        # 显示学生列表
        print("\n👨‍🎓 学生列表:")
        student_table = []
        for i, student in enumerate(self.students):
            student_table.append({
                "序号": i + 1,
                "姓名": student.name,
                "年龄": student.age,
                "学科": student.subject,
                "水平": f"{student.level:.2f}",
                "专注度": f"{student.attention:.2f}",
                "疲劳度": f"{student.fatigue:.2f}"
            })

        print(self.viz.create_simple_table(student_table, ["序号", "姓名", "年龄", "学科", "水平", "专注度", "疲劳度"]))

        # 显示学习策略
        print("\n🎯 可用学习策略:")
        strategy_table = []
        actions = self.learning_actions.actions
        for i, (action_id, action_info) in enumerate(actions.items()):
            if i < 10:  # 只显示前10个策略
                strategy_table.append({
                    "序号": i + 1,
                    "策略": action_info["name"],
                    "描述": action_info["description"][:20] + "..."
                })

        print(self.viz.create_simple_table(strategy_table, ["序号", "策略", "描述"]))


# ======================
# 主程序入口
# ======================

def main():
    """主函数"""
    print("🎓 欢迎使用增强智能学习系统")
    print("版本: 3.0 (整合增强版)")
    print("=" * 50)

    # 创建学习系统实例
    learning_system = EnhancedLearningSystem(use_database=True)

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
            print(f"{i + 1}. {student.name} ({student.subject}, {student.age}岁)")

        try:
            student_choice = int(input("\n请输入学生编号: ")) - 1
            if 0 <= student_choice < len(learning_system.students):
                sessions = input("请输入学习会话数量 (默认5): ").strip()
                num_sessions = int(sessions) if sessions.isdigit() else 5
                student = learning_system.students[student_choice]
                learning_system.enhanced_learning_process(student, num_sessions)
                learning_system.visualize_learning_progress(student.name)
            else:
                print("❌ 无效的学生编号")
        except ValueError:
            print("❌ 请输入有效的数字")
    elif choice == "3":
        # 仅显示系统信息
        learning_system.show_system_info()
        print("\nℹ️ 系统信息显示完成")
    elif choice == "4":
        print("👋 感谢使用，再见！")
        return
    else:
        print("❌ 无效的选择")

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
    finally:
        print("\n🎓 增强智能学习系统已关闭")