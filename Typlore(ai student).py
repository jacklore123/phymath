"""
智能自适应学习系统

新增功能：
1. 完整的蒙特卡洛树搜索（MCTS）路径规划
2. 认知先验知识集成模块
3. 多模态数据采集与模拟
4. 动态榜样对齐算法
5. 实验验证与压缩模拟
6. 游戏化学习机制
7. 进度条可视化系统（重点恢复）
"""

import random
import json
import datetime
import math
import sqlite3
import time
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any, Set
from collections import defaultdict, deque
from enum import Enum
from copy import deepcopy
CONFIG = {
    "total_days": 4380,  # K12阶段总天数
    "age_start": 6,  # 起始年龄
    "age_end": 18,  # 结束年龄
    "skill_min": 0.0,  # 技能最小值
    "skill_max": 5.0,  # 技能最大值
    "learning_rate": 0.1,
    "forgetting_rate": 0.001,
    "random_factor": 0.01,

    # 新增配置
    "mcts_simulations": 100,  # MCTS模拟次数
    "compression_ratio": 1000,  # 时间压缩比例（18年->分钟级）
    "token_per_day": 10000,  # 每天Token消耗量
    "cognitive_prior_weight": 0.3,  # 认知先验知识权重

    # 进度条配置
    "progress_bar_width": 50,  # 进度条宽度
    "show_detailed_progress": True,  # 显示详细进度
}


# ======================
# 进度条显示类
# ======================
class ProgressVisualizer:
    """进度条可视化系统 - 恢复原版进度条功能并增强"""

    def __init__(self):
        self.progress_history = []

    def create_progress_bar(self, value: float, max_value: float = 5.0,
                            bar_length: int = 50, show_percentage: bool = True,
                            show_fraction: bool = True) -> str:
        """创建进度条"""
        normalized_value = max(0, min(value, max_value))
        filled_length = int(normalized_value / max_value * bar_length)
        empty_length = bar_length - filled_length

        bar = "█" * filled_length + "░" * empty_length

        result = f"[{bar}]"

        if show_fraction:
            result += f" {normalized_value:.2f}/{max_value}"

        if show_percentage:
            percentage = (normalized_value / max_value) * 100
            result += f" ({percentage:.1f}%)"

        return result

    def create_skill_progress_bar(self, skill_name: str, current_value: float,
                                  target_value: float, bar_length: int = 30) -> str:
        """创建技能进度条"""
        if target_value <= 0:
            return f"{skill_name:12}: 目标值为零"

        percentage = min(100, (current_value / target_value) * 100)
        filled_length = int(percentage / 100 * bar_length)
        empty_length = bar_length - filled_length

        bar = "█" * filled_length + "░" * empty_length

        status = ""
        if percentage >= 100:
            status = " ✅"
        elif percentage >= 80:
            status = " 📈"
        elif percentage >= 60:
            status = " ⏳"
        else:
            status = " ⚠️"

        return f"{skill_name:12}: [{bar}] {current_value:.2f}/{target_value:.2f} ({percentage:.1f}%){status}"

    def create_milestone_progress(self, current_age: float, equivalent_age: float,
                                  max_age: float = 18.0) -> Dict:
        """创建里程碑进度（与牛顿模型对比）"""
        # 计算年龄进度
        current_progress = min(100, (current_age / max_age) * 100)
        equivalent_progress = min(100, (equivalent_age / max_age) * 100)

        # 计算领先/落后
        age_difference = equivalent_age - current_age

        if age_difference > 1:
            status = "超前"
            status_emoji = "🚀"
        elif age_difference > 0.5:
            status = "领先"
            status_emoji = "📈"
        elif abs(age_difference) <= 0.5:
            status = "同步"
            status_emoji = "✅"
        elif age_difference < -1:
            status = "落后"
            status_emoji = "⚠️"
        else:
            status = "稍慢"
            status_emoji = "⏳"

        return {
            "current_age": current_age,
            "equivalent_age": equivalent_age,
            "current_progress": current_progress,
            "equivalent_progress": equivalent_progress,
            "age_difference": age_difference,
            "status": status,
            "status_emoji": status_emoji,
            "message": f"相当于牛顿{equivalent_age:.1f}岁的水平"
        }

    def display_milestone_progress(self, milestone: Dict, bar_length: int = 40):
        """显示里程碑进度"""
        print(f"\n🎯 学习里程碑进度")
        print(f"  实际年龄: {milestone['current_age']:.1f}岁")
        print(f"  等效牛顿年龄: {milestone['equivalent_age']:.1f}岁")
        print(f"  状态: {milestone['status_emoji']} {milestone['status']} ({milestone['age_difference']:+.1f}岁)")
        print(f"  {milestone['message']}")

        # 显示双进度条
        current_bar = self.create_progress_bar(
            milestone['current_progress'], 100, bar_length, show_percentage=False
        )
        equivalent_bar = self.create_progress_bar(
            milestone['equivalent_progress'], 100, bar_length, show_percentage=False
        )

        print(f"  实际进度: {current_bar} {milestone['current_progress']:.1f}%")
        print(f"  等效进度: {equivalent_bar} {milestone['equivalent_progress']:.1f}%")

    def create_learning_journey_map(self, student_history: List[Dict],
                                    target_trajectory: Dict) -> str:
        """创建学习旅程地图（可视化进度）"""
        if not student_history:
            return "暂无学习历史"

        # 提取关键数据点
        ages = []
        levels = []
        equivalent_ages = []

        for record in student_history[-10:]:  # 最近10次记录
            if 'age' in record and 'level' in record:
                ages.append(record['age'])
                levels.append(record['level'])
            if 'equivalent_age' in record:
                equivalent_ages.append(record['equivalent_age'])

        if not ages:
            return "数据不足生成旅程地图"

        # 创建简单的ASCII地图
        map_height = 10
        map_width = 50

        # 初始化地图网格
        grid = [[' ' for _ in range(map_width)] for _ in range(map_height)]

        # 计算归一化坐标
        min_age = min(ages)
        max_age = max(ages)
        age_range = max_age - min_age if max_age > min_age else 1

        min_level = min(levels) if levels else 0
        max_level = max(levels) if levels else 1
        level_range = max_level - min_level if max_level > min_level else 1

        # 绘制学生轨迹
        for i, (age, level) in enumerate(zip(ages, levels)):
            x = int((age - min_age) / age_range * (map_width - 1))
            y = int((1 - (level - min_level) / level_range) * (map_height - 1))

            if 0 <= x < map_width and 0 <= y < map_height:
                grid[y][x] = '●'

        # 添加目标轨迹点
        if target_trajectory:
            target_ages = list(target_trajectory.keys())
            for age in target_ages:
                if isinstance(age, (int, float)) and age in target_trajectory:
                    target_level = target_trajectory[age].get('knowledge', 0) / 5.0
                    x = int((age - min_age) / age_range * (map_width - 1))
                    y = int((1 - (target_level - min_level) / level_range) * (map_height - 1))

                    if 0 <= x < map_width and 0 <= y < map_height and grid[y][x] == ' ':
                        grid[y][x] = '★'

        # 构建地图字符串
        map_lines = []
        for row in grid:
            map_lines.append(''.join(row))

        # 添加图例
        map_lines.append(f"\n图例: ● 你的轨迹 | ★ 牛顿目标 | 纵轴: 知识水平 | 横轴: 年龄")
        map_lines.append(f"年龄范围: {min_age:.1f} - {max_age:.1f}岁")
        map_lines.append(f"水平范围: {min_level:.2f} - {max_level:.2f}")

        return '\n'.join(map_lines)


# ======================
# 认知状态枚举（基于EduAgent论文）
# ======================
class CognitiveState(Enum):
    WORKLOAD = "workload"  # 认知负荷
    CURIOSITY = "curiosity"  # 好奇心
    FOCUS = "focus"  # 有效专注
    FOLLOWING = "following"  # 课程跟随
    ENGAGEMENT = "engagement"  # 参与度
    CONFUSION = "confusion"  # 困惑度


class MotorBehavior(Enum):
    MOUSE_MOVEMENT = "mouse_movement"  # 鼠标移动
    CLICK_PATTERN = "click_pattern"  # 点击模式
    SCROLL_BEHAVIOR = "scroll_behavior"  # 滚动行为


class GazeBehavior(Enum):
    FIXATION = "fixation"  # 注视点
    SACCADE = "saccade"  # 扫视
    BLINK_RATE = "blink_rate"  # 眨眼频率
    PUPIL_SIZE = "pupil_size"  # 瞳孔大小


# ======================
# 增强的数据类型定义
# ======================
@dataclass
class LearningGoal:
    """学习目标（增强版）"""
    module: str
    topic: str
    target_level: float
    current_difficulty: float = 4.5
    priority: float = 1.0  # 目标优先级

    def to_dict(self) -> Dict:
        return {
            "module": self.module,
            "topic": self.topic,
            "level": self.target_level,
            "difficulty": self.current_difficulty,
            "priority": self.priority
        }


@dataclass
class CognitiveProfile:
    """认知特征档案（基于EduAgent 705数据集）"""
    # 人口统计特征
    age_group: int = 0  # 0:18-24, 1:25-31, 2:32-38, 3:>39
    gender: int = 0  # 0:女, 1:男, 2:其他
    major: int = 0  # 专业类别
    education_level: int = 0  # 教育水平

    # 学习特征
    learning_attitude: float = 0.5  # 学习态度 (0-1)
    exam_performance: float = 0.5  # 考试表现 (0-1)
    focus_ability: float = 0.5  # 专注能力 (0-1)
    curiosity_level: float = 0.5  # 好奇心水平 (0-1)
    course_interest: float = 0.5  # 课程兴趣 (0-1)
    prior_knowledge: float = 0.5  # 先验知识 (0-1)
    compliance: float = 0.5  # 遵从性 (0-1)
    intelligence: float = 0.5  # 智力水平 (0-1)
    family_background: float = 0.5  # 家庭背景 (0-1)

    def get_aggregated_score(self) -> float:
        """计算聚合角色分数"""
        features = [
            self.learning_attitude, self.exam_performance,
            self.focus_ability, self.curiosity_level,
            self.course_interest, self.prior_knowledge,
            self.compliance, self.intelligence,
            self.family_background
        ]
        return sum(features) / len(features)


@dataclass
class CognitiveSkills:
    """认知技能维度（增强版）"""
    knowledge: float = 0.5  # 知识积累
    abstraction: float = 0.2  # 抽象思维
    reasoning: float = 0.3  # 逻辑推理
    speed: float = 0.2  # 思维速度
    creativity: float = 0.1  # 创造力
    memory: float = 0.4  # 记忆力

    # 新增认知状态（基于EduAgent）
    workload: float = 0.3  # 认知负荷
    curiosity: float = 0.4  # 好奇心
    focus: float = 0.6  # 有效专注
    following: float = 0.5  # 课程跟随
    engagement: float = 0.7  # 参与度
    confusion: float = 0.2  # 困惑度


# ======================
# 多模态数据采集模块
# ======================
class MultimodalDataCollector:
    """多模态数据采集器（模拟版）"""

    def __init__(self):
        self.gaze_history = []  # 注视轨迹
        self.motor_history = []  # 运动行为
        self.cognitive_history = []  # 认知状态历史

    def simulate_gaze_data(self, attention_level: float) -> Dict:
        """模拟注视数据"""
        # 基于注意力和认知科学原理模拟
        fixation_duration = random.uniform(0.2, 0.4) * attention_level
        saccade_amplitude = random.uniform(2.0, 8.0) * (1 - attention_level)
        blink_rate = 15 - attention_level * 10  # 专注时眨眼减少
        pupil_size = 3.0 + attention_level * 1.5  # 专注时瞳孔放大

        gaze_data = {
            "fixation_duration": fixation_duration,
            "saccade_amplitude": saccade_amplitude,
            "blink_rate": blink_rate,
            "pupil_size": pupil_size,
            "timestamp": datetime.datetime.now().isoformat()
        }

        self.gaze_history.append(gaze_data)
        return gaze_data

    def simulate_motor_data(self, engagement: float) -> Dict:
        """模拟运动数据（鼠标行为）"""
        # 鼠标移动速度和点击频率反映参与度
        movement_speed = random.uniform(2.0, 10.0) * engagement
        click_frequency = random.uniform(0.5, 3.0) * engagement
        scroll_activity = random.uniform(0.1, 2.0) * engagement

        motor_data = {
            "movement_speed": movement_speed,
            "click_frequency": click_frequency,
            "scroll_activity": scroll_activity,
            "timestamp": datetime.datetime.now().isoformat()
        }

        self.motor_history.append(motor_data)
        return motor_data

    def record_cognitive_state(self, cognitive_skills: CognitiveSkills) -> Dict:
        """记录认知状态"""
        cognitive_data = {
            "workload": cognitive_skills.workload,
            "curiosity": cognitive_skills.curiosity,
            "focus": cognitive_skills.focus,
            "following": cognitive_skills.following,
            "engagement": cognitive_skills.engagement,
            "confusion": cognitive_skills.confusion,
            "timestamp": datetime.datetime.now().isoformat()
        }

        self.cognitive_history.append(cognitive_data)
        return cognitive_data

    def get_behavior_correlation(self) -> Dict:
        """计算行为相关性（基于EduAgent论文）"""
        if len(self.cognitive_history) < 2:
            return {}

        # 计算注视与认知状态的相关性
        gaze_focus = [g.get("fixation_duration", 0) for g in self.gaze_history[-10:]]
        cognitive_focus = [c.get("focus", 0) for c in self.cognitive_history[-10:]]

        if len(gaze_focus) > 1 and len(cognitive_focus) > 1:
            try:
                correlation = np.corrcoef(gaze_focus, cognitive_focus)[0, 1]
            except:
                correlation = 0
        else:
            correlation = 0

        return {
            "gaze_focus_correlation": correlation,
            "data_points": len(self.gaze_history)
        }


# ======================
# 牛顿榜样模型（含进度条功能）
# ======================
class NewtonRoleModel:
    """牛顿榜样模型 - 目标学习路径（增强进度条功能）"""

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
        self.progress_visualizer = ProgressVisualizer()

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

        # 添加插值，得到更精确的等效年龄
        if best_similarity > 0.5 and best_age < 18:
            next_age = best_age + 1
            next_similarity = self.calculate_similarity(student_skills, next_age)

            # 线性插值
            weight = (best_similarity - 0.5) / (
                        best_similarity + next_similarity - 1.0) if best_similarity + next_similarity > 1.0 else 0
            equivalent_age = best_age + weight
        else:
            equivalent_age = best_age

        return equivalent_age

    def display_progress_comparison(self, student_skills: Dict[str, float],
                                    student_age: int, student_name: str = "学生"):
        """显示与牛顿的进度对比（原版进度条功能）"""
        equivalent_age = self.get_equivalent_age(student_skills)
        similarity = self.calculate_similarity(student_skills, student_age)

        # 创建里程碑进度
        milestone = self.progress_visualizer.create_milestone_progress(
            student_age, equivalent_age
        )

        print(f"\n{'=' * 60}")
        print("🎯 学习轨迹对齐对比")
        print(f"{'=' * 60}")

        print(f"\n🌟 【理想轨迹 - 牛顿】")
        print(f"   目标年龄: {student_age}岁")

        target = self.get_target_at_age(student_age)
        print("   目标技能水平:")
        for skill, value in target.items():
            bar = self.progress_visualizer.create_progress_bar(
                value, CONFIG["skill_max"], 15, show_percentage=False
            )
            print(f"     {skill:12}: {bar}")

        print(f"\n👨‍🎓 【学生当前轨迹 - {student_name}】")
        print(f"   年龄: {student_age}岁 | 综合水平: {sum(student_skills.values()) / len(student_skills):.2f}")
        print(f"   技能详情:")
        for skill, value in student_skills.items():
            bar = self.progress_visualizer.create_progress_bar(
                value, CONFIG["skill_max"], 15, show_percentage=False
            )
            print(f"     {skill:12}: {bar}")

        # 显示进度条对比
        self.progress_visualizer.display_milestone_progress(milestone)

        # 显示技能差距
        print(f"\n📊 对比分析:")
        print(f"   与牛顿相似度: {similarity:.3f}")

        gap_sum = 0
        for skill in student_skills:
            if skill in target:
                gap = target[skill] - student_skills[skill]
                gap_sum += abs(gap)

        print(f"   综合差距: {gap_sum:.2f}")

        if similarity > 0.8:
            print("   🎉 优秀！接近理想轨迹")
        elif similarity > 0.6:
            print("   📈 良好！稳步前进中")
        elif similarity > 0.4:
            print("   📚 加油！需要更多努力")
        else:
            print("   ⚠️  需调整学习策略")

        return {
            "similarity": similarity,
            "equivalent_age": equivalent_age,
            "milestone": milestone,
            "skill_gaps": gap_sum
        }


# ======================
# 蒙特卡洛树搜索（MCTS）实现
# ======================
class MCTSNode:
    """MCTS节点"""

    def __init__(self, state: Dict, parent=None, action: str = None):
        self.state = state  # 学生状态快照
        self.parent = parent  # 父节点
        self.action = action  # 到达此节点的动作
        self.children = []  # 子节点列表
        self.visits = 0  # 访问次数
        self.value = 0.0  # 累计价值
        self.untried_actions = []  # 未尝试的动作

    def uct_score(self, exploration_param: float = 1.41) -> float:
        """计算UCT分数"""
        if self.visits == 0:
            return float('inf')
        exploitation = self.value / self.visits
        exploration = exploration_param * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def is_fully_expanded(self) -> bool:
        """是否完全扩展"""
        return len(self.untried_actions) == 0

    def best_child(self) -> 'MCTSNode':
        """选择最佳子节点"""
        return max(self.children, key=lambda c: c.visits)


class MCTSPathPlanner:
    """MCTS路径规划器"""

    def __init__(self, actions_system, newton_model, exploration_param: float = 1.41):
        self.actions = actions_system
        self.newton = newton_model
        self.exploration_param = exploration_param
        self.root = None

    def search(self, student_state: Dict, simulations: int = 100) -> Dict:
        """搜索最优学习路径"""
        self.root = MCTSNode(student_state)

        for _ in range(simulations):
            # 选择阶段
            node = self._select(self.root)

            # 扩展阶段
            if not node.is_fully_expanded():
                node = self._expand(node)

            # 模拟阶段
            reward = self._simulate(node)

            # 回溯阶段
            self._backpropagate(node, reward)

        # 返回最优动作序列
        best_path = self._extract_best_path()
        return best_path

    def _select(self, node: MCTSNode) -> MCTSNode:
        """选择阶段：使用UCT算法选择节点"""
        while node.children:
            if not node.is_fully_expanded():
                return node

            # 选择UCT分数最高的子节点
            node = max(node.children, key=lambda c: c.uct_score(self.exploration_param))

        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """扩展阶段：扩展新节点"""
        if not node.untried_actions:
            # 初始化未尝试动作
            student_fatigue = node.state.get("fatigue", 0.5)
            node.untried_actions = self.actions.get_available_actions(student_fatigue)

        if node.untried_actions:
            # 选择一个未尝试的动作
            action = random.choice(node.untried_actions)
            node.untried_actions.remove(action)

            # 创建新状态
            new_state = self._apply_action_to_state(node.state, action)

            # 创建子节点
            child = MCTSNode(new_state, parent=node, action=action)
            node.children.append(child)

            return child

        return node

    def _simulate(self, node: MCTSNode, max_steps: int = 10) -> float:
        """模拟阶段：随机模拟学习过程"""
        simulated_state = deepcopy(node.state)
        total_reward = 0.0

        for step in range(max_steps):
            # 随机选择动作
            available_actions = self.actions.get_available_actions(
                simulated_state.get("fatigue", 0.5)
            )
            if not available_actions:
                break

            action = random.choice(available_actions)

            # 应用动作
            simulated_state = self._apply_action_to_state(simulated_state, action)

            # 计算即时奖励
            reward = self._calculate_reward(simulated_state, step)
            total_reward += reward * (0.9 ** step)  # 折扣因子

        return total_reward / max_steps if max_steps > 0 else 0

    def _backpropagate(self, node: MCTSNode, reward: float):
        """回溯阶段：更新节点统计"""
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def _apply_action_to_state(self, state: Dict, action: str) -> Dict:
        """将动作应用到状态（简化版）"""
        new_state = deepcopy(state)

        # 根据动作更新状态
        action_effects = self.actions.actions.get(action, {})
        effects = action_effects.get("effects", {})

        for skill, effect in effects.items():
            if skill in new_state.get("skills", {}):
                new_state["skills"][skill] += effect * random.uniform(0.8, 1.2)
                new_state["skills"][skill] = max(CONFIG["skill_min"],
                                                 min(CONFIG["skill_max"],
                                                     new_state["skills"][skill]))

        # 更新疲劳度
        if action == "rest":
            new_state["fatigue"] = max(0, new_state.get("fatigue", 0.5) - 0.3)
        else:
            new_state["fatigue"] = min(1, new_state.get("fatigue", 0.5) + 0.1)

        return new_state

    def _calculate_reward(self, state: Dict, step: int) -> float:
        """计算奖励函数"""
        skills = state.get("skills", {})

        # 1. 技能增长奖励
        skill_reward = sum(skills.values()) / len(skills) if skills else 0

        # 2. 牛顿对齐奖励
        similarity = self.newton.calculate_similarity(skills, state.get("age", 12))

        # 3. 疲劳惩罚
        fatigue_penalty = -state.get("fatigue", 0.5) * 0.2

        # 组合奖励
        total_reward = skill_reward * 0.4 + similarity * 0.5 + fatigue_penalty

        return max(0, total_reward)

    def _extract_best_path(self, horizon: int = 5) -> Dict:
        """提取最优路径"""
        if not self.root or not self.root.children:
            return {"actions": [], "expected_reward": 0}

        # 选择访问次数最多的动作序列
        path = []
        node = self.root
        expected_reward = 0

        for _ in range(horizon):
            if not node.children:
                break

            # 选择最佳子节点
            best_child = node.best_child()
            path.append(best_child.action)
            expected_reward = best_child.value / max(best_child.visits, 1)
            node = best_child

        return {
            "actions": path,
            "expected_reward": expected_reward,
            "total_simulations": self.root.visits
        }


# ======================
# 学习行为系统（原版功能恢复）
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

    def apply_action(self, student, action_id: str) -> Dict[str, float]:
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
        if hasattr(student, 'calculate_fatigue'):
            student.calculate_fatigue(action["fatigue"])
        elif hasattr(student, 'fatigue'):
            # 简单更新疲劳度
            student.fatigue = min(1.0, student.fatigue + action["fatigue"] * 0.1)

        # 更新综合水平
        if hasattr(student, '_update_level'):
            student._update_level()

        return effects

    def get_recommended_action(self, student, newton_model) -> str:
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
# 增强版学生体类（含进度记录）
# ======================
class EnhancedStudent:
    """增强版学生体（集成认知先验和多模态数据）"""

    def __init__(self, name: str = "default", age: int = 6,
                 subject: str = "物理", cognitive_profile: CognitiveProfile = None):
        self.name = name
        self.age = age
        self.subject = subject
        self.module = ""
        self.topic = ""
        self.day = 0

        # 认知技能和状态
        self.skills = CognitiveSkills()

        # 认知特征档案
        self.cognitive_profile = cognitive_profile or CognitiveProfile()

        # 学习状态
        self.level = 0.5
        self.attention = 0.8
        self.fatigue = 0.2
        self.learning_history = []  # 学习历史记录
        self.progress_history = []  # 进度历史记录
        self.last_updated = datetime.datetime.now().isoformat()

        # 认知发展曲线
        self.cognitive_development_curve = self._init_development_curve()

        # 多模态数据采集
        self.data_collector = MultimodalDataCollector()

        # 进度可视化
        self.progress_viz = ProgressVisualizer()

        # 学习路径记忆
        self.path_memory = deque(maxlen=100)

    def _init_development_curve(self) -> Dict[str, List[float]]:
        """初始化认知发展曲线（增强版）"""
        curves = {
            "knowledge": [0.3, 0.7, 0.1],
            "abstraction": [0.1, 0.5, 0.3],
            "reasoning": [0.2, 0.6, 0.2],
            "speed": [0.2, 0.5, 0.3],
            "creativity": [0.15, 0.55, 0.25],
            "memory": [0.25, 0.65, 0.2],
            # 认知状态发展
            "workload": [0.25, 0.6, 0.15],
            "curiosity": [0.35, 0.45, 0.3],
            "focus": [0.2, 0.5, 0.3],
            "following": [0.3, 0.55, 0.2],
            "engagement": [0.4, 0.5, 0.2],
            "confusion": [0.3, 0.4, 0.2]
        }
        return curves

    def update_age(self):
        """更新年龄"""
        if self.day % 365 == 0 and self.day > 0:
            self.age += 1

    def apply_cognitive_development(self):
        """应用认知发展规律（增强版）"""
        skills_dict = self.skills.__dict__

        for skill, curve in self.cognitive_development_curve.items():
            if skill in skills_dict:
                age_factor = self.age / 18.0
                k = curve[0] * 10
                m = curve[1]

                # S型发展曲线
                development_gain = 1 / (1 + math.exp(-k * (age_factor - m)))
                development_gain = min(development_gain, curve[2])

                # 加入个体差异
                if hasattr(self.cognitive_profile, 'intelligence'):
                    development_gain *= (0.8 + self.cognitive_profile.intelligence * 0.4)

                current_value = getattr(self.skills, skill)
                new_value = current_value + development_gain * CONFIG["learning_rate"]
                new_value = min(new_value, CONFIG["skill_max"])
                setattr(self.skills, skill, new_value)

        self._update_level()

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

    def collect_multimodal_data(self):
        """收集多模态数据"""
        # 模拟注视数据
        gaze_data = self.data_collector.simulate_gaze_data(self.attention)

        # 模拟运动数据
        motor_data = self.data_collector.simulate_motor_data(self.skills.engagement)

        # 记录认知状态
        cognitive_data = self.data_collector.record_cognitive_state(self.skills)

        return {
            "gaze": gaze_data,
            "motor": motor_data,
            "cognitive": cognitive_data
        }

    def update_cognitive_states(self, action_intensity: float):
        """更新认知状态（基于EduAgent原理）"""
        # 认知状态间的相互影响
        self.skills.focus = max(0, min(1,
                                       self.skills.focus * 0.8 + self.attention * 0.2 - self.fatigue * 0.1))

        self.skills.engagement = max(0, min(1,
                                            self.skills.engagement * 0.7 + action_intensity * 0.3 - self.skills.workload * 0.2))

        self.skills.workload = max(0, min(1,
                                          self.skills.workload * 0.6 + action_intensity * 0.4))

        self.skills.confusion = max(0, min(1,
                                           self.skills.confusion * 0.7 + random.uniform(-0.1, 0.1)))

    def get_state_for_mcts(self) -> Dict:
        """获取MCTS所需的状态表示"""
        skills_dict = self.skills.__dict__

        return {
            "name": self.name,
            "age": self.age,
            "skills": skills_dict,
            "level": self.level,
            "attention": self.attention,
            "fatigue": self.fatigue,
            "cognitive_profile": asdict(self.cognitive_profile),
            "day": self.day
        }

    def get_state(self) -> Dict:
        """获取当前状态（原版兼容）"""
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

    def record_progress(self, newton_model, session_id: str = ""):
        """记录进度（用于进度条）"""
        student_skills = {
            "knowledge": self.skills.knowledge,
            "abstraction": self.skills.abstraction,
            "reasoning": self.skills.reasoning,
            "speed": self.skills.speed,
            "creativity": self.skills.creativity,
            "memory": self.skills.memory
        }

        equivalent_age = newton_model.get_equivalent_age(student_skills)
        similarity = newton_model.calculate_similarity(student_skills, self.age)

        progress_record = {
            "session_id": session_id,
            "age": self.age,
            "level": self.level,
            "equivalent_age": equivalent_age,
            "similarity": similarity,
            "timestamp": datetime.datetime.now().isoformat(),
            "skills": student_skills
        }

        self.progress_history.append(progress_record)
        return progress_record

    def show_learning_position(self, progress_viz: ProgressVisualizer):
        """显示学习位置（原版功能）"""
        print(f"\n📚 学生：{self.name}")
        print(f"   年龄：{self.age}岁 | 学科：{self.subject}")
        print(f"   模块：{self.module} | 知识点：{self.topic}")
        print(f"   综合水平：{self.level:.2f}")
        print(f"   专注度：{self.attention:.2f} | 疲劳度：{self.fatigue:.2f}")

        # 显示技能详情
        skills_dict = {
            "knowledge": self.skills.knowledge,
            "abstraction": self.skills.abstraction,
            "reasoning": self.skills.reasoning,
            "speed": self.skills.speed,
            "creativity": self.skills.creativity,
            "memory": self.skills.memory
        }

        print("   技能详情：")
        for skill, value in skills_dict.items():
            bar = progress_viz.create_progress_bar(value, CONFIG["skill_max"], 10, False)
            print(f"     {skill:12}: {bar}")

    def _update_level(self):
        """更新综合水平（加权平均）"""
        skills_dict = {
            "knowledge": self.skills.knowledge,
            "abstraction": self.skills.abstraction,
            "reasoning": self.skills.reasoning,
            "speed": self.skills.speed,
            "creativity": self.skills.creativity,
            "memory": self.skills.memory
        }

        # 给不同技能不同权重
        weights = {
            "knowledge": 0.25,
            "reasoning": 0.20,
            "abstraction": 0.15,
            "memory": 0.15,
            "creativity": 0.10,
            "speed": 0.10,
            "focus": 0.05
        }

        total = 0
        weight_sum = 0

        for skill, value in skills_dict.items():
            weight = weights.get(skill, 0.05)
            total += value * weight
            weight_sum += weight

        self.level = total / weight_sum if weight_sum > 0 else 0


# ======================
# 文本可视化工具类
# ======================
class TextVisualizer:
    """文本可视化工具类"""

    def __init__(self):
        pass

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
# 增强学习系统主类（含完整进度条）
# ======================
class EnhancedLearningSystem:
    """增强学习系统 - 学术增强版（含完整进度条）"""

    def __init__(self, use_database: bool = False):
        print("🤖 初始化学术增强版AI学生智能体系统...")

        self.use_database = use_database

        # 初始化核心组件
        self.newton_model = NewtonRoleModel()
        self.learning_actions = LearningActions()
        self.viz = TextVisualizer()
        self.progress_viz = ProgressVisualizer()

        # 增强组件
        self.mcts_planner = MCTSPathPlanner(self.learning_actions, self.newton_model)

        # 初始化学生（增强版）
        self.students = self._initialize_enhanced_students()

        print(f"✅ 系统初始化完成，包含 {len(self.students)} 名增强版学生")
        print(f"📊 可用组件: MCTS路径规划、进度条可视化、多模态数据采集")

    def _initialize_enhanced_students(self) -> List[EnhancedStudent]:
        """初始化增强版学生"""
        students = []

        # 创建不同认知特征的学生
        profiles = [
            # 高专注高智商学生
            CognitiveProfile(
                learning_attitude=0.9,
                exam_performance=0.8,
                focus_ability=0.9,
                curiosity_level=0.7,
                intelligence=0.85
            ),
            # 高好奇心创造性学生
            CognitiveProfile(
                learning_attitude=0.8,
                curiosity_level=0.9,
                compliance=0.6
            ),
            # 普通学生
            CognitiveProfile(
                learning_attitude=0.6,
                exam_performance=0.5,
                focus_ability=0.5,
                prior_knowledge=0.4
            ),
            # 困难学生
            CognitiveProfile(
                learning_attitude=0.4,
                exam_performance=0.3,
                focus_ability=0.3
            )
        ]

        subjects = ["物理", "数学", "化学", "生物"]
        names = ["小明", "小红", "小刚", "小丽"]

        for i, profile in enumerate(profiles):
            student = EnhancedStudent(
                name=names[i],
                age=random.randint(12, 16),
                subject=subjects[i % len(subjects)],
                cognitive_profile=profile
            )

            # 设置学习内容
            if student.subject == "物理":
                student.module = "力学"
                student.topic = "牛顿运动定律"
            elif student.subject == "数学":
                student.module = "代数"
                student.topic = "二次方程"
            elif student.subject == "化学":
                student.module = "无机化学"
                student.topic = "化学反应"
            else:
                student.module = "细胞学"
                student.topic = "细胞分裂"

            students.append(student)

        return students

    def show_learning_position(self, student: EnhancedStudent):
        """显示学习位置（原版功能恢复）"""
        student.show_learning_position(self.progress_viz)

    def compare_with_ideal(self, student: EnhancedStudent):
        """与理想轨迹对比（原版进度条功能）"""
        student_skills = {
            "knowledge": student.skills.knowledge,
            "abstraction": student.skills.abstraction,
            "reasoning": student.skills.reasoning,
            "speed": student.skills.speed,
            "creativity": student.skills.creativity,
            "memory": student.skills.memory
        }

        return self.newton_model.display_progress_comparison(
            student_skills, student.age, student.name
        )

    def mcts_path_planning(self, student: EnhancedStudent) -> Dict:
        """MCTS路径规划"""
        print(f"\n🧭 为 {student.name} 进行MCTS路径规划...")

        student_state = student.get_state_for_mcts()

        start_time = time.time()
        path_result = self.mcts_planner.search(
            student_state,
            simulations=CONFIG["mcts_simulations"]
        )
        planning_time = time.time() - start_time

        print(f"  规划时间: {planning_time:.2f}秒")
        actions = path_result.get('actions', [])
        if actions:
            print(f"  推荐动作序列: {actions[:min(5, len(actions))]}...")
        else:
            print(f"  推荐动作序列: 无")
        print(f"  预期奖励: {path_result.get('expected_reward', 0):.3f}")

        return path_result

    def recommend_learning_strategy(self, student: EnhancedStudent) -> str:
        """推荐学习策略（含进度条考虑）"""
        # 检查进度状态
        student_skills = {
            "knowledge": student.skills.knowledge,
            "abstraction": student.skills.abstraction,
            "reasoning": student.skills.reasoning,
            "speed": student.skills.speed,
            "creativity": student.skills.creativity,
            "memory": student.skills.memory
        }

        equivalent_age = self.newton_model.get_equivalent_age(student_skills)
        age_gap = equivalent_age - student.age

        # 根据进度差距调整策略
        if age_gap < -1:  # 明显落后
            if student.fatigue > 0.6:
                return "rest"
            else:
                return self.learning_actions.get_recommended_action(student, self.newton_model)
        elif age_gap > 1:  # 明显超前
            return "creative_thinking"  # 鼓励创造性学习
        else:  # 正常进度
            if student.fatigue > 0.7:
                return "rest"
            elif student.attention < 0.5:
                return "interactive"
            elif student.level < 2.5:
                return "lecture"
            elif 2.5 <= student.level <= 3.5:
                return "example"
            else:
                return self.learning_actions.get_recommended_action(student, self.newton_model)

    def apply_learning_action(self, student: EnhancedStudent, action_id: str, session_id: str = "") -> Dict:
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

        # 更新认知状态
        action_intensity = self.learning_actions.actions.get(action_id, {}).get("fatigue", 0.3)
        student.update_cognitive_states(action_intensity)

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

        # 记录进度
        progress_record = student.record_progress(self.newton_model, session_id)

        return {
            "level_before": level_before,
            "level_after": student.level,
            "attention_before": attention_before,
            "attention_after": student.attention,
            "fatigue_before": fatigue_before,
            "fatigue_after": student.fatigue,
            "progress_record": progress_record
        }

    def enhanced_learning_process(self, student: EnhancedStudent, num_sessions: int = 5):
        """增强学习过程（含完整进度条）"""
        print(f"\n{'=' * 60}")
        print(f"🚀 开始 {student.name} 的个性化学习旅程")
        print(f"📚 学科：{student.subject} | 初始水平：{student.level:.2f}")
        print(f"{'=' * 60}")

        # 初始进度对比
        print(f"\n📊 初始进度评估:")
        initial_comparison = self.compare_with_ideal(student)

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

            # 智能推荐策略（考虑进度）
            strategy = self.recommend_learning_strategy(student)
            print(f"\n🤖 智能推荐策略：{strategy}")

            # 应用学习行为
            record = self.apply_learning_action(student, strategy, f"{session_id}_{session}")

            # 显示更新后的状态
            print(f"\n📊 更新后状态：")
            self.show_learning_position(student)

            # 每2次会话显示一次进度对比
            if session % 2 == 0:
                print(f"\n📈 进度更新：")
                current_progress = student.progress_history[-1]
                milestone = self.progress_viz.create_milestone_progress(
                    student.age, current_progress["equivalent_age"]
                )
                self.progress_viz.display_milestone_progress(milestone)

            # 更新天数
            student.day += 1
            student.update_age()

        # 最终进度对比
        print(f"\n{'=' * 60}")
        print("🎓 学习旅程结束")
        print(f"{'=' * 60}")

        final_comparison = self.compare_with_ideal(student)

        # 显示学习旅程地图
        if student.progress_history:
            print(f"\n🗺️ 学习旅程地图：")
            journey_map = self.progress_viz.create_learning_journey_map(
                student.progress_history, self.newton_model.target_trajectory
            )
            print(journey_map)

        return {
            "initial_comparison": initial_comparison,
            "final_comparison": final_comparison,
            "progress_history": student.progress_history
        }

    def run_comprehensive_demo(self):
        """运行综合演示（含完整进度条）"""
        print(f"\n{'=' * 70}")
        print("🎯 AI学生智能体系统 - 综合演示模式（含进度条）")
        print(f"{'=' * 70}")

        # 1. 学生基本信息展示
        print(f"\n1️⃣ 学生基本信息")
        student = self.students[0]
        self.show_learning_position(student)

        # 2. 初始进度对比
        print(f"\n2️⃣ 初始进度对比（与牛顿模型）")
        self.compare_with_ideal(student)

        # 3. MCTS路径规划
        print(f"\n3️⃣ MCTS路径规划演示")
        mcts_result = self.mcts_path_planning(student)

        # 4. 增强学习过程
        print(f"\n4️⃣ 增强学习过程演示（3次会话）")
        learning_result = self.enhanced_learning_process(student, num_sessions=3)

        # 5. 最终进度对比
        print(f"\n5️⃣ 学习效果总结")

        if learning_result.get("progress_history"):
            progress_history = learning_result["progress_history"]
            if len(progress_history) >= 2:
                initial = progress_history[0]
                final = progress_history[-1]

                print(f"   初始等效年龄: {initial['equivalent_age']:.1f}岁")
                print(f"   最终等效年龄: {final['equivalent_age']:.1f}岁")
                print(f"   进步: {final['equivalent_age'] - initial['equivalent_age']:+.1f}岁")
                print(f"   相似度提升: {final['similarity'] - initial['similarity']:+.3f}")

        print(f"\n{'=' * 70}")
        print("🎉 综合演示完成！")
        print(f"{'=' * 70}")

    def run_single_student_demo(self):
        """单人学生演示"""
        print(f"\n请选择学生:")
        for i, student in enumerate(self.students):
            print(f"{i + 1}. {student.name} ({student.subject}, {student.age}岁)")

        try:
            student_choice = int(input("\n请输入学生编号: ")) - 1
            if 0 <= student_choice < len(self.students):
                sessions = input("请输入学习会话数量 (默认5): ").strip()
                num_sessions = int(sessions) if sessions.isdigit() else 5
                student = self.students[student_choice]
                self.enhanced_learning_process(student, num_sessions)
            else:
                print("❌ 无效的学生编号")
        except ValueError:
            print("❌ 请输入有效的数字")

    def show_system_info(self):
        """显示系统信息"""
        print(f"\n{'=' * 60}")
        print("📋 系统信息")
        print(f"{'=' * 60}")
        print(f"系统版本: 4.0 (增强版含进度条)")
        print(f"学生数量: {len(self.students)}")
        print(f"学习策略: {len(self.learning_actions.actions)} 种")
        print(f"认知技能维度: 6 种 + 6种认知状态")
        print(f"进度条功能: ✅ 已启用")
        print(f"MCTS路径规划: ✅ 已启用")
        print(f"多模态数据: ✅ 已启用")
        print(f"{'=' * 60}")


# ======================
# 主程序入口
# ======================
def main():
    """主函数"""
    print("🎓 欢迎使用学术增强版AI学生智能体系统（含进度条）")
    print("版本: 4.0 (基于发明专利与学术论文)")
    print("=" * 60)

    # 创建增强系统
    learning_system = EnhancedLearningSystem(use_database=False)

    # 运行模式选择
    print("\n请选择运行模式:")
    print("1. 综合演示模式（完整功能展示）")
    print("2. 单人学生演示")
    print("3. MCTS路径规划测试")
    print("4. 进度条功能测试")
    print("5. 显示系统信息")
    print("6. 退出系统")

    choice = input("\n请输入选择 (1-6): ").strip()

    if choice == "1":
        learning_system.run_comprehensive_demo()
    elif choice == "2":
        learning_system.run_single_student_demo()
    elif choice == "3":
        student = learning_system.students[0]
        result = learning_system.mcts_path_planning(student)
        print(f"\n📋 MCTS规划结果:")
        print(f"  动作序列: {result.get('actions', [])}")
        print(f"  预期奖励: {result.get('expected_reward', 0):.3f}")
    elif choice == "4":
        print(f"\n📊 进度条功能测试")
        student = learning_system.students[0]

        # 显示当前进度
        learning_system.show_learning_position(student)

        # 与牛顿对比
        comparison = learning_system.compare_with_ideal(student)

        # 模拟学习并显示进度变化
        print(f"\n🔄 模拟学习过程...")
        for i in range(3):
            action = learning_system.recommend_learning_strategy(student)
            learning_system.apply_learning_action(student, action, f"test_{i}")

        # 显示学习后的进度
        print(f"\n📈 学习后进度:")
        learning_system.show_learning_position(student)
        learning_system.compare_with_ideal(student)
    elif choice == "5":
        learning_system.show_system_info()
    elif choice == "6":
        print("👋 感谢使用，再见！")
        return
    else:
        print("❌ 无效的选择")

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
        print("\n🎓 学术增强版AI学生智能体系统已关闭")