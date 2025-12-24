"""
Ai学生教育预测系统
"""

import numpy as np
import pandas as pd
import json
import random
import math
import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import pickle
import os
from pathlib import Path


# ======================
# 数据模型定义
# ======================

class CognitiveDimension(Enum):
    """认知维度枚举"""
    KNOWLEDGE = "知识积累"
    REASONING = "逻辑推理"
    CREATIVITY = "创造力"
    MEMORY = "记忆力"
    FOCUS = "专注力"
    SPEED = "思维速度"
    METACOGNITION = "元认知"


class LearningStage(Enum):
    """学习阶段枚举"""
    EARLY_CHILDHOOD = "幼儿期"  # 0-6岁
    PRIMARY = "小学阶段"  # 6-12岁
    MIDDLE = "中学阶段"  # 12-15岁
    HIGH = "高中阶段"  # 15-18岁
    COLLEGE = "大学阶段"  # 18-22岁
    ADULT = "成人阶段"  # 22-100岁


class LearningStrategy(Enum):
    """学习策略枚举"""
    EXPLICIT_INSTRUCTION = "显性教学"
    DISCOVERY_LEARNING = "发现学习"
    PROJECT_BASED = "项目式学习"
    PROBLEM_SOLVING = "问题解决"
    COLLABORATIVE = "协作学习"
    GAME_BASED = "游戏化学习"
    SCAFFOLDING = "脚手架学习"
    SPACED_REPETITION = "间隔重复"


# ======================
# 核心数据类
# ======================

@dataclass
class CognitiveProfile:
    """认知能力档案"""
    knowledge: float = 0.5
    reasoning: float = 0.5
    creativity: float = 0.5
    memory: float = 0.5
    focus: float = 0.5
    speed: float = 0.5
    metacognition: float = 0.5

    def to_dict(self):
        return {
            "knowledge": self.knowledge,
            "reasoning": self.reasoning,
            "creativity": self.creativity,
            "memory": self.memory,
            "focus": self.focus,
            "speed": self.speed,
            "metacognition": self.metacognition
        }

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass
class DailyLearningRecord:
    """每日学习记录"""
    date: datetime.date
    age_days: int
    cognitive_load: float  # 认知负荷
    engagement: float  # 参与度
    learning_time_minutes: int  # 学习时长
    topics_covered: List[str]  # 学习主题
    strategies_used: List[LearningStrategy]  # 使用的策略
    assessment_score: Optional[float] = None  # 评估分数


@dataclass
class KnowledgePoint:
    """知识点"""
    id: str
    name: str
    subject: str  # 学科
    difficulty: float  # 难度系数 0-1
    prerequisites: List[str]  # 前置知识点ID
    cognitive_requirements: Dict[str, float]  # 认知要求
    estimated_learning_time: int  # 预计学习分钟数


# ======================
# 0-100岁学习者标准模型
# ======================

class ZeroToHundredLearnerModel:
    """0-100岁学习者标准模型"""

    def __init__(self):
        self.age_stages = {
            0: LearningStage.EARLY_CHILDHOOD,
            6: LearningStage.PRIMARY,
            12: LearningStage.MIDDLE,
            15: LearningStage.HIGH,
            18: LearningStage.COLLEGE,
            22: LearningStage.ADULT
        }

        # 认知发展曲线参数
        self.cognitive_curves = self._init_cognitive_curves()

        # 学习数据存储
        self.daily_records = []  # 36500天的学习记录
        self.cognitive_profiles = {}  # 各年龄的认知档案

    def _init_cognitive_curves(self):
        """初始化认知发展曲线"""
        curves = {}

        # 使用S型曲线模拟认知发展
        for age in range(0, 101):
            normalized_age = age / 100.0

            # 不同认知维度的发展曲线
            curves[age] = {
                "knowledge": self._sigmoid_curve(normalized_age, 0.5, 8),
                "reasoning": self._sigmoid_curve(normalized_age, 0.4, 7),
                "creativity": self._double_peak_curve(normalized_age),
                "memory": self._sigmoid_curve(normalized_age, 0.3, 6),
                "focus": self._sigmoid_curve(normalized_age, 0.6, 9),
                "speed": self._inverse_u_curve(normalized_age),
                "metacognition": self._logistic_curve(normalized_age, 0.7, 10)
            }

        return curves

    def _sigmoid_curve(self, x, shift, steepness):
        """S型曲线"""
        return 1 / (1 + math.exp(-steepness * (x - shift)))

    def _double_peak_curve(self, x):
        """双峰曲线（创造力发展）"""
        return 0.7 * math.exp(-((x - 0.25) ** 2) / 0.02) + \
            0.8 * math.exp(-((x - 0.65) ** 2) / 0.03)

    def _inverse_u_curve(self, x):
        """倒U型曲线（思维速度）"""
        return 4 * x * (1 - x)

    def _logistic_curve(self, x, midpoint, growth_rate):
        """逻辑斯蒂曲线"""
        return 1 / (1 + math.exp(-growth_rate * (x - midpoint)))

    def generate_daily_records(self, num_years=100):
        """生成每日学习记录"""
        total_days = num_years * 365

        for day in range(total_days):
            age_years = day // 365
            age_days = day % 365

            # 确定学习阶段
            stage = self._get_learning_stage(age_years)

            # 生成认知档案
            if age_years <= 100:
                cognitive_profile = self._generate_cognitive_profile(age_years)
                self.cognitive_profiles[age_years] = cognitive_profile

            # 生成学习记录
            record = self._generate_learning_record(day, age_years, stage)
            self.daily_records.append(record)

        print(f"✅ 已生成 {len(self.daily_records)} 天的学习记录")
        return self.daily_records

    def _get_learning_stage(self, age):
        """获取学习阶段"""
        for threshold in sorted(self.age_stages.keys(), reverse=True):
            if age >= threshold:
                return self.age_stages[threshold]
        return LearningStage.EARLY_CHILDHOOD

    def _generate_cognitive_profile(self, age):
        """生成认知档案"""
        if age in self.cognitive_curves:
            curves = self.cognitive_curves[age]
            return CognitiveProfile(
                knowledge=curves["knowledge"],
                reasoning=curves["reasoning"],
                creativity=curves["creativity"],
                memory=curves["memory"],
                focus=curves["focus"],
                speed=curves["speed"],
                metacognition=curves["metacognition"]
            )
        return CognitiveProfile()

    def _generate_learning_record(self, day, age_years, stage):
        """生成学习记录"""
        # 模拟学习活动
        learning_time = self._get_learning_time_by_stage(stage)
        cognitive_load = random.uniform(0.3, 0.8)
        engagement = random.uniform(0.4, 0.9)

        # 学习主题
        topics = self._get_topics_by_stage(stage, age_years)

        # 学习策略
        strategies = self._get_strategies_by_stage(stage)

        return DailyLearningRecord(
            date=datetime.date(2000, 1, 1) + datetime.timedelta(days=day),
            age_days=day,
            cognitive_load=cognitive_load,
            engagement=engagement,
            learning_time_minutes=learning_time,
            topics_covered=topics,
            strategies_used=strategies
        )

    def _get_learning_time_by_stage(self, stage):
        """根据阶段获取学习时间"""
        times = {
            LearningStage.EARLY_CHILDHOOD: random.randint(30, 90),
            LearningStage.PRIMARY: random.randint(120, 240),
            LearningStage.MIDDLE: random.randint(180, 300),
            LearningStage.HIGH: random.randint(240, 360),
            LearningStage.COLLEGE: random.randint(180, 300),
            LearningStage.ADULT: random.randint(60, 180)
        }
        return times.get(stage, 120)

    def _get_topics_by_stage(self, stage, age):
        """根据阶段获取学习主题"""
        topics = []

        if stage == LearningStage.EARLY_CHILDHOOD:
            topics = ["语言发展", "基础认知", "社交技能", "运动能力"]
        elif stage == LearningStage.PRIMARY:
            topics = ["语文", "数学", "英语", "科学", "艺术"]
        elif stage == LearningStage.MIDDLE:
            topics = ["物理", "化学", "生物", "历史", "地理", "数学"]
        elif stage == LearningStage.HIGH:
            topics = ["高级数学", "物理原理", "化学实验", "文学分析", "外语"]
        elif stage == LearningStage.COLLEGE:
            topics = ["专业课程", "研究方法", "论文写作", "项目实践"]
        else:
            topics = ["职业技能", "终身学习", "兴趣发展"]

        return random.sample(topics, min(3, len(topics)))

    def _get_strategies_by_stage(self, stage):
        """根据阶段获取学习策略"""
        strategies = []

        if stage == LearningStage.EARLY_CHILDHOOD:
            strategies = [LearningStrategy.GAME_BASED, LearningStrategy.DISCOVERY_LEARNING]
        elif stage == LearningStage.PRIMARY:
            strategies = [LearningStrategy.EXPLICIT_INSTRUCTION, LearningStrategy.GAME_BASED]
        elif stage in [LearningStage.MIDDLE, LearningStage.HIGH]:
            strategies = [LearningStrategy.PROBLEM_SOLVING, LearningStrategy.SCAFFOLDING]
        elif stage == LearningStage.COLLEGE:
            strategies = [LearningStrategy.PROJECT_BASED, LearningStrategy.COLLABORATIVE]
        else:
            strategies = [LearningStrategy.SPACED_REPETITION, LearningStrategy.DISCOVERY_LEARNING]

        return random.sample(strategies, min(2, len(strategies)))

    def extract_k12_subset(self):
        """提取K12阶段数据子集（6-18岁）"""
        k12_start = 6 * 365
        k12_end = 18 * 365

        k12_records = []
        k12_profiles = {}

        for day in range(k12_start, k12_end):
            if day < len(self.daily_records):
                k12_records.append(self.daily_records[day])

            age_years = day // 365
            if 6 <= age_years <= 18:
                if age_years in self.cognitive_profiles:
                    k12_profiles[age_years] = self.cognitive_profiles[age_years]

        print(f"📚 已提取K12阶段数据: {len(k12_records)} 天记录")
        return k12_records, k12_profiles

    def visualize_cognitive_development(self):
        """可视化认知发展"""
        ages = list(range(0, 101))

        # 提取各年龄的认知维度数据
        knowledge = [self.cognitive_curves[age]["knowledge"] for age in ages]
        reasoning = [self.cognitive_curves[age]["reasoning"] for age in ages]
        creativity = [self.cognitive_curves[age]["creativity"] for age in ages]
        memory = [self.cognitive_curves[age]["memory"] for age in ages]

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(ages, knowledge, 'b-', linewidth=2)
        plt.title('知识积累发展曲线')
        plt.xlabel('年龄')
        plt.ylabel('发展水平')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 2)
        plt.plot(ages, reasoning, 'g-', linewidth=2)
        plt.title('逻辑推理发展曲线')
        plt.xlabel('年龄')
        plt.ylabel('发展水平')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 3)
        plt.plot(ages, creativity, 'r-', linewidth=2)
        plt.title('创造力发展曲线')
        plt.xlabel('年龄')
        plt.ylabel('发展水平')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 4)
        plt.plot(ages, memory, 'm-', linewidth=2)
        plt.title('记忆力发展曲线')
        plt.xlabel('年龄')
        plt.ylabel('发展水平')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# ======================
# 榜样模型（牛顿为例）
# ======================

class RoleModelLibrary:
    """榜样模型知识库"""

    def __init__(self):
        self.models = {}
        self._init_newton_model()

    def _init_newton_model(self):
        """初始化牛顿榜样模型"""
        # 牛顿的成长轨迹（0-84岁）
        newton_timeline = {}

        # 关键发展阶段
        key_ages = {
            0: {"stage": "婴儿期", "milestone": "基础感知"},
            6: {"stage": "童年期", "milestone": "基础教育开始"},
            12: {"stage": "少年期", "milestone": "对科学产生兴趣"},
            18: {"stage": "青年期", "milestone": "进入剑桥大学"},
            23: {"stage": "奇迹年", "milestone": "发明微积分、发现万有引力"},
            30: {"stage": "成熟期", "milestone": "发表《自然哲学的数学原理》"},
            45: {"stage": "中年期", "milestone": "皇家学会主席"},
            60: {"stage": "晚年期", "milestone": "铸币局局长"},
            84: {"stage": "终年", "milestone": "逝世"}
        }

        # 构建详细的学习轨迹
        for age in range(0, 85):
            newton_timeline[age] = self._generate_newton_learning_data(age)

        self.models["牛顿"] = {
            "name": "艾萨克·牛顿",
            "lifespan": (1643, 1727),
            "field": ["物理学", "数学", "天文学", "自然哲学"],
            "timeline": newton_timeline,
            "cognitive_profile": self._generate_newton_cognitive_profile(),
            "key_discoveries": [
                "万有引力定律",
                "运动三定律",
                "微积分",
                "光的色散理论",
                "反射望远镜"
            ]
        }

    def _generate_newton_learning_data(self, age):
        """生成牛顿的学习数据"""
        # 基于历史记录和合理推断
        if age <= 5:
            return {
                "daily_learning_hours": 2,
                "main_focus": ["基础读写", "算术", "宗教教育"],
                "cognitive_intensity": 0.3
            }
        elif 6 <= age <= 11:
            return {
                "daily_learning_hours": 4,
                "main_focus": ["拉丁语", "希腊语", "数学基础", "圣经研究"],
                "cognitive_intensity": 0.5
            }
        elif 12 <= age <= 17:
            return {
                "daily_learning_hours": 6,
                "main_focus": ["几何学", "天文学", "自然哲学", "实验方法"],
                "cognitive_intensity": 0.7
            }
        elif 18 <= age <= 22:  # 剑桥大学时期
            return {
                "daily_learning_hours": 10,
                "main_focus": ["数学", "光学", "力学", "炼金术"],
                "cognitive_intensity": 0.9
            }
        elif 23 <= age <= 30:  # 奇迹年及之后
            return {
                "daily_learning_hours": 12,
                "main_focus": ["微积分", "万有引力", "光学实验", "自然哲学体系"],
                "cognitive_intensity": 1.0
            }
        else:
            return {
                "daily_learning_hours": 8,
                "main_focus": ["科学研究", "行政管理", "神学研究"],
                "cognitive_intensity": 0.8
            }

    def _generate_newton_cognitive_profile(self):
        """生成牛顿的认知档案"""
        return CognitiveProfile(
            knowledge=4.9,
            reasoning=4.8,
            creativity=4.7,
            memory=4.6,
            focus=4.9,
            speed=4.5,
            metacognition=4.8
        )

    def get_model(self, name="牛顿"):
        """获取榜样模型"""
        return self.models.get(name)

    def calculate_similarity(self, student_profile, model_name="牛顿", age=18):
        """计算学生与榜样模型的相似度"""
        if model_name not in self.models:
            return 0.0

        model_profile = self.models[model_name]["cognitive_profile"]

        # 计算欧氏距离
        student_dict = student_profile.to_dict()
        model_dict = model_profile.to_dict()

        distance = 0
        for key in student_dict:
            if key in model_dict:
                distance += (student_dict[key] - model_dict[key]) ** 2

        similarity = 1 / (1 + math.sqrt(distance))
        return similarity

    def get_equivalent_age(self, student_profile, model_name="牛顿"):
        """计算相当于榜样模型的年龄"""
        if model_name not in self.models:
            return 0

        best_age = 0
        best_similarity = 0

        for age in range(0, 85):
            # 模拟该年龄的牛顿认知状态
            newton_at_age = self._estimate_newton_at_age(age)
            similarity = self._profile_similarity(student_profile, newton_at_age)

            if similarity > best_similarity:
                best_similarity = similarity
                best_age = age

        return best_age

    def _estimate_newton_at_age(self, age):
        """估计牛顿在特定年龄的认知状态"""
        # 简化估计：线性增长到峰值
        peak_age = 30
        if age <= peak_age:
            factor = age / peak_age
        else:
            factor = 1.0 - (age - peak_age) / 50

        base_profile = self.models["牛顿"]["cognitive_profile"]

        return CognitiveProfile(
            knowledge=base_profile.knowledge * factor,
            reasoning=base_profile.reasoning * factor,
            creativity=base_profile.creativity * factor,
            memory=base_profile.memory * factor,
            focus=base_profile.focus * factor,
            speed=base_profile.speed * factor,
            metacognition=base_profile.metacognition * factor
        )

    def _profile_similarity(self, profile1, profile2):
        """计算两个认知档案的相似度"""
        dict1 = profile1.to_dict()
        dict2 = profile2.to_dict()

        similarities = []
        for key in dict1:
            if key in dict2:
                diff = abs(dict1[key] - dict2[key])
                similarity = 1 - diff / 5.0  # 假设最大值为5
                similarities.append(similarity)

        return np.mean(similarities) if similarities else 0


# ======================
# 空白学生体
# ======================

class BlankStudentBody:
    """空白学生体（无知识状态的学习者）"""

    def __init__(self, name="空白学生", age=6, initial_conditions=None):
        self.name = name
        self.age_years = age
        self.age_days = age * 365

        # 认知档案（可自定义初始条件）
        if initial_conditions:
            self.cognitive_profile = CognitiveProfile(**initial_conditions)
        else:
            self.cognitive_profile = self._generate_initial_profile()

        # 学习历史
        self.learning_history = []

        # 知识掌握状态
        self.mastered_knowledge = set()
        self.learning_path = []

        # 实时状态
        self.fatigue = 0.3
        self.motivation = 0.7
        self.engagement = 0.6

    def _generate_initial_profile(self):
        """生成初始认知档案"""
        return CognitiveProfile(
            knowledge=random.uniform(0.3, 0.7),
            reasoning=random.uniform(0.3, 0.7),
            creativity=random.uniform(0.3, 0.7),
            memory=random.uniform(0.3, 0.7),
            focus=random.uniform(0.4, 0.8),
            speed=random.uniform(0.3, 0.7),
            metacognition=random.uniform(0.2, 0.6)
        )

    def learn_knowledge_point(self, knowledge_point, learning_time):
        """学习一个知识点"""
        # 计算学习效果
        effectiveness = self._calculate_learning_effectiveness(knowledge_point)

        # 更新认知能力
        self._update_cognitive_skills(knowledge_point, effectiveness)

        # 记录学习
        learning_record = {
            "timestamp": datetime.datetime.now(),
            "knowledge_point": knowledge_point.id,
            "learning_time": learning_time,
            "effectiveness": effectiveness,
            "fatigue_before": self.fatigue,
            "motivation_before": self.motivation
        }

        self.learning_history.append(learning_record)

        # 如果掌握足够好，添加到已掌握集合
        if effectiveness > 0.7:
            self.mastered_knowledge.add(knowledge_point.id)

        # 更新状态
        self.fatigue = min(1.0, self.fatigue + 0.1)
        self.motivation = max(0.1, self.motivation - 0.05)

        return effectiveness

    def _calculate_learning_effectiveness(self, knowledge_point):
        """计算学习效果"""
        # 基础效果
        base_effectiveness = 0.5

        # 认知能力影响
        cognitive_factors = {
            "knowledge": self.cognitive_profile.knowledge * 0.2,
            "reasoning": self.cognitive_profile.reasoning * 0.3,
            "memory": self.cognitive_profile.memory * 0.2,
            "focus": self.cognitive_profile.focus * 0.2,
            "metacognition": self.cognitive_profile.metacognition * 0.1
        }

        cognitive_boost = sum(cognitive_factors.values())

        # 状态影响
        state_factors = self.motivation * 0.3 + (1 - self.fatigue) * 0.2

        # 难度调整
        difficulty_factor = 1.0 - knowledge_point.difficulty * 0.3

        effectiveness = (base_effectiveness + cognitive_boost + state_factors) * difficulty_factor

        return min(1.0, max(0.0, effectiveness))

    def _update_cognitive_skills(self, knowledge_point, effectiveness):
        """更新认知技能"""
        # 根据知识点要求提升相关技能
        for skill, requirement in knowledge_point.cognitive_requirements.items():
            if hasattr(self.cognitive_profile, skill):
                current = getattr(self.cognitive_profile, skill)
                improvement = requirement * effectiveness * 0.01
                setattr(self.cognitive_profile, skill, min(5.0, current + improvement))

    def simulate_day(self, curriculum):
        """模拟一天的学习"""
        daily_plan = self._generate_daily_plan(curriculum)

        daily_summary = {
            "date": datetime.datetime.now().date(),
            "age_days": self.age_days,
            "knowledge_points_learned": [],
            "total_learning_time": 0,
            "average_effectiveness": 0
        }

        total_effectiveness = 0
        points_learned = 0

        for knowledge_point, planned_time in daily_plan:
            if self.fatigue > 0.8:
                break  # 疲劳过高，停止学习

            effectiveness = self.learn_knowledge_point(knowledge_point, planned_time)
            total_effectiveness += effectiveness
            points_learned += 1

            daily_summary["knowledge_points_learned"].append({
                "id": knowledge_point.id,
                "name": knowledge_point.name,
                "time": planned_time,
                "effectiveness": effectiveness
            })
            daily_summary["total_learning_time"] += planned_time

        if points_learned > 0:
            daily_summary["average_effectiveness"] = total_effectiveness / points_learned

        # 年龄增长
        self.age_days += 1
        if self.age_days % 365 == 0:
            self.age_years += 1

        # 状态恢复
        self._recover_overnight()

        return daily_summary

    def _generate_daily_plan(self, curriculum):
        """生成每日学习计划"""
        # 查找下一个可学习的知识点
        next_points = curriculum.get_next_knowledge_points(self.mastered_knowledge)

        daily_plan = []
        time_remaining = 240  # 4小时学习时间

        for point in next_points:
            if time_remaining <= 0:
                break

            # 估计学习时间
            estimated_time = point.estimated_learning_time
            actual_time = min(estimated_time, time_remaining)

            if actual_time >= 30:  # 至少学习30分钟
                daily_plan.append((point, actual_time))
                time_remaining -= actual_time

        return daily_plan

    def _recover_overnight(self):
        """过夜恢复"""
        self.fatigue = max(0.0, self.fatigue - 0.4)
        self.motivation = min(1.0, self.motivation + 0.2)

    def get_status_report(self):
        """获取状态报告"""
        return {
            "name": self.name,
            "age_years": self.age_years,
            "age_days": self.age_days,
            "cognitive_profile": self.cognitive_profile.to_dict(),
            "mastered_knowledge_count": len(self.mastered_knowledge),
            "fatigue": self.fatigue,
            "motivation": self.motivation,
            "total_learning_days": len(self.learning_history)
        }


# ======================
# 个性化学习路径生成
# ======================

class PersonalizedLearningPath:
    """个性化学习路径生成器"""

    def __init__(self, curriculum, role_model_lib):
        self.curriculum = curriculum
        self.role_model_lib = role_model_lib

        # 路径搜索算法参数
        self.exploration_weight = 1.41  # UCT算法中的探索权重
        self.simulation_depth = 10  # 模拟深度
        self.num_simulations = 100  # 模拟次数

    def generate_path(self, student, target_age=18, target_model="牛顿"):
        """生成个性化学习路径"""
        # 使用蒙特卡洛树搜索
        mcts_tree = MCTSTree(
            root_state=student,
            curriculum=self.curriculum,
            role_model=self.role_model_lib.get_model(target_model),
            exploration_weight=self.exploration_weight
        )

        # 运行搜索
        for i in range(self.num_simulations):
            mcts_tree.run_simulation()

        # 提取最佳路径
        best_path = mcts_tree.get_best_path()

        # 使用大模型模拟"举一反三"（简化版）
        enhanced_path = self._enhance_with_llm_simulation(best_path, student)

        return enhanced_path

    def _enhance_with_llm_simulation(self, base_path, student):
        """使用LLM模拟举一反三的衍生"""
        # 这里简化实现，实际应调用LLM API
        enhanced_path = []

        for step in base_path:
            # 为每个步骤生成变体
            variants = self._generate_variants(step, student.cognitive_profile)
            enhanced_path.append({
                "base_step": step,
                "variants": variants,
                "recommended_variant": self._select_best_variant(variants, student)
            })

        return enhanced_path

    def _generate_variants(self, step, cognitive_profile):
        """生成学习步骤的变体"""
        variants = []

        # 基于认知特征生成不同策略
        if cognitive_profile.creativity > 0.7:
            variants.append({
                "strategy": "项目式学习",
                "description": "通过实际项目掌握知识",
                "estimated_time": step["estimated_time"] * 1.2,
                "effectiveness_boost": 0.1
            })

        if cognitive_profile.memory > 0.7:
            variants.append({
                "strategy": "间隔重复",
                "description": "分多次学习，增强记忆",
                "estimated_time": step["estimated_time"] * 1.3,
                "effectiveness_boost": 0.15
            })

        if cognitive_profile.focus > 0.7:
            variants.append({
                "strategy": "深度学习",
                "description": "长时间专注学习",
                "estimated_time": step["estimated_time"] * 0.9,
                "effectiveness_boost": 0.05
            })

        # 默认策略
        variants.append({
            "strategy": "标准学习",
            "description": "传统学习方法",
            "estimated_time": step["estimated_time"],
            "effectiveness_boost": 0.0
        })

        return variants

    def _select_best_variant(self, variants, student):
        """选择最佳变体"""
        # 基于学生特征选择
        scores = []
        for variant in variants:
            score = self._calculate_variant_score(variant, student)
            scores.append((score, variant))

        return max(scores, key=lambda x: x[0])[1]

    def _calculate_variant_score(self, variant, student):
        """计算变体得分"""
        # 考虑时间效率和效果提升
        time_score = 1.0 / variant["estimated_time"]
        effectiveness_score = 1.0 + variant["effectiveness_boost"]

        # 考虑学生适应性
        if variant["strategy"] == "项目式学习" and student.cognitive_profile.creativity < 0.5:
            effectiveness_score *= 0.7

        return time_score * effectiveness_score


# ======================
# 蒙特卡洛树搜索算法
# ======================

class MCTSNode:
    """MCTS树节点"""

    def __init__(self, state, parent=None, action=None):
        self.state = state  # 学生状态
        self.parent = parent
        self.action = action  # 导致此状态的动作

        self.children = []
        self.visits = 0
        self.total_value = 0.0
        self.untried_actions = None

    def is_fully_expanded(self):
        """是否完全扩展"""
        return len(self.untried_actions) == 0

    def best_child(self, exploration_weight):
        """选择最佳子节点"""
        return max(self.children,
                   key=lambda c: c.total_value / (c.visits + 1e-6) +
                                 exploration_weight * math.sqrt(2 * math.log(self.visits + 1) / (c.visits + 1e-6)))

    def add_child(self, action, state):
        """添加子节点"""
        child = MCTSNode(state, parent=self, action=action)
        self.children.append(child)
        return child

    def update(self, value):
        """更新节点统计"""
        self.visits += 1
        self.total_value += value


class MCTSTree:
    """蒙特卡洛树搜索"""

    def __init__(self, root_state, curriculum, role_model, exploration_weight=1.41):
        self.root = MCTSNode(root_state)
        self.curriculum = curriculum
        self.role_model = role_model
        self.exploration_weight = exploration_weight

    def run_simulation(self):
        """运行一次模拟"""
        node = self.root

        # 选择阶段
        while not node.is_fully_expanded() and node.children:
            node = node.best_child(self.exploration_weight)

        # 扩展阶段
        if not node.is_fully_expanded():
            action = self._select_untried_action(node)
            new_state = self._apply_action(node.state, action)
            node = node.add_child(action, new_state)

        # 模拟阶段
        value = self._simulate(node.state)

        # 回溯更新
        while node is not None:
            node.update(value)
            node = node.parent

    def _select_untried_action(self, node):
        """选择未尝试的动作"""
        if node.untried_actions is None:
            # 获取可能的下一步知识点
            mastered = node.state.mastered_knowledge
            possible_points = self.curriculum.get_next_knowledge_points(mastered)
            node.untried_actions = possible_points[:5]  # 限制数量

        return random.choice(node.untried_actions)

    def _apply_action(self, student_state, knowledge_point):
        """应用动作（学习知识点）"""
        # 创建新学生状态副本
        import copy
        new_state = copy.deepcopy(student_state)

        # 模拟学习
        effectiveness = new_state.learn_knowledge_point(
            knowledge_point,
            knowledge_point.estimated_learning_time
        )

        return new_state

    def _simulate(self, state, depth=10):
        """模拟剩余路径"""
        simulated_state = state

        for _ in range(depth):
            if len(simulated_state.mastered_knowledge) >= 50:  # 假设总共50个知识点
                break

            # 随机选择一个可学习的知识点
            possible_points = self.curriculum.get_next_knowledge_points(
                simulated_state.mastered_knowledge
            )

            if not possible_points:
                break

            point = random.choice(possible_points)
            simulated_state.learn_knowledge_point(
                point,
                point.estimated_learning_time
            )

        # 计算模拟结果的价值
        return self._calculate_state_value(simulated_state)

    def _calculate_state_value(self, state):
        """计算状态价值"""
        # 考虑知识掌握度和与榜样模型的相似度
        knowledge_score = len(state.mastered_knowledge) / 50.0  # 归一化

        # 计算与榜样的相似度（简化）
        if self.role_model:
            model_profile = self.role_model.get("cognitive_profile", None)
            if model_profile:
                similarity = self._calculate_similarity(state.cognitive_profile, model_profile)
                return knowledge_score * 0.6 + similarity * 0.4

        return knowledge_score

    def _calculate_similarity(self, profile1, profile2):
        """计算相似度"""
        dict1 = profile1.to_dict()
        dict2 = profile2.to_dict() if hasattr(profile2, 'to_dict') else profile2

        similarities = []
        for key in dict1:
            if key in dict2:
                diff = abs(dict1[key] - dict2[key])
                similarity = 1 - diff / 5.0
                similarities.append(similarity)

        return np.mean(similarities) if similarities else 0

    def get_best_path(self):
        """获取最佳路径"""
        path = []
        node = self.root

        while node.children:
            # 选择访问次数最多的子节点
            best_child = max(node.children, key=lambda c: c.visits)

            if best_child.action:
                path.append({
                    "knowledge_point": best_child.action.id,
                    "name": best_child.action.name,
                    "estimated_time": best_child.action.estimated_learning_time
                })

            node = best_child

        return path


# ======================
# 课程体系
# ======================

class Curriculum:
    """课程体系"""

    def __init__(self, subject="物理"):
        self.subject = subject
        self.knowledge_points = {}
        self._init_knowledge_points()

    def _init_knowledge_points(self):
        """初始化知识点"""
        # 物理学科知识点示例
        points = [
            KnowledgePoint(
                id="phy_001",
                name="牛顿第一定律",
                subject="物理",
                difficulty=0.3,
                prerequisites=[],
                cognitive_requirements={
                    "knowledge": 0.3,
                    "reasoning": 0.4,
                    "memory": 0.2
                },
                estimated_learning_time=120
            ),
            KnowledgePoint(
                id="phy_002",
                name="牛顿第二定律",
                subject="物理",
                difficulty=0.5,
                prerequisites=["phy_001"],
                cognitive_requirements={
                    "knowledge": 0.4,
                    "reasoning": 0.6,
                    "memory": 0.3
                },
                estimated_learning_time=180
            ),
            KnowledgePoint(
                id="phy_003",
                name="牛顿第三定律",
                subject="物理",
                difficulty=0.4,
                prerequisites=["phy_001"],
                cognitive_requirements={
                    "knowledge": 0.3,
                    "reasoning": 0.5,
                    "memory": 0.3
                },
                estimated_learning_time=150
            ),
            KnowledgePoint(
                id="phy_004",
                name="万有引力定律",
                subject="物理",
                difficulty=0.7,
                prerequisites=["phy_001", "phy_002", "phy_003"],
                cognitive_requirements={
                    "knowledge": 0.6,
                    "reasoning": 0.7,
                    "creativity": 0.5,
                    "memory": 0.4
                },
                estimated_learning_time=240
            ),
            KnowledgePoint(
                id="phy_005",
                name="运动学基础",
                subject="物理",
                difficulty=0.4,
                prerequisites=[],
                cognitive_requirements={
                    "knowledge": 0.3,
                    "reasoning": 0.5,
                    "memory": 0.3
                },
                estimated_learning_time=150
            ),
            KnowledgePoint(
                id="phy_006",
                name="动能定理",
                subject="物理",
                difficulty=0.6,
                prerequisites=["phy_005", "phy_002"],
                cognitive_requirements={
                    "knowledge": 0.5,
                    "reasoning": 0.7,
                    "memory": 0.4
                },
                estimated_learning_time=210
            ),
        ]

        for point in points:
            self.knowledge_points[point.id] = point

    def get_next_knowledge_points(self, mastered_set):
        """获取下一个可学习的知识点"""
        next_points = []

        for point_id, point in self.knowledge_points.items():
            if point_id in mastered_set:
                continue

            # 检查前置条件是否满足
            prerequisites_met = all(p in mastered_set for p in point.prerequisites)

            if prerequisites_met:
                next_points.append(point)

        # 按难度排序
        next_points.sort(key=lambda x: x.difficulty)
        return next_points

    def get_knowledge_graph(self):
        """获取知识图谱"""
        graph = {"nodes": [], "edges": []}

        for point_id, point in self.knowledge_points.items():
            graph["nodes"].append({
                "id": point_id,
                "name": point.name,
                "difficulty": point.difficulty
            })

            for prereq in point.prerequisites:
                graph["edges"].append({
                    "from": prereq,
                    "to": point_id,
                    "type": "prerequisite"
                })

        return graph


# ======================
# 进度条机制
# ======================

class ProgressBarSystem:
    """进度条系统"""

    def __init__(self, role_model_lib):
        self.role_model_lib = role_model_lib

    def calculate_progress(self, student, model_name="牛顿"):
        """计算学习进度"""
        # 获取等效年龄
        equivalent_age = self.role_model_lib.get_equivalent_age(
            student.cognitive_profile,
            model_name
        )

        # 计算进度百分比
        if model_name == "牛顿":
            total_age = 84  # 牛顿的年龄
            progress = min(100, (equivalent_age / total_age) * 100)
        else:
            progress = equivalent_age  # 假设其他榜样也是100岁

        # 获取详细比较
        comparison = self._get_detailed_comparison(student, model_name, equivalent_age)

        return {
            "progress_percentage": progress,
            "equivalent_age": equivalent_age,
            "current_age": student.age_years,
            "comparison": comparison,
            "message": self._generate_progress_message(progress, equivalent_age, student.age_years)
        }

    def _get_detailed_comparison(self, student, model_name, equivalent_age):
        """获取详细比较"""
        model_data = self.role_model_lib.get_model(model_name)

        if not model_data:
            return {}

        # 获取榜样在等效年龄的成就
        model_at_age = model_data["timeline"].get(int(equivalent_age), {})

        comparison = {
            "model_achievements": model_at_age.get("main_focus", ["数据不足"]),
            "model_learning_hours": model_at_age.get("daily_learning_hours", 0),
            "student_current_state": student.get_status_report(),
            "age_gap": equivalent_age - student.age_years
        }

        return comparison

    def _generate_progress_message(self, progress, equivalent_age, current_age):
        """生成进度消息"""
        if equivalent_age > current_age:
            status = "超前"
            age_diff = equivalent_age - current_age
            message = f"🎉 很棒！你的知识水平相当于牛顿{equivalent_age:.1f}岁的水平，比实际年龄超前{age_diff:.1f}岁！"
        elif equivalent_age < current_age:
            status = "滞后"
            age_diff = current_age - equivalent_age
            message = f"📚 加油！你的知识水平相当于牛顿{equivalent_age:.1f}岁的水平，落后实际年龄{age_diff:.1f}岁。"
        else:
            status = "同步"
            message = f"✅ 很好！你的学习进度与实际年龄同步，保持当前的学习节奏。"

        if progress > 80:
            message += " 接近牛顿的巅峰水平！"
        elif progress > 60:
            message += " 已具备牛顿大学时期的水平！"
        elif progress > 40:
            message += " 已达到牛顿少年时期的科学兴趣阶段！"

        return message

    def visualize_progress(self, progress_data):
        """可视化进度"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 进度条
        ax1 = axes[0]
        progress = progress_data["progress_percentage"]

        ax1.barh([0], [progress], color='skyblue', edgecolor='navy', height=0.3)
        ax1.barh([0], [100 - progress], left=[progress], color='lightgray', edgecolor='gray', height=0.3)
        ax1.set_xlim(0, 100)
        ax1.set_yticks([])
        ax1.set_xlabel('进度百分比')
        ax1.set_title(f'学习进度: {progress:.1f}%')

        # 添加进度文本
        ax1.text(progress / 2, 0, f'{progress:.1f}%',
                 ha='center', va='center', fontsize=12, fontweight='bold')

        # 年龄对比图
        ax2 = axes[1]
        ages = ['实际年龄', '等效牛顿年龄']
        values = [progress_data["current_age"], progress_data["equivalent_age"]]
        colors = ['lightblue', 'lightcoral']

        bars = ax2.bar(ages, values, color=colors, edgecolor='black')
        ax2.set_ylabel('年龄（岁）')
        ax2.set_title('年龄对比')

        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{value:.1f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

        # 打印进度消息
        print("\n" + "=" * 60)
        print("📊 进度报告")
        print("=" * 60)
        print(progress_data["message"])
        print(f"\n🔍 详细比较:")
        print(f"   实际年龄: {progress_data['current_age']}岁")
        print(f"   等效牛顿年龄: {progress_data['equivalent_age']:.1f}岁")
        print(f"   年龄差距: {progress_data['comparison']['age_gap']:.1f}岁")


# ======================
# 主演示系统
# ======================

class AIStudentEducationSystem:
    """AI学生教育预测系统"""

    def __init__(self):
        print("🎓 AI学生教育预测系统初始化...")
        print("=" * 60)

        # 初始化各组件
        self.learner_model = ZeroToHundredLearnerModel()
        self.role_model_lib = RoleModelLibrary()
        self.curriculum = Curriculum("物理")
        self.progress_system = ProgressBarSystem(self.role_model_lib)

        # 创建空白学生体
        self.blank_student = None

        print("✅ 系统初始化完成")

    def demonstrate_0_to_100_model(self):
        """演示0-100岁学习者模型"""
        print("\n" + "=" * 60)
        print("📈 0-100岁学习者标准模型演示")
        print("=" * 60)

        # 生成数据
        self.learner_model.generate_daily_records(num_years=50)  # 只生成50年加速演示

        # 提取K12数据
        k12_records, k12_profiles = self.learner_model.extract_k12_subset()

        print(f"📚 K12阶段数据统计:")
        print(f"   记录天数: {len(k12_records)}")
        print(f"   认知档案数: {len(k12_profiles)}")

        # 可视化认知发展
        self.learner_model.visualize_cognitive_development()

        return k12_records, k12_profiles

    def create_blank_student(self, name="测试学生", age=12, initial_conditions=None):
        """创建空白学生体"""
        print(f"\n👤 创建空白学生体: {name}")

        self.blank_student = BlankStudentBody(
            name=name,
            age=age,
            initial_conditions=initial_conditions
        )

        status = self.blank_student.get_status_report()
        print(f"   年龄: {status['age_years']}岁")
        print(f"   认知档案: {status['cognitive_profile']}")

        return self.blank_student

    def simulate_learning_days(self, num_days=30):
        """模拟多天学习"""
        if not self.blank_student:
            print("❌ 请先创建空白学生体")
            return

        print(f"\n" + "=" * 60)
        print(f"📖 模拟 {num_days} 天学习过程")
        print("=" * 60)

        daily_summaries = []

        for day in range(1, num_days + 1):
            summary = self.blank_student.simulate_day(self.curriculum)
            daily_summaries.append(summary)

            if day % 5 == 0:  # 每5天显示一次进度
                print(f"📅 第{day}天完成:")
                print(f"   学习知识点: {len(summary['knowledge_points_learned'])}个")
                print(f"   总学习时间: {summary['total_learning_time']}分钟")
                print(f"   平均效果: {summary['average_effectiveness']:.2f}")

        # 显示最终状态
        final_status = self.blank_student.get_status_report()
        print(f"\n🎯 {num_days}天学习后:")
        print(f"   掌握知识点数: {final_status['mastered_knowledge_count']}")
        print(f"   认知能力变化: {final_status['cognitive_profile']}")

        return daily_summaries

    def generate_personalized_path(self):
        """生成个性化学习路径"""
        if not self.blank_student:
            print("❌ 请先创建空白学生体")
            return

        print("\n" + "=" * 60)
        print("🛤️ 生成个性化学习路径")
        print("=" * 60)

        # 创建路径生成器
        path_generator = PersonalizedLearningPath(
            self.curriculum,
            self.role_model_lib
        )

        # 生成路径
        path = path_generator.generate_path(self.blank_student)

        print(f"📋 生成的个性化学习路径 ({len(path)}个步骤):")
        for i, step in enumerate(path[:5]):  # 只显示前5步
            if isinstance(step, dict) and "base_step" in step:
                base = step["base_step"]
                variant = step.get("recommended_variant", {})
                print(f"  {i + 1}. {base.get('name', '未知')}")
                print(f"     策略: {variant.get('strategy', '标准学习')}")
                print(f"     预计时间: {variant.get('estimated_time', 120)}分钟")

        if len(path) > 5:
            print(f"  ... 还有{len(path) - 5}个步骤")

        return path

    def show_progress_bar(self):
        """显示进度条"""
        if not self.blank_student:
            print("❌ 请先创建空白学生体")
            return

        print("\n" + "=" * 60)
        print("📊 学习进度条系统")
        print("=" * 60)

        # 计算进度
        progress_data = self.progress_system.calculate_progress(
            self.blank_student,
            model_name="牛顿"
        )

        # 可视化进度
        self.progress_system.visualize_progress(progress_data)

        return progress_data

    def demonstrate_moba_game_mechanism(self):
        """演示MOBA游戏机制"""
        print("\n" + "=" * 60)
        print("🎮 MOBA游戏化学习机制演示")
        print("=" * 60)

        # 将知识点转化为技能
        skills = {}
        for point_id, point in self.curriculum.knowledge_points.items():
            skill_level = int(point.difficulty * 5) + 1  # 1-5级技能

            skills[point_id] = {
                "name": point.name,
                "level": skill_level,
                "damage": skill_level * 10,  # 技能伤害
                "cooldown": max(30, 60 - skill_level * 10),  # 冷却时间
                "mana_cost": skill_level * 5  # 消耗法力
            }

        # 创建游戏角色
        character = {
            "name": "学习勇者",
            "level": self.blank_student.age_years if self.blank_student else 12,
            "skills": list(skills.values())[:3],  # 前3个技能
            "health": 100,
            "mana": 100,
            "experience": 0
        }

        print(f"🎯 游戏角色: {character['name']} (Lv.{character['level']})")
        print(f"🛡️  生命值: {character['health']} | 法力值: {character['mana']}")
        print(f"📚 已掌握技能:")
        for skill in character["skills"]:
            print(f"    {skill['name']} (Lv.{skill['level']}) - 伤害: {skill['damage']}")

        # 模拟战斗
        print(f"\n⚔️  模拟学习战斗:")
        print("   击败'数学难题怪兽'获得经验值!")

        # 计算经验获取
        if self.blank_student:
            experience_gain = self.blank_student.mastered_knowledge_count * 10
            character["experience"] += experience_gain
            print(f"   获得 {experience_gain} 经验值!")
            print(f"   总经验值: {character['experience']}")

        return character

    def calculate_computational_cost(self):
        """计算算力成本"""
        print("\n" + "=" * 60)
        print("💻 算力资源配置计算")
        print("=" * 60)

        # 按专利中的公式计算
        k12_days = 4380  # 12年×365天

        # Token消耗估计
        daily_tokens = {
            "教材阅读": 5000,
            "课堂听讲": 2500,
            "习题训练": 1600
        }

        total_tokens_per_day = sum(daily_tokens.values())
        total_tokens_k12 = k12_days * total_tokens_per_day

        # 成本估计（假设每百万Token 10元）
        cost_per_million = 10
        total_cost = (total_tokens_k12 / 1_000_000) * cost_per_million

        print(f"📊 K12阶段算力消耗分析:")
        print(f"   总天数: {k12_days}天")
        print(f"   每日Token消耗:")
        for item, tokens in daily_tokens.items():
            print(f"     {item}: {tokens:,} Token")
        print(f"   每日总计: {total_tokens_per_day:,} Token")
        print(f"   K12阶段总计: {total_tokens_k12:,} Token")
        print(f"   生成成本: ¥{total_cost:.2f} (按10元/百万Token)")

        return {
            "k12_days": k12_days,
            "daily_tokens": daily_tokens,
            "total_tokens_k12": total_tokens_k12,
            "estimated_cost": total_cost
        }

    def run_full_demo(self):
        """运行完整演示"""
        print("🚀 AI学生教育预测模型完整演示")
        print("=" * 60)

        # 1. 演示0-100岁模型
        self.demonstrate_0_to_100_model()

        # 2. 创建空白学生体
        initial_conditions = {
            "knowledge": 0.4,
            "reasoning": 0.5,
            "creativity": 0.6,
            "memory": 0.7,
            "focus": 0.6,
            "speed": 0.5,
            "metacognition": 0.4
        }
        self.create_blank_student("小明", 12, initial_conditions)

        # 3. 模拟学习过程
        input("\n按回车键开始模拟学习过程...")
        self.simulate_learning_days(15)

        # 4. 生成个性化路径
        input("\n按回车键生成个性化学习路径...")
        self.generate_personalized_path()

        # 5. 显示进度条
        input("\n按回车键查看学习进度...")
        self.show_progress_bar()

        # 6. 演示游戏化机制
        input("\n按回车键体验游戏化学习...")
        self.demonstrate_moba_game_mechanism()

        # 7. 计算算力成本
        input("\n按回车键计算算力成本...")
        self.calculate_computational_cost()

        print("\n" + "=" * 60)
        print("🎉 演示完成！")
        print("=" * 60)


# ======================
# 主程序
# ======================

def main():
    """主函数"""
    print("🎓 基于AI学生的教育预测模型系统")
    print("版本: 1.0 (专利实现版)")
    print("=" * 60)
    print("本系统基于专利《一种基于ai学生的教育预测模型的建构方法》实现")
    print("=" * 60)

    # 创建系统
    system = AIStudentEducationSystem()

    # 显示菜单
    while True:
        print("\n" + "=" * 60)
        print("📋 主菜单")
        print("=" * 60)
        print("1. 运行完整演示")
        print("2. 演示0-100岁学习者模型")
        print("3. 创建并测试空白学生体")
        print("4. 生成个性化学习路径")
        print("5. 显示学习进度条")
        print("6. 演示MOBA游戏机制")
        print("7. 计算算力成本")
        print("8. 退出系统")
        print("=" * 60)

        choice = input("请输入选择 (1-8): ").strip()

        try:
            if choice == "1":
                system.run_full_demo()
            elif choice == "2":
                system.demonstrate_0_to_100_model()
            elif choice == "3":
                name = input("请输入学生姓名: ") or "测试学生"
                age = int(input("请输入学生年龄(6-18): ") or "12")
                system.create_blank_student(name, age)
                days = int(input("请输入模拟天数: ") or "10")
                system.simulate_learning_days(days)
            elif choice == "4":
                if not system.blank_student:
                    print("⚠️ 未创建学生，使用默认学生")
                    system.create_blank_student()
                system.generate_personalized_path()
            elif choice == "5":
                if not system.blank_student:
                    print("⚠️ 未创建学生，使用默认学生")
                    system.create_blank_student()
                system.show_progress_bar()
            elif choice == "6":
                system.demonstrate_moba_game_mechanism()
            elif choice == "7":
                system.calculate_computational_cost()
            elif choice == "8":
                print("👋 感谢使用，再见！")
                break
            else:
                print("❌ 无效选择，请重新输入")

            input("\n按回车键继续...")

        except Exception as e:
            print(f"❌ 发生错误: {e}")
            import traceback
            traceback.print_exc()
            input("\n按回车键继续...")


# ======================
# 程序启动
# ======================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 程序被用户中断")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("\n🎓 AI学生教育预测系统已关闭")



