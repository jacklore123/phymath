"""
AI学生：单一智能体的4380次展开
核心理解：不是4380个并行对象，而是同一个智能体在时间维度上的4380次状态演化
修复版本：解决KeyError问题
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
import random
from collections import deque, defaultdict
# 在文件开头添加以下代码
import matplotlib
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 设置字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


# ======================
# 核心：单一智能体定义
# ======================

class SingleAIStudent:
    """单一AI学生智能体 - 将在4380天内持续演化"""

    def __init__(self, name="AI学生", initial_age=6):
        self.name = name
        self.age_days = initial_age * 365  # 初始天数

        # 🔥 核心：只有一个智能体，但随时间演化
        self.cognitive_state = self._initialize_state()
        self.knowledge_base = {}
        self.learning_history = []  # 所有历史记录

        # 过程引擎（在时间中展开的核心）
        self.process_engine = CognitiveProcessEngine()

        # 世界模型（随时间学习）
        self.world_model = StudentWorldModel()

        # 当前目标（随时间变化）
        self.current_goals = self._get_age_based_goals()

        print(f"🎓 创建单一AI学生: {name}")
        print(f"   将从这个状态开始，在4380天内持续演化")
        print(f"   初始认知状态: {self._summarize_state()}")

    def _initialize_state(self):
        """初始化认知状态"""
        return {
            # 认知能力维度（随时间发展）
            "working_memory": 0.3,
            "processing_speed": 0.4,
            "logical_reasoning": 0.3,
            "abstract_thinking": 0.2,
            "creativity": 0.4,
            "focus": 0.5,

            # 情感与动机维度（每天波动）
            "motivation": 0.7,
            "curiosity": 0.8,
            "confidence": 0.6,
            "frustration": 0.2,

            # 元认知维度（随时间学习）
            "self_awareness": 0.4,
            "strategy_knowledge": 0.3,
            "error_monitoring": 0.3,

            # 生理状态（每天变化）
            "energy": 0.8,
            "stress": 0.2
        }

    def _get_age_based_goals(self):
        """根据年龄获取目标"""
        age_years = self.age_days // 365

        if age_years < 9:
            return ["基础技能", "好奇探索", "社交学习"]
        elif age_years < 13:
            return ["系统知识", "逻辑思维", "兴趣发展"]
        elif age_years < 16:
            return ["抽象思维", "学科深化", "自我认知"]
        else:
            return ["专业方向", "批判思维", "独立学习"]

    def _summarize_state(self):
        """简要总结状态"""
        return {
            "knowledge_domains": len(self.knowledge_base),
            "avg_knowledge": np.mean(list(self.knowledge_base.values())) if self.knowledge_base else 0,
            "cognitive_ability": np.mean(
                [self.cognitive_state[k] for k in ["logical_reasoning", "abstract_thinking", "creativity"]]),
            "motivational_state": self.cognitive_state["motivation"]
        }


# ======================
# 过程引擎：在时间中展开思考
# ======================

class CognitiveProcessEngine:
    """认知过程引擎 - 智能在时间中展开"""

    def execute(self, start_state, material, world_model):
        """执行认知过程"""

        # 🔥 核心：思考过程在内部时间中展开
        thought_steps = []
        current_state = start_state.copy()

        # 步骤1：感知与编码
        encoded = self._encode_material(current_state, material)
        # 只更新或添加新键到当前状态
        current_state.update(encoded)
        thought_steps.append({"step": "encode", "state": current_state.copy()})

        # 步骤2：与已有知识整合
        if "prior_knowledge" in material and material["prior_knowledge"] > 0.3:
            integrated = self._integrate_with_prior(current_state, encoded, material)
            current_state.update(integrated)
            thought_steps.append({"step": "integrate", "state": current_state.copy()})

        # 步骤3：理解与推理
        understood = self._comprehend(current_state, material)
        current_state.update(understood)
        thought_steps.append({"step": "comprehend", "state": current_state.copy()})

        # 步骤4：应用与迁移
        if random.random() > 0.3:  # 70%概率尝试迁移
            transferred = self._transfer_learning(current_state, material)
            current_state.update(transferred)
            thought_steps.append({"step": "transfer", "state": current_state.copy()})

        # 步骤5：反思与巩固
        reflected = self._reflect_on_learning(current_state, thought_steps)
        current_state.update(reflected)
        thought_steps.append({"step": "reflect", "state": current_state.copy()})

        # 计算学习收益
        learning_gain = self._calculate_learning_gain(start_state, current_state, material)

        # 检查突破
        breakthrough = learning_gain > 0.15 and len(thought_steps) > 3

        # 挣扎程度
        struggle = 1.0 - (len([s for s in thought_steps if "state" in s]) / 5.0)

        return current_state, {
            "thought_steps": thought_steps,
            "learning_gain": learning_gain,
            "process_length": len(thought_steps),
            "breakthrough": breakthrough,
            "struggle": struggle,
            "material_difficulty": material.get("difficulty", 0.5)
        }

    def _encode_material(self, state, material):
        """编码学习材料"""
        # 编码效率受工作记忆和专注力影响
        encoding_efficiency = state["working_memory"] * 0.5 + state["focus"] * 0.5
        return {
            "encoded_strength": encoding_efficiency * material.get("difficulty", 0.5),
            "attention_level": state["focus"]
        }

    def _integrate_with_prior(self, state, encoded, material):
        """与先前知识整合"""
        prior_strength = material.get("prior_knowledge", 0)
        integration_quality = state["logical_reasoning"] * 0.3 + prior_strength * 0.7

        return {
            "integration_quality": integration_quality,
            "schema_strength": min(1.0, prior_strength + integration_quality * 0.1)
        }

    def _comprehend(self, state, material):
        """理解过程"""
        comprehension = state["logical_reasoning"] * 0.4 + state["abstract_thinking"] * 0.3 + state["creativity"] * 0.3
        comprehension *= (1.0 - material.get("difficulty", 0.5) * 0.3)

        return {
            "comprehension_level": comprehension,
            "conceptual_clarity": comprehension * state["focus"]
        }

    def _transfer_learning(self, state, material):
        """迁移学习"""
        transfer_ability = state["abstract_thinking"] * 0.5 + state["creativity"] * 0.5
        return {
            "transfer_success": random.random() < transfer_ability,
            "analogies_made": 1 if transfer_ability > 0.6 else 0
        }

    def _reflect_on_learning(self, state, thought_steps):
        """反思学习"""
        reflection_depth = state["self_awareness"] * 0.7 + state["error_monitoring"] * 0.3
        insights = len(thought_steps) * 0.1 * reflection_depth

        return {
            "insights_gained": insights,
            "metacognitive_awareness": reflection_depth,
            "confidence_change": insights * 0.2
        }

    def _calculate_learning_gain(self, start_state, end_state, material):
        """计算学习收益"""
        # 只计算两个状态中都存在的键
        common_keys = set(start_state.keys()) & set(end_state.keys())
        dimension_improvements = []

        for key in ["logical_reasoning", "abstract_thinking", "creativity", "confidence"]:
            if key in common_keys:
                improvement = end_state[key] - start_state[key]
                dimension_improvements.append(max(0, improvement))

        avg_improvement = np.mean(dimension_improvements) if dimension_improvements else 0

        # 材料难度调整
        difficulty_factor = 1.0 - material.get("difficulty", 0.5) * 0.2

        return avg_improvement * difficulty_factor * 10  # 放大到合理范围


# ======================
# 世界模型：随时间学习规律
# ======================

class StudentWorldModel:
    """学生心智世界模型 - 随时间学习"""

    def __init__(self):
        self.observations = []
        self.learned_patterns = {
            "best_learning_time": "morning",  # 随时间调整
            "optimal_study_duration": 45,  # 分钟
            "effective_strategies": [],  # 逐渐发现
            "personal_rhythms": {}  # 个人学习节律
        }

    def learn_from_experience(self, experience):
        """从经验中学习"""
        self.observations.append(experience)

        # 当有足够数据时，开始发现模式
        if len(self.observations) > 100:
            self._discover_patterns()

        # 简单学习：记录什么情况下学习效果好
        if experience["gain"] > 0.1:  # 学习效果好
            state_before = experience["state_before"]

            # 记录高动机时的学习效果
            if state_before.get("motivation", 0) > 0.7:
                if "high_motivation_success" not in self.learned_patterns:
                    self.learned_patterns["high_motivation_success"] = []
                self.learned_patterns["high_motivation_success"].append(experience["gain"])

    def _discover_patterns(self):
        """发现学习模式"""
        if len(self.observations) < 50:
            return

        # 分析最佳学习时间
        morning_gains = []
        afternoon_gains = []

        for obs in self.observations[-50:]:
            # 简单模拟：根据day的奇偶模拟上下午
            if obs["day"] % 2 == 0:
                morning_gains.append(obs["gain"])
            else:
                afternoon_gains.append(obs["gain"])

        if morning_gains and afternoon_gains:
            avg_morning = np.mean(morning_gains)
            avg_afternoon = np.mean(afternoon_gains)

            if avg_morning > avg_afternoon:
                self.learned_patterns["best_learning_time"] = "morning"
            else:
                self.learned_patterns["best_learning_time"] = "afternoon"

        # 发现有效策略
        successful_obs = [obs for obs in self.observations if obs["gain"] > 0.15]
        if successful_obs:
            strategies = [obs["action"].get("strategy", "") for obs in successful_obs]
            strategy_counts = {}
            for s in strategies:
                if s:
                    strategy_counts[s] = strategy_counts.get(s, 0) + 1

            if strategy_counts:
                best_strategy = max(strategy_counts, key=strategy_counts.get)
                if best_strategy not in self.learned_patterns["effective_strategies"]:
                    self.learned_patterns["effective_strategies"].append(best_strategy)

    def predict(self, current_state, planned_action):
        """预测行动效果"""
        # 基于已学到的模式做简单预测
        prediction = {
            "expected_gain": 0.08,  # 基础预期
            "confidence": 0.6
        }

        # 应用已学模式
        if self.learned_patterns["best_learning_time"] == "morning":
            # 如果是"上午"且状态好，提高预期
            if current_state.get("energy", 0) > 0.7:
                prediction["expected_gain"] += 0.03

        # 应用有效策略知识
        action_strategy = planned_action.get("strategy", "")
        if action_strategy in self.learned_patterns["effective_strategies"]:
            prediction["expected_gain"] += 0.04
            prediction["confidence"] += 0.1

        # 状态影响
        if current_state.get("motivation", 0) > 0.7:
            prediction["expected_gain"] += 0.02

        if current_state.get("focus", 0) > 0.7:
            prediction["expected_gain"] += 0.02

        return prediction


# ======================
# 课程生成器
# ======================

class CurriculumGenerator:
    """随时间演化的课程生成"""

    def generate_for_day(self, age_days):
        """生成某一天的课程"""
        age_years = age_days // 365

        # 基础主题
        base_subjects = ["数学", "语文", "科学", "历史", "艺术", "体育"]

        # 随着年龄增加主题复杂度
        if age_years < 9:
            topics = [f"{subject}_基础" for subject in base_subjects[:4]]
        elif age_years < 13:
            topics = [f"{subject}_进阶" for subject in base_subjects]
        elif age_years < 16:
            topics = [f"{subject}_深入" for subject in base_subjects]
        else:
            topics = [f"{subject}_专业" for subject in base_subjects]

        # 每日选择一个主题
        selected = random.choice(topics)

        # 难度随年龄增加
        base_difficulty = min(0.9, 0.3 + age_years * 0.04)

        return {
            "topics": [selected],
            "difficulty": random.uniform(base_difficulty - 0.1, base_difficulty + 0.1),
            "social": random.choice(["individual", "group"]),
            "duration": random.randint(40, 60)
        }


# ======================
# 辅助函数
# ======================

def _perceive_environment(student, curriculum):
    """感知环境"""
    return {
        "available_topics": curriculum.get("topics", []),
        "difficulty": curriculum.get("difficulty", 0.5),
        "social_context": curriculum.get("social", "individual"),
        "student_mood": student.cognitive_state["motivation"] * 0.7 + student.cognitive_state["curiosity"] * 0.3
    }


def _make_learning_decision(student, perception):
    """制定学习决策"""
    # 基于目标、兴趣、知识缺口决策
    topics = perception["available_topics"]

    if not topics:
        return {"selected_topic": "general_learning", "strategy": "self_study"}

    # 目标匹配
    for goal in student.current_goals:
        for topic in topics:
            if goal in topic or any(word in topic for word in goal.split("_")):
                return {"selected_topic": topic, "strategy": "goal_driven"}

    # 兴趣匹配
    topic_interests = {topic: random.uniform(0.3, 0.9) for topic in topics}
    most_interesting = max(topic_interests, key=topic_interests.get)

    # 知识缺口匹配
    if student.knowledge_base:
        gaps = {topic: 1.0 - student.knowledge_base.get(topic, 0) for topic in topics}
        biggest_gap = max(gaps, key=gaps.get)

        # 权衡：兴趣 vs 知识缺口
        if gaps[biggest_gap] > 0.6:  # 缺口很大
            selected = biggest_gap
        else:
            selected = most_interesting
    else:
        selected = most_interesting

    return {
        "selected_topic": selected,
        "strategy": "interest_based" if selected == most_interesting else "gap_filling",
        "interest_score": topic_interests.get(selected, 0.5),
        "gap_score": 1.0 - student.knowledge_base.get(selected, 0) if student.knowledge_base else 0.5
    }


def _apply_daily_adjustments(student):
    """应用每日调整"""
    # 遗忘曲线
    for topic in list(student.knowledge_base.keys()):
        # 简单遗忘模型
        forgetting_rate = 0.01  # 每天遗忘1%
        student.knowledge_base[topic] *= (1 - forgetting_rate)

    # 疲劳恢复
    student.cognitive_state["energy"] = min(1.0, student.cognitive_state["energy"] + 0.3)

    # 动机波动
    motivation_change = random.uniform(-0.05, 0.05)
    student.cognitive_state["motivation"] = max(0.1, min(1.0,
                                                         student.cognitive_state["motivation"] + motivation_change))


# ======================
# 智能体的单日展开（关键函数）
# ======================

def single_day_unfolding(student, day_curriculum):
    """
    单一智能体的一日展开
    返回：这一天的学习快照（第N个agent）
    """

    # 1. 获取当前状态（这一天开始时的状态）
    start_state = student.cognitive_state.copy()
    start_knowledge = student.knowledge_base.copy()

    # 2. 当日目标更新
    student.current_goals = student._get_age_based_goals()

    # 3. 感知与决策
    perception = _perceive_environment(student, day_curriculum)
    decision = _make_learning_decision(student, perception)

    # 4. 学习过程执行
    learning_material = {
        "topic": decision["selected_topic"],
        "difficulty": perception["difficulty"],
        "prior_knowledge": student.knowledge_base.get(decision["selected_topic"], 0.2)
    }

    # 核心：执行认知过程
    final_state, process_trace = student.process_engine.execute(
        start_state, learning_material, student.world_model
    )

    # 5. 更新智能体状态（智能体演化！）
    # 安全更新：合并最终状态到认知状态
    for key, value in final_state.items():
        student.cognitive_state[key] = value

    # 6. 知识更新
    knowledge_gain = process_trace["learning_gain"]
    topic = decision["selected_topic"]
    student.knowledge_base[topic] = student.knowledge_base.get(topic, 0) + knowledge_gain

    # 7. 世界模型学习
    student.world_model.learn_from_experience({
        "day": student.age_days,
        "state_before": start_state,
        "action": decision,
        "state_after": final_state,
        "gain": knowledge_gain
    })

    # 8. 每日状态调整（遗忘、疲劳等）
    _apply_daily_adjustments(student)

    # 9. 年龄增长
    student.age_days += 1

    # 10. 记录历史
    # 修复KeyError：只计算两个状态中都存在的键的差值
    common_keys = set(start_state.keys()) & set(final_state.keys())
    state_delta = {k: final_state[k] - start_state.get(k, 0) for k in common_keys}

    daily_snapshot = {
        "day": student.age_days - 1,  # 这一天结束时的天数
        "age_years": (student.age_days - 1) // 365,

        # 状态快照（这就是第N个"agent"）
        "state_snapshot": {
            "cognitive": student.cognitive_state.copy(),
            "knowledge": student.knowledge_base.copy(),
            "goals": student.current_goals.copy()
        },

        # 过程记录
        "process": process_trace,
        "decision": decision,
        "perception": perception,

        # 变化量（修复后的）
        "state_delta": state_delta,
        "knowledge_gain": knowledge_gain,

        # 元信息
        "is_breakthrough": process_trace.get("breakthrough", False),
        "struggle_level": process_trace.get("struggle", 0)
    }

    student.learning_history.append(daily_snapshot)

    return daily_snapshot


# ======================
# 主系统：4380次展开
# ======================

class AIStudentSystem:
    """AI学生系统：单一智能体的4380次展开"""

    def __init__(self, name="AI学生", start_age=6, total_years=12):
        self.name = name
        self.start_age = start_age
        self.total_years = total_years
        self.total_days = total_years * 365

        # 🔥 核心：只有一个智能体
        print("=" * 60)
        print(f"🎓 创建AI学生系统")
        print(f"   学生姓名: {name}")
        print(f"   起始年龄: {start_age}岁")
        print(f"   总年限: {total_years}年 ({self.total_days}天)")
        print(f"   🔥 核心哲学: 1个智能体 × {self.total_days}次展开")
        print("=" * 60)

        self.student = SingleAIStudent(name, start_age)
        self.curriculum_gen = CurriculumGenerator()

        # 记录所有展开的快照
        self.all_snapshots = []  # 这就是4380个"agent"的快照

    def simulate_full_development(self):
        """模拟完整发展过程"""
        print(f"\n🚀 开始模拟 {self.total_years} 年发展...")
        print(f"   每天执行一次智能体展开")
        print(f"   将生成 {self.total_days} 个状态快照")
        print("-" * 60)

        for day in range(self.total_days):
            # 生成当日课程
            curriculum = self.curriculum_gen.generate_for_day(self.student.age_days)

            # 🔥 核心：智能体单日展开
            snapshot = single_day_unfolding(self.student, curriculum)

            self.all_snapshots.append(snapshot)

            # 进度报告
            if day % 365 == 0 and day > 0:
                self._annual_report(day)

            if day % 100 == 0:
                self._progress_update(day)

        print("=" * 60)
        print(f"✅ 模拟完成!")
        print(f"   总展开次数: {len(self.all_snapshots)}")
        print(f"   最终年龄: {self.student.age_days // 365}岁")
        print(f"   知识领域数: {len(self.student.knowledge_base)}")
        print("=" * 60)

        return self.all_snapshots

    def _annual_report(self, day):
        """年度报告"""
        year = day // 365 + self.start_age
        snapshot = self.all_snapshots[-1]

        print(f"\n📅 第{year}年报告:")
        print(f"   认知能力: {snapshot['state_snapshot']['cognitive']['logical_reasoning']:.2f}")
        print(f"   动机水平: {snapshot['state_snapshot']['cognitive']['motivation']:.2f}")
        print(f"   知识领域: {len(snapshot['state_snapshot']['knowledge'])}个")

        # 年度学习统计
        year_snapshots = self.all_snapshots[-365:]
        breakthroughs = sum(1 for s in year_snapshots if s.get('is_breakthrough', False))
        avg_gain = np.mean([s.get('knowledge_gain', 0) for s in year_snapshots])

        print(f"   突破次数: {breakthroughs}")
        print(f"   平均日收益: {avg_gain:.3f}")

    def _progress_update(self, day):
        """进度更新"""
        if day > 0 and day % 100 == 0:
            snapshot = self.all_snapshots[-1]
            print(f"   Day {day}: 知识={len(snapshot['state_snapshot']['knowledge'])}领域, "
                  f"动机={snapshot['state_snapshot']['cognitive']['motivation']:.2f}")

    def get_agent_at_day(self, day):
        """获取第N天的"agent"（状态快照）"""
        if 0 <= day < len(self.all_snapshots):
            return self.all_snapshots[day]
        return None

    def visualize_development(self):
        """可视化发展过程"""
        if not self.all_snapshots:
            print("❌ 没有模拟数据")
            return

        # 提取数据
        days = list(range(len(self.all_snapshots)))
        ages = [s['age_years'] for s in self.all_snapshots]

        # 认知能力
        reasoning = [s['state_snapshot']['cognitive']['logical_reasoning'] for s in self.all_snapshots]
        abstract = [s['state_snapshot']['cognitive']['abstract_thinking'] for s in self.all_snapshots]

        # 动机与知识
        motivation = [s['state_snapshot']['cognitive']['motivation'] for s in self.all_snapshots]
        knowledge_domains = [len(s['state_snapshot']['knowledge']) for s in self.all_snapshots]

        # 学习收益
        gains = [s.get('knowledge_gain', 0) for s in self.all_snapshots]

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # 1. 认知能力发展
        axes[0, 0].plot(days, reasoning, 'b-', label='逻辑推理', alpha=0.7)
        axes[0, 0].plot(days, abstract, 'r-', label='抽象思维', alpha=0.7)
        axes[0, 0].set_title('认知能力发展')
        axes[0, 0].set_xlabel('天数')
        axes[0, 0].set_ylabel('能力值')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 动机变化
        axes[0, 1].plot(days, motivation, 'g-', alpha=0.7)
        axes[0, 1].set_title('动机水平变化')
        axes[0, 1].set_xlabel('天数')
        axes[0, 1].set_ylabel('动机')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 知识领域增长
        axes[1, 0].plot(days, knowledge_domains, 'purple', alpha=0.7)
        axes[1, 0].set_title('知识领域扩展')
        axes[1, 0].set_xlabel('天数')
        axes[1, 0].set_ylabel('领域数量')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 日学习收益（平滑）
        if len(gains) > 30:
            window = 30
            gains_smooth = np.convolve(gains, np.ones(window) / window, mode='valid')
            days_smooth = days[window - 1:]
            axes[1, 1].plot(days_smooth, gains_smooth, 'orange', alpha=0.7)
            axes[1, 1].set_title(f'学习收益（{window}天平滑）')
            axes[1, 1].set_xlabel('天数')
            axes[1, 1].set_ylabel('日收益')
            axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(f'{self.name}的发展轨迹（{self.total_years}年）', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()

        # 打印世界模型学到的知识
        print("\n🧠 世界模型学到的规律:")
        for key, value in self.student.world_model.learned_patterns.items():
            if isinstance(value, list) and len(value) > 0:
                if len(value) > 3:
                    print(f"   {key}: {value[:3]}... (共{len(value)}条)")
                else:
                    print(f"   {key}: {value}")
            else:
                print(f"   {key}: {value}")


# ======================
# 演示函数
# ======================

def main():
    """主演示"""
    print("=" * 60)
    print("🔥 AI学生：单一智能体的4380次展开")
    print("=" * 60)
    print("核心理解:")
    print("  不是4380个并行智能体")
    print("  而是1个智能体 × 4380次状态演化")
    print("  每天生成一个'快照'（第N天的认知状态）")
    print("=" * 60)

    # 创建系统（12年，4380天）
    system = AIStudentSystem(
        name="小明",
        start_age=6,
        total_years=3  # 演示用3年，实际可以12年
    )

    # 模拟发展
    input("\n按回车开始模拟...")
    snapshots = system.simulate_full_development()

    # 展示关键快照
    print("\n📸 关键日期的智能体快照:")

    # 第1天
    day1 = system.get_agent_at_day(0)
    if day1:
        print(f"\nDay 1 (6岁第1天):")
        print(f"  认知状态: 逻辑推理={day1['state_snapshot']['cognitive']['logical_reasoning']:.2f}")
        print(f"  知识领域: {len(day1['state_snapshot']['knowledge'])}个")

    # 第365天（1年后）
    day365 = system.get_agent_at_day(364)
    if day365:
        print(f"\nDay 365 (7岁):")
        print(f"  认知状态: 逻辑推理={day365['state_snapshot']['cognitive']['logical_reasoning']:.2f}")
        print(f"  知识领域: {len(day365['state_snapshot']['knowledge'])}个")

    # 最后一天
    last_day = system.get_agent_at_day(len(snapshots) - 1)
    if last_day:
        print(f"\nDay {len(snapshots)} ({last_day['age_years']}岁):")
        print(f"  认知状态: 逻辑推理={last_day['state_snapshot']['cognitive']['logical_reasoning']:.2f}")
        print(f"  知识领域: {len(last_day['state_snapshot']['knowledge'])}个")

        if last_day['state_snapshot']['knowledge']:
            top_3 = sorted(last_day['state_snapshot']['knowledge'].items(),
                           key=lambda x: x[1], reverse=True)[:3]
            print(f"  掌握最好的3个领域:")
            for topic, level in top_3:
                print(f"    {topic}: {level:.2f}")

    # 可视化
    input("\n按回车查看可视化图表...")
    system.visualize_development()

    # 展示智能体的连续性
    print("\n" + "=" * 60)
    print("🔄 智能体连续性验证:")
    print("=" * 60)

    # 检查相邻天的状态变化
    for i in [0, 100, 200, 364]:
        if i + 1 < len(snapshots):
            day_i = snapshots[i]
            day_next = snapshots[i + 1]

            if 'logical_reasoning' in day_i['state_snapshot']['cognitive'] and 'logical_reasoning' in \
                    day_next['state_snapshot']['cognitive']:
                reasoning_diff = abs(
                    day_next['state_snapshot']['cognitive']['logical_reasoning'] -
                    day_i['state_snapshot']['cognitive']['logical_reasoning']
                )

                print(f"Day {i} → Day {i + 1}:")
                print(f"  逻辑推理变化: {reasoning_diff:.4f} (微小连续变化)")
                print(f"  是同一个智能体: {reasoning_diff < 0.1} ✓")

    print("\n" + "=" * 60)
    print("✅ 演示完成!")
    print(f"   证明了: 4380个agent = 1个智能体的4380次展开")
    print("=" * 60)


# ======================
# 运行
# ======================

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 程序被用户中断")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        import traceback

        traceback.print_exc()