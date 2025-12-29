"""
🌱 智能雏形系统 - 可生长的认知生命体
从种子到森林的完整生长过程
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import random
import math
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import pickle
import os
from pathlib import Path

# 设置matplotlib避免中文警告
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


# ======================
# 🧬 1. 核心生长引擎
# ======================

class GrowthEngine:
    """生长引擎 - 驱动智能雏形发展的核心"""

    def __init__(self, dna_blueprint=None):
        # 🧬 DNA蓝图：定义生长潜力和方向
        self.dna = dna_blueprint or self._create_default_dna()

        # 🌱 生长阶段
        self.growth_stages = {
            "seed": {"age_range": (0, 3), "focus": "基础结构建立"},
            "sprout": {"age_range": (3, 6), "focus": "快速吸收"},
            "sapling": {"age_range": (6, 12), "focus": "系统扩展"},
            "young_tree": {"age_range": (12, 18), "focus": "专业化"},
            "mature_tree": {"age_range": (18, 30), "focus": "深化整合"},
            "forest": {"age_range": (30, 100), "focus": "创造传承"}
        }

        # 🔥 生长动力
        self.growth_drivers = {
            "curiosity": 0.8,  # 探索未知的欲望
            "mastery_urge": 0.7,  # 掌握技能的冲动
            "meaning_seeking": 0.6,  # 寻求意义和理解
            "competence_need": 0.5,  # 变得有能力的需要
            "autonomy_drive": 0.4  # 自主性需求
        }

        # 📈 生长记录
        self.growth_history = []
        self.growth_milestones = []

        print("🧬 生长引擎初始化完成")
        print(f"   DNA特征: {len(self.dna['traits'])}个特质")
        print(f"   生长阶段: {len(self.growth_stages)}个阶段")

    def _create_default_dna(self):
        """创建默认DNA蓝图"""
        return {
            "traits": {
                "learning_speed": random.uniform(0.3, 0.9),
                "memory_capacity": random.uniform(0.4, 0.8),
                "creativity_potential": random.uniform(0.2, 0.7),
                "analytical_strength": random.uniform(0.3, 0.8),
                "social_intelligence": random.uniform(0.3, 0.7),
                "resilience": random.uniform(0.4, 0.9)
            },
            "preferences": {
                "preferred_learning_style": random.choice(["visual", "auditory", "kinesthetic", "logical"]),
                "optimal_learning_time": random.choice(["morning", "afternoon", "evening", "night"]),
                "interaction_preference": random.choice(["individual", "group", "mixed"]),
                "challenge_tolerance": random.uniform(0.3, 0.8)
            },
            "growth_patterns": {
                "burst_growth_frequency": random.uniform(0.05, 0.2),
                "consolidation_periods": random.randint(3, 10),
                "plateau_length": random.randint(5, 20),
                "breakthrough_threshold": random.uniform(0.6, 0.9)
            }
        }

    def get_current_stage(self, age_years):
        """获取当前生长阶段"""
        for stage_name, stage_info in self.growth_stages.items():
            start, end = stage_info["age_range"]
            if start <= age_years < end:
                return stage_name, stage_info
        return "seed", self.growth_stages["seed"]

    def calculate_growth_potential(self, current_state, environment):
        """计算生长潜力"""
        # 基础潜力
        base_potential = np.mean(list(self.dna["traits"].values()))

        # 环境匹配度
        env_match = self._calculate_environment_match(current_state, environment)

        # 内在动力
        intrinsic_motivation = np.mean(list(self.growth_drivers.values()))

        # 当前状态能量
        state_energy = current_state.get("energy", 0.5) * current_state.get("focus", 0.5)

        # 综合潜力
        total_potential = (
                base_potential * 0.3 +
                env_match * 0.3 +
                intrinsic_motivation * 0.2 +
                state_energy * 0.2
        )

        return min(1.0, max(0.1, total_potential))

    def _calculate_environment_match(self, state, environment):
        """计算与环境匹配度"""
        # 简化计算
        match_score = 0.5  # 基础匹配度

        # 学习风格匹配
        if "learning_style" in environment:
            if environment["learning_style"] == self.dna["preferences"]["preferred_learning_style"]:
                match_score += 0.2

        # 时间匹配
        if "time_of_day" in environment:
            if environment["time_of_day"] == self.dna["preferences"]["optimal_learning_time"]:
                match_score += 0.15

        # 社交环境匹配
        if "social_setting" in environment:
            if environment["social_setting"] == self.dna["preferences"]["interaction_preference"]:
                match_score += 0.15

        return min(1.0, match_score)

    def generate_growth_impulse(self, current_state, age_days):
        """生成生长冲动（决定今天如何生长）"""
        age_years = age_days / 365.0
        stage_name, stage_info = self.get_current_stage(age_years)

        # 基础生长类型
        growth_types = []

        # 基于DNA特质
        if self.dna["traits"]["creativity_potential"] > 0.6:
            growth_types.append("exploratory_growth")  # 探索性生长

        if self.dna["traits"]["analytical_strength"] > 0.6:
            growth_types.append("structured_growth")  # 结构性生长

        # 基于当前需求
        if current_state.get("knowledge_gap", 0) > 0.3:
            growth_types.append("gap_filling_growth")  # 填补缺口生长

        if current_state.get("curiosity", 0) > 0.7:
            growth_types.append("curiosity_driven_growth")  # 好奇心驱动生长

        # 基于生长阶段
        if stage_name in ["sprout", "sapling"]:
            growth_types.append("rapid_expansion")  # 快速扩展

        if stage_name in ["young_tree", "mature_tree"]:
            growth_types.append("deepening_growth")  # 深化生长

        # 如果没有生长类型，使用默认
        if not growth_types:
            growth_types = ["balanced_growth"]

        # 选择主要生长类型
        main_growth = random.choice(growth_types)

        return {
            "growth_type": main_growth,
            "growth_stage": stage_name,
            "stage_focus": stage_info["focus"],
            "potential_energy": self.calculate_growth_potential(current_state, {}),
            "growth_priority": self._determine_growth_priority(current_state),
            "recommended_duration": random.randint(30, 120)  # 推荐学习分钟数
        }

    def _determine_growth_priority(self, current_state):
        """确定生长优先级"""
        priorities = []

        # 检查认知短板
        cognitive_weaknesses = []
        for trait, value in self.dna["traits"].items():
            if value < 0.5 and trait in ["analytical_strength", "memory_capacity", "creativity_potential"]:
                cognitive_weaknesses.append(trait)

        if cognitive_weaknesses:
            priorities.append(f"strengthen_{random.choice(cognitive_weaknesses)}")

        # 检查知识结构
        if current_state.get("knowledge_diversity", 0) < 0.4:
            priorities.append("expand_knowledge_horizon")

        if current_state.get("skill_depth", 0) > 0.7:
            priorities.append("deepen_expertise")

        # 默认优先级
        if not priorities:
            priorities = ["balanced_development"]

        return random.choice(priorities)


# ======================
# 🧠 2. 认知架构（可生长）
# ======================

class GrowingCognitiveArchitecture:
    """可生长的认知架构"""

    def __init__(self, initial_complexity=10):
        self.components = {}
        self.connections = {}
        self.complexity = initial_complexity

        # 🏗️ 初始化基础架构
        self._initialize_foundation()

        # 📊 生长记录
        self.growth_log = []
        self.component_history = []

        print(f"🧠 认知架构初始化完成 - 初始复杂度: {self.complexity}")

    def _initialize_foundation(self):
        """初始化基础认知组件"""
        # 基础认知处理器
        self.components["working_memory"] = {
            "capacity": 0.3,
            "efficiency": 0.4,
            "age": 0,
            "growth_rate": 0.01
        }

        self.components["attention_system"] = {
            "focus": 0.5,
            "selectivity": 0.4,
            "sustained": 0.3,
            "age": 0,
            "growth_rate": 0.008
        }

        self.components["basic_reasoning"] = {
            "logical": 0.3,
            "causal": 0.2,
            "deductive": 0.3,
            "age": 0,
            "growth_rate": 0.012
        }

        # 初始化连接
        self.connections = {
            ("working_memory", "attention_system"): 0.5,
            ("working_memory", "basic_reasoning"): 0.4,
            ("attention_system", "basic_reasoning"): 0.3
        }

    def grow_for_day(self, growth_impulse, learning_experience):
        """一天的生长"""
        daily_growth = {}

        # 1. 组件自然生长（像肌肉锻炼）
        for comp_name, comp_data in self.components.items():
            natural_growth = comp_data.get("growth_rate", 0.005) * random.uniform(0.8, 1.2)

            # 应用生长
            for key in ["capacity", "efficiency", "focus", "selectivity", "sustained", "logical", "causal",
                        "deductive"]:
                if key in comp_data:
                    comp_data[key] = min(1.0, comp_data[key] + natural_growth)

            comp_data["age"] += 1
            daily_growth[comp_name] = natural_growth

        # 2. 基于生长冲动的专门生长
        growth_type = growth_impulse.get("growth_type", "")

        if "exploratory" in growth_type:
            # 探索性生长：可能发展新组件
            if random.random() < 0.05 and len(self.components) < 20:
                self._develop_new_component()

        elif "structured" in growth_type:
            # 结构性生长：强化现有连接
            self._strengthen_connections()

        elif "deepening" in growth_type:
            # 深化生长：提高现有组件效率
            self._deepen_existing_components()

        # 3. 基于学习经验的适应性生长
        if learning_experience.get("success", False):
            # 成功学习：强化相关组件
            relevant_comps = self._identify_relevant_components(learning_experience)
            for comp in relevant_comps:
                if comp in self.components:
                    for key in self.components[comp]:
                        if isinstance(self.components[comp][key], (int, float)):
                            self.components[comp][key] = min(1.0,
                                                             self.components[comp][key] * 1.02)

        # 4. 更新复杂度
        old_complexity = self.complexity
        self.complexity = self._calculate_complexity()

        # 记录生长
        growth_record = {
            "day": len(self.growth_log),
            "growth_type": growth_type,
            "component_growth": daily_growth,
            "complexity_change": self.complexity - old_complexity,
            "new_components": len(self.components) - len(daily_growth),
            "connection_strength": np.mean(list(self.connections.values())) if self.connections else 0
        }

        self.growth_log.append(growth_record)

        return growth_record

    def _develop_new_component(self):
        """发展新的认知组件"""
        # 潜在的新组件
        potential_components = [
            "abstract_thinking", "metacognition", "pattern_recognition",
            "conceptual_integration", "creative_synthesis", "critical_evaluation",
            "emotional_intelligence", "social_cognition", "temporal_reasoning",
            "spatial_reasoning", "moral_reasoning", "aesthetic_appreciation"
        ]

        # 选择尚未存在的组件
        available = [c for c in potential_components if c not in self.components]
        if not available:
            return

        new_component = random.choice(available)

        # 初始化新组件
        self.components[new_component] = {
            "strength": 0.1,
            "integration": 0.05,
            "utility": 0.1,
            "age": 0,
            "growth_rate": 0.015  # 新组件生长更快
        }

        # 建立连接（连接到最相关的现有组件）
        if self.components:
            existing = random.choice(list(self.components.keys()))
            if existing != new_component:
                self.connections[(existing, new_component)] = 0.1
                self.connections[(new_component, existing)] = 0.1

        print(f"   🌱 发展新认知组件: {new_component}")

    def _strengthen_connections(self):
        """强化连接"""
        if not self.connections:
            return

        # 随机选择一些连接进行强化
        connections_to_strengthen = random.sample(
            list(self.connections.keys()),
            min(3, len(self.connections))
        )

        for conn in connections_to_strengthen:
            self.connections[conn] = min(1.0, self.connections[conn] * 1.05)

    def _deepen_existing_components(self):
        """深化现有组件"""
        if not self.components:
            return

        # 选择一些组件进行深化
        components_to_deepen = random.sample(
            list(self.components.keys()),
            min(2, len(self.components))
        )

        for comp in components_to_deepen:
            for key in self.components[comp]:
                if isinstance(self.components[comp][key], (int, float)):
                    self.components[comp][key] = min(1.0,
                                                     self.components[comp][key] * 1.03)

    def _identify_relevant_components(self, learning_experience):
        """识别与学习经验相关的组件"""
        topic = learning_experience.get("topic", "")
        relevant = []

        # 基于主题的简单映射
        if "math" in topic.lower() or "logic" in topic.lower():
            relevant = ["basic_reasoning", "working_memory"]
        elif "creative" in topic.lower() or "art" in topic.lower():
            relevant = ["attention_system"]  # 注意系统也参与创造
        elif "social" in topic.lower() or "language" in topic.lower():
            relevant = ["working_memory", "attention_system"]

        return relevant

    def _calculate_complexity(self):
        """计算架构复杂度"""
        # 组件数量
        num_components = len(self.components)

        # 连接密度
        max_possible_connections = num_components * (num_components - 1)
        if max_possible_connections == 0:
            connection_density = 0
        else:
            connection_density = len(self.connections) / max_possible_connections

        # 组件成熟度
        avg_maturity = np.mean(
            [comp.get("age", 0) for comp in self.components.values()]) / 100 if self.components else 0

        # 连接强度
        avg_connection_strength = np.mean(list(self.connections.values())) if self.connections else 0

        # 综合复杂度
        complexity = (
                num_components * 0.3 +
                connection_density * 0.3 +
                avg_maturity * 0.2 +
                avg_connection_strength * 0.2
        )

        return complexity

    def get_architecture_summary(self):
        """获取架构摘要"""
        return {
            "total_components": len(self.components),
            "total_connections": len(self.connections),
            "average_component_age": np.mean(
                [c.get("age", 0) for c in self.components.values()]) if self.components else 0,
            "architecture_complexity": self.complexity,
            "recent_growth_rate": self._calculate_recent_growth_rate()
        }

    def _calculate_recent_growth_rate(self):
        """计算近期生长率"""
        if len(self.growth_log) < 10:
            return 0

        recent = self.growth_log[-10:]
        growth_rates = [g.get("complexity_change", 0) for g in recent]
        return np.mean(growth_rates)


# ======================
# 🌳 3. 知识森林（可生长）
# ======================

class KnowledgeForest:
    """知识森林 - 可生长的知识结构"""

    def __init__(self):
        # 🌲 知识树：领域→主题→知识点
        self.trees = {}  # 领域树
        self.roots = []  # 基础知识根节点
        self.cross_connections = {}  # 跨领域连接

        # 🌱 生长参数
        self.growth_zones = {
            "comfort_zone": [],  # 舒适区：已掌握
            "growth_zone": [],  # 生长区：正在学习
            "challenge_zone": []  # 挑战区：未来目标
        }

        # 🍃 知识叶子（具体知识点）
        self.leaves = {}
        self.leaf_lifespan = {}  # 叶子寿命（会遗忘）

        print("🌳 知识森林初始化完成")

    def plant_seed(self, domain, basic_concepts):
        """种植知识种子（创建新领域）"""
        if domain not in self.trees:
            self.trees[domain] = {
                "trunk": [],  # 主干知识
                "branches": {},  # 分支主题
                "depth": 0,  # 知识深度
                "breadth": 0,  # 知识广度
                "age_days": 0  # 领域年龄
            }

            # 添加基础概念作为根
            for concept in basic_concepts:
                leaf_id = self._create_leaf(domain, concept, "root", difficulty=0.3)
                self.trees[domain]["trunk"].append(leaf_id)
                self.roots.append(leaf_id)

            print(f"   🌱 种植新知识领域: {domain} (种子数: {len(basic_concepts)})")

    def grow_tree(self, domain, growth_impulse):
        """生长知识树"""
        if domain not in self.trees:
            return {"error": f"领域 {domain} 不存在"}

        tree = self.trees[domain]
        growth_results = {
            "new_leaves": 0,
            "deepened_branches": 0,
            "new_connections": 0
        }

        # 根据生长冲动决定生长方式
        growth_type = growth_impulse.get("growth_type", "")

        if "expansion" in growth_type:
            # 扩展生长：增加新分支
            growth_results["new_branches"] = self._expand_branches(domain)

        elif "deepening" in growth_type:
            # 深化生长：增加知识深度
            growth_results["deepened_branches"] = self._deepen_knowledge(domain)

        elif "integration" in growth_type:
            # 整合生长：建立跨领域连接
            growth_results["new_connections"] = self._create_cross_connections(domain)

        else:
            # 平衡生长：混合方式
            if random.random() > 0.5:
                growth_results["new_leaves"] = self._add_new_leaves(domain)
            else:
                growth_results["deepened_branches"] = self._deepen_existing(domain)

        # 更新树的状态
        tree["age_days"] += 1
        tree["depth"] = self._calculate_tree_depth(domain)
        tree["breadth"] = self._calculate_tree_breadth(domain)

        # 管理叶子生命周期（遗忘机制）
        self._manage_leaf_lifespan()

        return growth_results

    def _create_leaf(self, domain, concept, branch="main", difficulty=0.5):
        """创建知识叶子"""
        leaf_id = f"{domain}_{concept}_{len(self.leaves)}"

        self.leaves[leaf_id] = {
            "concept": concept,
            "domain": domain,
            "branch": branch,
            "understanding": 0.1,  # 初始理解度
            "retrieval_strength": 0.1,  # 提取强度
            "connections": [],
            "created_day": len(self.leaves),
            "difficulty": difficulty,
            "last_reviewed": 0
        }

        # 设置叶子寿命（基于难度）
        base_lifespan = 30  # 基础寿命30天
        difficulty_factor = 1.0 - (difficulty * 0.5)  # 难度越高越容易遗忘
        self.leaf_lifespan[leaf_id] = base_lifespan * difficulty_factor

        return leaf_id

    def _add_new_leaves(self, domain):
        """添加新叶子（新知识点）"""
        if domain not in self.trees:
            return 0

        tree = self.trees[domain]
        new_leaves = 0

        # 根据现有知识生成相关新概念
        existing_concepts = []
        for leaf_id in self.leaves.values():
            if leaf_id["domain"] == domain:
                existing_concepts.append(leaf_id["concept"])

        if existing_concepts:
            # 生成相关新概念
            base_concept = random.choice(existing_concepts)
            new_concept = f"{base_concept}_advanced_{random.randint(1, 3)}"

            # 确定分支
            if tree["branches"]:
                branch = random.choice(list(tree["branches"].keys()))
            else:
                branch = "main"

            # 创建新叶子
            leaf_id = self._create_leaf(domain, new_concept, branch, difficulty=0.6)

            # 添加到分支
            if branch not in tree["branches"]:
                tree["branches"][branch] = []
            tree["branches"][branch].append(leaf_id)

            # 建立连接
            if existing_concepts:
                related_leaf = None
                for lid, leaf in self.leaves.items():
                    if leaf["concept"] == base_concept and leaf["domain"] == domain:
                        related_leaf = lid
                        break

                if related_leaf:
                    self.leaves[leaf_id]["connections"].append(related_leaf)
                    self.leaves[related_leaf]["connections"].append(leaf_id)

            new_leaves += 1

        return new_leaves

    def _deepen_existing(self, domain):
        """深化现有知识"""
        deepened = 0

        # 随机选择一些叶子进行深化
        domain_leaves = [lid for lid, leaf in self.leaves.items() if leaf["domain"] == domain]
        if not domain_leaves:
            return 0

        leaves_to_deepen = random.sample(domain_leaves, min(3, len(domain_leaves)))

        for leaf_id in leaves_to_deepen:
            leaf = self.leaves[leaf_id]

            # 提高理解度和提取强度
            understanding_increase = random.uniform(0.02, 0.05)
            retrieval_increase = random.uniform(0.01, 0.03)

            leaf["understanding"] = min(1.0, leaf["understanding"] + understanding_increase)
            leaf["retrieval_strength"] = min(1.0, leaf["retrieval_strength"] + retrieval_increase)
            leaf["last_reviewed"] = len(self.leaves)

            # 延长寿命（复习巩固）
            if leaf_id in self.leaf_lifespan:
                self.leaf_lifespan[leaf_id] *= 1.1

            deepened += 1

        return deepened

    def _expand_branches(self, domain):
        """扩展新分支"""
        if domain not in self.trees:
            return 0

        tree = self.trees[domain]

        # 50%概率创建新分支
        if random.random() > 0.5 and len(tree["branches"]) < 10:
            new_branch = f"branch_{len(tree['branches']) + 1}"
            tree["branches"][new_branch] = []

            # 为新分支创建基础叶子
            base_concept = f"{domain}_fundamental_{new_branch}"
            leaf_id = self._create_leaf(domain, base_concept, new_branch, difficulty=0.4)
            tree["branches"][new_branch].append(leaf_id)

            return 1

        return 0

    def _create_cross_connections(self, domain):
        """创建跨领域连接"""
        if len(self.trees) < 2:
            return 0

        # 选择另一个领域
        other_domains = [d for d in self.trees.keys() if d != domain]
        if not other_domains:
            return 0

        other_domain = random.choice(other_domains)

        # 从每个领域选择一个叶子
        domain_leaves = [lid for lid, leaf in self.leaves.items() if leaf["domain"] == domain]
        other_leaves = [lid for lid, leaf in self.leaves.items() if leaf["domain"] == other_domain]

        if not domain_leaves or not other_leaves:
            return 0

        leaf1 = random.choice(domain_leaves)
        leaf2 = random.choice(other_leaves)

        # 创建连接
        connection_id = f"{leaf1}<->{leaf2}"
        if connection_id not in self.cross_connections:
            self.cross_connections[connection_id] = {
                "strength": 0.1,
                "domain1": domain,
                "domain2": other_domain,
                "leaf1": leaf1,
                "leaf2": leaf2
            }

            # 更新叶子连接
            self.leaves[leaf1]["connections"].append(leaf2)
            self.leaves[leaf2]["connections"].append(leaf1)

            return 1

        return 0

    def _manage_leaf_lifespan(self):
        """管理叶子生命周期（遗忘）"""
        leaves_to_remove = []

        for leaf_id, lifespan in list(self.leaf_lifespan.items()):
            # 减少寿命
            self.leaf_lifespan[leaf_id] -= 1

            # 如果寿命耗尽，理解度下降
            if lifespan <= 0:
                if leaf_id in self.leaves:
                    # 不是立即删除，而是理解度下降
                    self.leaves[leaf_id]["understanding"] *= 0.8
                    self.leaves[leaf_id]["retrieval_strength"] *= 0.7

                    # 重置寿命（但更短）
                    self.leaf_lifespan[leaf_id] = 15 * random.uniform(0.8, 1.2)

                    # 如果理解度太低，标记为遗忘
                    if self.leaves[leaf_id]["understanding"] < 0.1:
                        leaves_to_remove.append(leaf_id)

        # 移除完全遗忘的叶子
        for leaf_id in leaves_to_remove:
            if leaf_id in self.leaves:
                # 从所有连接中移除
                for other_id in self.leaves[leaf_id]["connections"]:
                    if other_id in self.leaves:
                        if leaf_id in self.leaves[other_id]["connections"]:
                            self.leaves[other_id]["connections"].remove(leaf_id)

                # 从知识树中移除
                domain = self.leaves[leaf_id]["domain"]
                if domain in self.trees:
                    tree = self.trees[domain]

                    # 从主干移除
                    if leaf_id in tree["trunk"]:
                        tree["trunk"].remove(leaf_id)

                    # 从分支移除
                    for branch_name, branch_leaves in tree["branches"].items():
                        if leaf_id in branch_leaves:
                            branch_leaves.remove(leaf_id)

                # 删除叶子
                del self.leaves[leaf_id]
                if leaf_id in self.leaf_lifespan:
                    del self.leaf_lifespan[leaf_id]

    def _calculate_tree_depth(self, domain):
        """计算树深度"""
        if domain not in self.trees:
            return 0

        # 简单估算：基于分支数量和叶子理解度
        tree = self.trees[domain]
        branch_count = len(tree["branches"])

        # 平均理解深度
        domain_leaves = [leaf for leaf in self.leaves.values() if leaf["domain"] == domain]
        if not domain_leaves:
            return 0

        avg_understanding = np.mean([leaf["understanding"] for leaf in domain_leaves])

        return branch_count * 0.3 + avg_understanding * 0.7

    def _calculate_tree_breadth(self, domain):
        """计算树广度"""
        if domain not in self.trees:
            return 0

        # 领域内的叶子数量
        domain_leaves = [leaf for leaf in self.leaves.values() if leaf["domain"] == domain]

        # 跨领域连接数量
        cross_conn_count = 0
        for conn in self.cross_connections.values():
            if conn["domain1"] == domain or conn["domain2"] == domain:
                cross_conn_count += 1

        return len(domain_leaves) * 0.7 + cross_conn_count * 0.3

    def get_forest_summary(self):
        """获取森林摘要"""
        return {
            "total_domains": len(self.trees),
            "total_leaves": len(self.leaves),
            "total_cross_connections": len(self.cross_connections),
            "average_understanding": np.mean(
                [leaf["understanding"] for leaf in self.leaves.values()]) if self.leaves else 0,
            "forest_health": self._calculate_forest_health()
        }

    def _calculate_forest_health(self):
        """计算森林健康度"""
        if not self.leaves:
            return 0

        # 理解度健康
        understanding_scores = [leaf["understanding"] for leaf in self.leaves.values()]
        understanding_health = np.mean(understanding_scores)

        # 连接健康
        connection_counts = [len(leaf["connections"]) for leaf in self.leaves.values()]
        avg_connections = np.mean(connection_counts) if connection_counts else 0
        connection_health = min(1.0, avg_connections / 5.0)  # 假设每个叶子理想连接5个

        # 多样性健康
        domain_count = len(self.trees)
        diversity_health = min(1.0, domain_count / 8.0)  # 假设理想有8个领域

        # 综合健康度
        total_health = (
                understanding_health * 0.4 +
                connection_health * 0.3 +
                diversity_health * 0.3
        )

        return total_health


# ======================
# 🌟 4. 智能雏形本体
# ======================

class IntelligentGerm:
    """智能雏形 - 可生长的认知生命体"""

    def __init__(self, name="智能雏形", initial_age=6):
        self.name = name
        self.age_days = initial_age * 365

        # 🧬 核心生长系统
        self.growth_engine = GrowthEngine()
        self.cognitive_architecture = GrowingCognitiveArchitecture()
        self.knowledge_forest = KnowledgeForest()

        # 🎯 当前状态
        self.current_state = self._initialize_state()
        self.daily_experiences = []
        self.growth_history = []

        # 🌱 初始知识种子
        self._plant_initial_seeds()

        print("=" * 60)
        print(f"🌟 智能雏形 '{name}' 创建成功!")
        print(f"   初始年龄: {initial_age}岁")
        print(f"   认知架构: {self.cognitive_architecture.complexity:.2f} 复杂度")
        print(f"   DNA特质: {len(self.growth_engine.dna['traits'])}个")
        print("=" * 60)

    def _initialize_state(self):
        """初始化当前状态"""
        return {
            "energy": 0.8,
            "focus": 0.6,
            "curiosity": 0.7,
            "motivation": 0.7,
            "confidence": 0.5,
            "stress": 0.2,
            "knowledge_diversity": 0.3,
            "skill_depth": 0.2,
            "cognitive_flexibility": 0.4
        }

    def _plant_initial_seeds(self):
        """种植初始知识种子"""
        # 基础领域种子
        basic_domains = {
            "语言": ["字母", "基础词汇", "简单句子"],
            "数学": ["数字", "计数", "基本形状"],
            "世界认知": ["颜色", "动物", "家庭"]
        }

        for domain, concepts in basic_domains.items():
            self.knowledge_forest.plant_seed(domain, concepts)

    def live_one_day(self, daily_environment=None):
        """度过一天（完整的生长周期）"""
        if daily_environment is None:
            daily_environment = self._generate_daily_environment()

        # 1. 早晨：获取生长冲动
        growth_impulse = self.growth_engine.generate_growth_impulse(
            self.current_state, self.age_days
        )

        # 2. 上午：认知架构生长
        arch_growth = self.cognitive_architecture.grow_for_day(
            growth_impulse, {}
        )

        # 3. 下午：知识森林生长
        # 选择今天重点生长的领域
        domains = list(self.knowledge_forest.trees.keys())
        if domains:
            focus_domain = random.choice(domains)
            forest_growth = self.knowledge_forest.grow_tree(
                focus_domain, growth_impulse
            )
        else:
            forest_growth = {}

        # 4. 学习体验（模拟）
        learning_experience = self._simulate_learning_experience(
            growth_impulse, daily_environment
        )

        # 5. 状态更新
        self._update_daily_state(learning_experience)

        # 6. 年龄增长
        self.age_days += 1

        # 7. 记录这一天
        daily_record = {
            "day": self.age_days - 1,
            "age_years": (self.age_days - 1) / 365.0,
            "growth_impulse": growth_impulse,
            "cognitive_growth": arch_growth,
            "knowledge_growth": forest_growth,
            "learning_experience": learning_experience,
            "current_state": self.current_state.copy(),
            "architecture_summary": self.cognitive_architecture.get_architecture_summary(),
            "forest_summary": self.knowledge_forest.get_forest_summary(),
            "environment": daily_environment
        }

        self.daily_experiences.append(daily_record)
        self.growth_history.append(daily_record)

        # 检查里程碑
        self._check_milestones(daily_record)

        return daily_record

    def _generate_daily_environment(self):
        """生成每日环境"""
        environments = [
            {"type": "structured", "richness": 0.7, "challenge": 0.5},
            {"type": "exploratory", "richness": 0.8, "challenge": 0.4},
            {"type": "social", "richness": 0.6, "challenge": 0.3},
            {"type": "creative", "richness": 0.9, "challenge": 0.6}
        ]

        return random.choice(environments)

    def _simulate_learning_experience(self, growth_impulse, environment):
        """模拟学习体验"""
        # 基于生长类型和环境生成学习体验
        growth_type = growth_impulse.get("growth_type", "")

        experiences = {
            "exploratory_growth": {"success": 0.6, "insights": 2, "struggle": 0.3},
            "structured_growth": {"success": 0.8, "insights": 1, "struggle": 0.2},
            "gap_filling_growth": {"success": 0.7, "insights": 1, "struggle": 0.4},
            "curiosity_driven_growth": {"success": 0.5, "insights": 3, "struggle": 0.5},
            "rapid_expansion": {"success": 0.6, "insights": 2, "struggle": 0.6},
            "deepening_growth": {"success": 0.7, "insights": 1, "struggle": 0.3},
            "balanced_growth": {"success": 0.7, "insights": 2, "struggle": 0.4}
        }

        base_exp = experiences.get(growth_type, {"success": 0.6, "insights": 1, "struggle": 0.4})

        # 环境调整
        env_factor = environment.get("richness", 0.5)
        success_rate = base_exp["success"] * (0.8 + env_factor * 0.4)

        # 状态调整
        state_factor = self.current_state.get("focus", 0.5) * 0.3 + self.current_state.get("motivation", 0.5) * 0.3
        success_rate *= (0.7 + state_factor * 0.6)

        # 确定是否成功
        success = random.random() < success_rate

        return {
            "success": success,
            "growth_type": growth_type,
            "insights_gained": base_exp["insights"] + (1 if success else 0),
            "struggle_level": base_exp["struggle"] * random.uniform(0.8, 1.2),
            "environment_match": env_factor,
            "state_support": state_factor,
            "topic": self._select_learning_topic(growth_impulse)
        }

    def _select_learning_topic(self, growth_impulse):
        """选择学习主题"""
        domains = list(self.knowledge_forest.trees.keys())
        if not domains:
            return "general_knowledge"

        growth_priority = growth_impulse.get("growth_priority", "")

        if "expand" in growth_priority:
            # 扩展：可能探索新领域或深化现有
            if random.random() < 0.3 and len(domains) < 8:
                new_domain = f"新领域_{len(domains) + 1}"
                return f"探索{new_domain}"
            else:
                return f"深化{random.choice(domains)}"
        elif "strengthen" in growth_priority:
            # 强化：选择需要加强的领域
            weak_domains = [d for d in domains if self.knowledge_forest.trees[d]["depth"] < 0.5]
            if weak_domains:
                return f"强化{random.choice(weak_domains)}"
            else:
                return f"深化{random.choice(domains)}"
        else:
            # 平衡：随机选择
            return random.choice(domains)

    def _update_daily_state(self, learning_experience):
        """更新每日状态"""
        # 能量消耗
        energy_cost = 0.1 + learning_experience.get("struggle_level", 0.3) * 0.2
        self.current_state["energy"] = max(0.1, self.current_state["energy"] - energy_cost)

        # 成功学习提升信心和动机
        if learning_experience.get("success", False):
            self.current_state["confidence"] = min(1.0, self.current_state["confidence"] + 0.03)
            self.current_state["motivation"] = min(1.0, self.current_state["motivation"] + 0.02)
        else:
            # 失败适当降低，但保持韧性
            self.current_state["confidence"] = max(0.2, self.current_state["confidence"] - 0.02)
            self.current_state["motivation"] = max(0.3, self.current_state["motivation"] - 0.01)

        # 好奇心波动
        curiosity_change = random.uniform(-0.05, 0.05)
        self.current_state["curiosity"] = max(0.3, min(1.0,
                                                       self.current_state["curiosity"] + curiosity_change))

        # 压力管理
        struggle = learning_experience.get("struggle_level", 0.3)
        stress_increase = struggle * 0.1
        stress_decay = 0.05  # 自然衰减
        self.current_state["stress"] = max(0.0, min(0.8,
                                                    self.current_state["stress"] + stress_increase - stress_decay))

        # 更新知识多样性（基于森林状态）
        forest_summary = self.knowledge_forest.get_forest_summary()
        self.current_state["knowledge_diversity"] = min(1.0,
                                                        forest_summary.get("total_domains", 0) / 10.0)

        # 更新技能深度（基于平均理解度）
        self.current_state["skill_depth"] = min(1.0,
                                                forest_summary.get("average_understanding", 0) * 1.2)

    def _check_milestones(self, daily_record):
        """检查生长里程碑"""
        age_years = daily_record["age_years"]
        arch_summary = daily_record["architecture_summary"]
        forest_summary = daily_record["forest_summary"]

        milestones = []

        # 认知架构里程碑
        if arch_summary["total_components"] >= 10:
            milestones.append("认知架构达到10个组件")

        if arch_summary["architecture_complexity"] >= 5.0:
            milestones.append("认知复杂度突破5.0")

        # 知识森林里程碑
        if forest_summary["total_domains"] >= 5:
            milestones.append("知识领域达到5个")

        if forest_summary["average_understanding"] >= 0.7:
            milestones.append("平均理解度达到70%")

        if forest_summary["forest_health"] >= 0.8:
            milestones.append("知识森林健康度优秀")

        # 年龄里程碑
        if age_years >= 7 and "小学阶段开始" not in [m.get("title", "") for m in self.growth_engine.growth_milestones]:
            milestones.append("小学阶段开始")

        if age_years >= 12 and "中学阶段开始" not in [m.get("title", "") for m in self.growth_engine.growth_milestones]:
            milestones.append("中学阶段开始")

        # 记录里程碑
        for milestone in milestones:
            if milestone not in [m.get("title", "") for m in self.growth_engine.growth_milestones]:
                self.growth_engine.growth_milestones.append({
                    "title": milestone,
                    "age_years": age_years,
                    "day": daily_record["day"],
                    "arch_complexity": arch_summary["architecture_complexity"],
                    "forest_health": forest_summary["forest_health"]
                })
                print(f"   🏆 里程碑达成: {milestone} (年龄{age_years:.1f}岁)")

    def grow_for_period(self, years=1, show_progress=True):
        """生长一段时间"""
        total_days = years * 365

        if show_progress:
            print(f"\n🌱 开始{years}年生长周期 ({total_days}天)...")

        records = []

        for day in range(total_days):
            record = self.live_one_day()
            records.append(record)

            # 进度显示
            if show_progress and day % 100 == 0:
                self._show_progress(day, total_days, record)

        if show_progress:
            print(f"\n✅ {years}年生长完成!")
            self._show_final_summary()

        return records

    def _show_progress(self, current_day, total_days, record):
        """显示进度"""
        progress = (current_day + 1) / total_days * 100
        age_years = record["age_years"]

        arch = record["architecture_summary"]
        forest = record["forest_summary"]

        print(f"   📅 进度: {progress:.1f}% | 年龄: {age_years:.1f}岁")
        print(f"     认知组件: {arch['total_components']}个 | 复杂度: {arch['architecture_complexity']:.2f}")
        print(f"     知识领域: {forest['total_domains']}个 | 健康度: {forest['forest_health']:.2f}")

    def _show_final_summary(self):
        """显示最终摘要"""
        print("\n" + "=" * 60)
        print(f"🌟 智能雏形 '{self.name}' 生长报告")
        print("=" * 60)

        # 基础信息
        print(f"📊 基础信息:")
        print(f"   最终年龄: {self.age_days / 365:.1f}岁")
        print(f"   总生长天数: {len(self.growth_history)}天")
        print(f"   里程碑数量: {len(self.growth_engine.growth_milestones)}个")

        # 认知架构
        arch_summary = self.cognitive_architecture.get_architecture_summary()
        print(f"\n🧠 认知架构:")
        print(f"   组件数量: {arch_summary['total_components']}")
        print(f"   连接数量: {arch_summary['total_connections']}")
        print(f"   架构复杂度: {arch_summary['architecture_complexity']:.2f}")
        print(f"   近期生长率: {arch_summary['recent_growth_rate']:.4f}/天")

        # 知识森林
        forest_summary = self.knowledge_forest.get_forest_summary()
        print(f"\n🌳 知识森林:")
        print(f"   领域数量: {forest_summary['total_domains']}")
        print(f"   知识叶子: {forest_summary['total_leaves']}")
        print(f"   跨领域连接: {forest_summary['total_cross_connections']}")
        print(f"   平均理解度: {forest_summary['average_understanding']:.2%}")
        print(f"   森林健康度: {forest_summary['forest_health']:.2f}")

        # 当前状态
        print(f"\n💡 当前状态:")
        for key, value in self.current_state.items():
            print(f"   {key}: {value:.2f}")

        # 里程碑
        if self.growth_engine.growth_milestones:
            print(f"\n🏆 重要里程碑:")
            for i, milestone in enumerate(self.growth_engine.growth_milestones[-5:]):  # 显示最近5个
                print(f"   {i + 1}. {milestone['title']} (年龄{milestone['age_years']:.1f}岁)")

        print("=" * 60)

    def visualize_growth(self):
        """可视化生长过程"""
        if len(self.growth_history) < 10:
            print("❌ 生长数据不足，至少需要10天数据")
            return

        # 提取数据
        days = [r["day"] for r in self.growth_history]
        ages = [r["age_years"] for r in self.growth_history]

        # 认知复杂度
        complexities = [r["architecture_summary"]["architecture_complexity"] for r in self.growth_history]

        # 知识森林健康度
        forest_health = [r["forest_summary"]["forest_health"] for r in self.growth_history]

        # 当前状态（能量和动机）
        energies = [r["current_state"]["energy"] for r in self.growth_history]
        motivations = [r["current_state"]["motivation"] for r in self.growth_history]

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # 1. 认知复杂度生长
        axes[0, 0].plot(days, complexities, 'b-', linewidth=2, alpha=0.7)
        axes[0, 0].set_title('Cognitive Architecture Growth')
        axes[0, 0].set_xlabel('Days')
        axes[0, 0].set_ylabel('Complexity')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 知识森林健康度
        axes[0, 1].plot(days, forest_health, 'g-', linewidth=2, alpha=0.7)
        axes[0, 1].set_title('Knowledge Forest Health')
        axes[0, 1].set_xlabel('Days')
        axes[0, 1].set_ylabel('Health Index')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 能量水平
        axes[1, 0].plot(days, energies, 'orange', linewidth=2, alpha=0.7)
        axes[1, 0].set_title('Energy Level')
        axes[1, 0].set_xlabel('Days')
        axes[1, 0].set_ylabel('Energy')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 动机水平
        axes[1, 1].plot(days, motivations, 'purple', linewidth=2, alpha=0.7)
        axes[1, 1].set_title('Motivation Level')
        axes[1, 1].set_xlabel('Days')
        axes[1, 1].set_ylabel('Motivation')
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(f'{self.name} Growth Trajectory', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()


# ======================
# 🚀 5. 演示主函数
# ======================

def main():
    """主演示函数"""
    print("=" * 60)
    print("🌟 智能雏形系统演示")
    print("=" * 60)
    print("这是一个可以自主生长、进化的认知生命体。")
    print("它从简单的种子开始，通过每天的生长逐渐发展。")
    print("=" * 60)

    # 创建智能雏形
    germ = IntelligentGerm(
        name="CognitiveSeed",  # 名称
        initial_age=6  # 初始年龄6岁
    )

    # 生长3年演示（可以修改为更长）
    print("\n开始生长演示（3年）...")

    try:
        # 生长3年（1095天）
        records = germ.grow_for_period(years=3, show_progress=True)

        # 可视化
        print("\n生成生长可视化图表...")
        germ.visualize_growth()

        # 交互探索
        while True:
            print("\n" + "=" * 60)
            print("🔍 探索智能雏形:")
            print("1. 查看当前状态")
            print("2. 查看生长里程碑")
            print("3. 查看认知架构详情")
            print("4. 查看知识森林详情")
            print("5. 继续生长1年")
            print("6. 退出")
            print("=" * 60)

            choice = input("请选择 (1-6): ").strip()

            if choice == "1":
                # 查看当前状态
                print(f"\n💡 {germ.name} 当前状态 (年龄{germ.age_days / 365:.1f}岁):")
                for key, value in germ.current_state.items():
                    print(f"   {key}: {value:.2f}")

            elif choice == "2":
                # 查看生长里程碑
                milestones = germ.growth_engine.growth_milestones
                print(f"\n🏆 生长里程碑 (共{len(milestones)}个):")
                for i, milestone in enumerate(milestones[-10:]):  # 显示最近10个
                    print(f"   {i + 1}. {milestone['title']}")
                    print(f"      年龄: {milestone['age_years']:.1f}岁")
                    print(f"      认知复杂度: {milestone.get('arch_complexity', 0):.2f}")
                    print(f"      森林健康度: {milestone.get('forest_health', 0):.2f}")

            elif choice == "3":
                # 查看认知架构详情
                arch_summary = germ.cognitive_architecture.get_architecture_summary()
                print(f"\n🧠 认知架构详情:")
                print(f"   总组件数: {arch_summary['total_components']}")
                print(f"   总连接数: {arch_summary['total_connections']}")
                print(f"   平均组件年龄: {arch_summary['average_component_age']:.1f}天")
                print(f"   架构复杂度: {arch_summary['architecture_complexity']:.2f}")

                # 显示组件列表
                print(f"\n   当前组件:")
                components = list(germ.cognitive_architecture.components.keys())
                for i in range(0, len(components), 3):
                    print(f"     {', '.join(components[i:i + 3])}")

            elif choice == "4":
                # 查看知识森林详情
                forest_summary = germ.knowledge_forest.get_forest_summary()
                print(f"\n🌳 知识森林详情:")
                print(f"   总领域数: {forest_summary['total_domains']}")
                print(f"   总叶子数: {forest_summary['total_leaves']}")
                print(f"   跨领域连接: {forest_summary['total_cross_connections']}")
                print(f"   平均理解度: {forest_summary['average_understanding']:.2%}")
                print(f"   森林健康度: {forest_summary['forest_health']:.2f}")

                # 显示领域列表
                print(f"\n   当前知识领域:")
                domains = list(germ.knowledge_forest.trees.keys())
                for domain in domains:
                    tree = germ.knowledge_forest.trees[domain]
                    print(
                        f"     {domain}: 深度{tree['depth']:.2f}, 广度{tree['breadth']:.2f}, 年龄{tree['age_days']}天")

            elif choice == "5":
                # 继续生长1年
                print("\n继续生长1年...")
                germ.grow_for_period(years=1, show_progress=True)
                germ.visualize_growth()

            elif choice == "6":
                print("\n👋 退出演示")
                break

            else:
                print("❌ 无效选择")

    except KeyboardInterrupt:
        print("\n\n👋 演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示出错: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("✅ 智能雏形演示完成")
    print("=" * 60)


# ======================
# 运行
# ======================

if __name__ == "__main__":
    # 设置随机种子确保可重复性
    random.seed(42)
    np.random.seed(42)

    main()