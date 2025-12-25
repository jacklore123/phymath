import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from dataclasses import dataclass


# ============================================================================
# 版本1: 最简版本 - 2维状态 (知识+动力)
# ============================================================================
def simulate_simple_student(days=180, seed=42):
    """最简版本：只有知识和动力两个维度"""
    np.random.seed(seed)

    # 参数
    alpha = 0.05  # 学习效率
    beta = 0.01  # 遗忘强度
    gamma = 0.02  # 动力正反馈
    delta = 0.03  # 疲劳损耗

    # 初始状态
    K = 0.1  # 初始知识
    M = 0.6  # 初始动力

    # 课程输入
    curriculum = np.random.uniform(0.3, 0.8, days)

    # 历史记录
    K_hist, M_hist = [], []

    # 仿真循环
    for t in range(days):
        C = curriculum[t]

        # 状态转移
        forget = beta * K
        gain = alpha * C * M
        K = np.clip(K + gain - forget, 0, 1)

        reward = gamma * gain
        burnout = delta * C
        M = np.clip(M + reward - burnout, 0, 1)

        # 记录
        K_hist.append(K)
        M_hist.append(M)

    return K_hist, M_hist, curriculum


# ============================================================================
# 版本2: 增强版本 - 加入突破和卡顿机制
# ============================================================================
def simulate_student_with_breakthroughs(days=365, seed=42):
    """增强版本：加入卡顿、突破和挫败感"""
    np.random.seed(seed)

    # 参数
    alpha = 0.05  # 学习效率
    beta = 0.01  # 遗忘强度
    gamma = 0.02  # 动力正反馈
    delta = 0.03  # 疲劳损耗
    threshold_breakthrough = 0.7  # 突破阈值

    # 状态
    K = 0.1  # 知识
    M = 0.6  # 动力
    S = 0.0  # 挫败感
    struggle_duration = 0

    # 课程（加入阶段性挑战）
    curriculum = np.random.uniform(0.3, 0.8, days)
    for i in range(0, days, 30):
        if i + 5 < days:
            curriculum[i:i + 5] = np.random.uniform(0.7, 0.95, 5)

    # 历史记录
    K_hist, M_hist, S_hist = [], [], []
    breakthrough_days = []

    # 仿真
    for t in range(days):
        C = curriculum[t]

        # 检测卡顿
        is_struggling = (S > 0.6) and (K < threshold_breakthrough * 0.9)
        if is_struggling:
            struggle_duration += 1
            effective_alpha = alpha * 0.3
            S = min(S + 0.05, 1.0)
        else:
            struggle_duration = 0
            effective_alpha = alpha
            S = max(S - 0.01, 0)

        # 检查突破
        if K > threshold_breakthrough and struggle_duration > 3:
            K = min(K + 0.15, 1.0)
            M = min(M + 0.2, 1.0)
            S = max(S - 0.3, 0)
            breakthrough_days.append(t)
            struggle_duration = 0

        # 正常更新
        forget = beta * K
        gain = effective_alpha * C * M * (1 - S * 0.5)
        K = np.clip(K + gain - forget, 0, 1)

        reward = gamma * gain
        burnout = delta * C
        M = np.clip(M + reward - burnout, 0, 1)

        # 记录
        K_hist.append(K)
        M_hist.append(M)
        S_hist.append(S)

    return K_hist, M_hist, S_hist, breakthrough_days, curriculum


# ============================================================================
# 版本3: 多知识领域版本
# ============================================================================
def simulate_multi_domain_student(days=365, n_domains=3, seed=42):
    """多知识领域版本：模拟多个学科的学习"""
    np.random.seed(seed)

    # 参数
    alpha = 0.05
    beta = 0.01
    gamma = 0.02
    delta = 0.03

    # 初始状态
    K = np.array([0.1, 0.15, 0.05])  # 各领域初始知识
    M = 0.6  # 全局动力

    # 迁移矩阵（知识相互促进）
    transfer_matrix = np.array([
        [1.00, 0.20, 0.10],  # 领域1 -> 领域1,2,3
        [0.15, 1.00, 0.25],  # 领域2 -> 领域1,2,3
        [0.05, 0.10, 1.00]  # 领域3 -> 领域1,2,3
    ])

    # 课程（每天重点学习一个领域）
    curriculum = np.random.uniform(0.3, 0.8, days)

    # 历史记录
    K_history = []
    M_history = []

    for t in range(days):
        C = curriculum[t]
        focus_domain = t % n_domains

        # 计算学习增益
        gains = np.zeros(n_domains)
        gains[focus_domain] = alpha * C * M

        # 应用迁移学习
        effective_gains = transfer_matrix @ gains

        # 更新知识
        forget = beta * K
        K = np.clip(K + effective_gains - forget, 0, 1)

        # 更新动力
        total_gain = np.sum(effective_gains)
        reward = gamma * total_gain
        burnout = delta * C
        M = np.clip(M + reward - burnout, 0, 1)

        # 记录
        K_history.append(K.copy())
        M_history.append(M)

    return np.array(K_history), np.array(M_history), curriculum


# ============================================================================
# 版本4: 神经网络版本 - 使用PyTorch
# ============================================================================
class NeuralTransitionNet(nn.Module):
    """神经网络状态转移函数"""

    def __init__(self, state_dim=128, experience_dim=10, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + experience_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        # 初始化小权重，确保小步更新
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.01)

    def forward(self, state, experience):
        x = torch.cat([state, experience], dim=-1)
        delta = self.net(x) * 0.01  # 小步更新，模拟日常变化
        return state + delta


class DevelopmentalConstraint:
    """发展心理学约束"""

    @staticmethod
    def apply_stage_constraint(state_vector, day):
        """根据年龄施加发展阶段约束"""
        age_years = 6 + day / 365
        state = state_vector.clone()

        # 前运算阶段（约2-7岁）：抑制抽象思维
        if age_years < 7:
            abstract_indices = torch.arange(50, 70)
            state[abstract_indices] = state[abstract_indices] * 0.1

        # 具体运算阶段（约7-11岁）：允许具体逻辑
        elif age_years < 11:
            abstract_indices = torch.arange(50, 70)
            state[abstract_indices] = state[abstract_indices] * 0.5

        # 青春期：动力波动
        if 12 < age_years < 16:
            motivation_indices = torch.arange(100, 110)
            fluctuation = torch.randn_like(state[motivation_indices]) * 0.1
            state[motivation_indices] = state[motivation_indices] + fluctuation

        return state


class AIStudentNN:
    """神经网络版本的AI学生"""

    def __init__(self, state_dim=128, experience_dim=10):
        self.state_dim = state_dim
        self.experience_dim = experience_dim
        self.day = 0

        # 神经网络组件
        self.transition_net = NeuralTransitionNet(state_dim, experience_dim)
        self.constraint = DevelopmentalConstraint()

        # 初始状态（128维认知状态向量）
        self.state_vector = torch.randn(state_dim) * 0.1

        # 解码器：从状态向量到可解释特征
        self.decoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16)  # 16个可解释特征
        )

    def encode_experience(self, experience_dict):
        """编码日常经验为向量（简化版本）"""
        # 实际应用中这里应该是一个复杂的编码器
        exp_vector = torch.randn(self.experience_dim) * 0.5
        return exp_vector

    def decode_state(self, state_vector):
        """将状态向量解码为可解释特征"""
        with torch.no_grad():
            features = self.decoder(state_vector)
        return {
            "knowledge_level": features[0:5].mean().item(),
            "cognitive_ability": features[5:10].mean().item(),
            "motivation": features[10].item(),
            "frustration": features[11].item(),
            "curiosity": features[12].item(),
            "confidence": features[13].item(),
        }

    def step(self, daily_experience):
        """一天的状态转移"""
        # 编码经验
        exp_vector = self.encode_experience(daily_experience)

        # 神经网络更新
        with torch.no_grad():
            new_state = self.transition_net(
                self.state_vector.unsqueeze(0),
                exp_vector.unsqueeze(0)
            ).squeeze(0)

        # 应用发展约束
        new_state = self.constraint.apply_stage_constraint(new_state, self.day)

        # 更新状态
        self.state_vector = new_state
        self.day += 1

        # 解码为可解释特征
        features = self.decode_state(new_state)
        features["day"] = self.day

        return features

    def simulate(self, curriculum, days=180):
        """模拟多天发展"""
        trajectory = []
        for t in range(days):
            # 获取当天的经验（简化：随机生成）
            daily_exp = {"instruction": f"Day {t} lesson", "difficulty": np.random.uniform(0.3, 0.8)}

            # 执行一天的状态转移
            snapshot = self.step(daily_exp)
            trajectory.append(snapshot)

        return trajectory


# ============================================================================
# 可视化函数
# ============================================================================
def plot_simple_simulation(K_hist, M_hist, title="Simple AI Student Simulation"):
    """可视化最简版本"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 知识增长曲线
    axes[0].plot(K_hist, label="Knowledge", color='blue')
    axes[0].set_title("Knowledge Growth")
    axes[0].set_xlabel("Day")
    axes[0].set_ylabel("Knowledge Level")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 动力变化曲线
    axes[1].plot(M_hist, label="Motivation", color='orange')
    axes[1].set_title("Motivation Dynamics")
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("Motivation Level")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_breakthrough_simulation(K_hist, M_hist, S_hist, breakthroughs, title="AI Student with Breakthroughs"):
    """可视化带突破的版本"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 知识增长
    axes[0, 0].plot(K_hist, label="Knowledge", color='blue')
    for bd in breakthroughs:
        axes[0, 0].axvline(x=bd, color='green', alpha=0.3, linestyle=':',
                           label='Breakthrough' if bd == breakthroughs[0] else "")
    axes[0, 0].set_title("Knowledge with Breakthroughs")
    axes[0, 0].set_xlabel("Day")
    axes[0, 0].set_ylabel("Knowledge Level")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # 动力与挫败感
    axes[0, 1].plot(M_hist, label="Motivation", color='orange')
    axes[0, 1].plot(S_hist, label="Struggle", color='red', alpha=0.7)
    axes[0, 1].set_title("Motivation & Struggle")
    axes[0, 1].set_xlabel("Day")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # 相位图
    axes[1, 0].scatter(K_hist, M_hist, c=range(len(K_hist)), cmap='viridis', s=5, alpha=0.6)
    axes[1, 0].set_xlabel("Knowledge")
    axes[1, 0].set_ylabel("Motivation")
    axes[1, 0].set_title("Phase Space: Knowledge vs Motivation")
    axes[1, 0].grid(True, alpha=0.3)

    # 学习效率
    learning_efficiency = [0.05 * M_hist[i] * (1 - S_hist[i] * 0.5) for i in range(len(K_hist))]
    axes[1, 1].plot(learning_efficiency, color='purple')
    axes[1, 1].set_ylabel("Effective Learning Rate")
    axes[1, 1].set_xlabel("Day")
    axes[1, 1].set_title("Dynamic Learning Efficiency")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_multi_domain_simulation(K_history, domain_names=["Math", "Language", "Science"]):
    """可视化多领域版本"""
    plt.figure(figsize=(10, 5))
    colors = ['blue', 'orange', 'green', 'red', 'purple']

    for i in range(K_history.shape[1]):
        plt.plot(K_history[:, i], label=domain_names[i], color=colors[i], alpha=0.8)

    plt.title("Multi-Domain Knowledge Development")
    plt.xlabel("Day")
    plt.ylabel("Knowledge Level")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# ============================================================================
# 主程序：运行所有版本
# ============================================================================
def main():
    print("=" * 60)
    print("AI Student Development Simulation System")
    print("=" * 60)

    # 1. 运行最简版本
    print("\n[1] Running Simple Version (2D State)...")
    K_simple, M_simple, curriculum_simple = simulate_simple_student(days=180)
    plot_simple_simulation(K_simple, M_simple, "Simple AI Student (180 days)")

    # 2. 运行带突破版本
    print("[2] Running Version with Breakthroughs & Struggles...")
    K_break, M_break, S_break, breakthroughs, curriculum_break = simulate_student_with_breakthroughs(days=365)
    plot_breakthrough_simulation(K_break, M_break, S_break, breakthroughs, "AI Student with Breakthroughs (1 year)")
    print(f"   Breakthroughs detected at days: {breakthroughs}")

    # 3. 运行多领域版本
    print("[3] Running Multi-Domain Version...")
    K_multi, M_multi, _ = simulate_multi_domain_student(days=365, n_domains=3)
    plot_multi_domain_simulation(K_multi)

    # 4. 运行神经网络版本
    print("[4] Running Neural Network Version...")
    student_nn = AIStudentNN(state_dim=128, experience_dim=10)
    trajectory = student_nn.simulate([], days=100)  # 模拟100天

    # 提取神经网络版本的轨迹数据
    nn_days = [t["day"] for t in trajectory]
    nn_knowledge = [t["knowledge_level"] for t in trajectory]
    nn_motivation = [t["motivation"] for t in trajectory]
    nn_curiosity = [t["curiosity"] for t in trajectory]

    # 可视化神经网络版本
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(nn_knowledge, color='blue')
    axes[0, 0].set_title("NN: Knowledge Level")
    axes[0, 0].set_xlabel("Day")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(nn_motivation, color='orange')
    axes[0, 1].set_title("NN: Motivation")
    axes[0, 1].set_xlabel("Day")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(nn_curiosity, color='green')
    axes[1, 0].set_title("NN: Curiosity")
    axes[1, 0].set_xlabel("Day")
    axes[1, 0].grid(True, alpha=0.3)

    # 特征相关性
    axes[1, 1].scatter(nn_knowledge, nn_motivation, c=nn_curiosity, cmap='coolwarm', s=20, alpha=0.6)
    axes[1, 1].set_xlabel("Knowledge Level")
    axes[1, 1].set_ylabel("Motivation")
    axes[1, 1].set_title("NN: Knowledge vs Motivation (colored by Curiosity)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Neural Network AI Student Simulation", y=1.02)
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("Simulation Complete!")
    print("=" * 60)


# ============================================================================
# 运行入口
# ============================================================================
if __name__ == "__main__":
    main()