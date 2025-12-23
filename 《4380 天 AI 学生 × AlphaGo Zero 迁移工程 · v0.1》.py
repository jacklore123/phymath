import math
import random
import copy

# ======================
# 1. 空白学生体（World State）
# ======================

def create_blank_student():
    return {
        "level": 0.5,
        "attention": 0.8,
        "fatigue": 0.2,
        "thinking_speed": 1.0,
        "allowed_formulas": [],
        "max_reasoning_chain": 1
    }

# ======================
# 2. 牛顿认知生长模型（4380 天）
# ======================

def cognitive_stage(day):
    if day < 700:
        return "perception"
    elif day < 1500:
        return "rule"
    elif day < 2400:
        return "model"
    else:
        return "reasoning"

NEWTON_COGNITION = {
    "perception": {
        "formulas": ["v=s/t"],
        "chain": 1,
        "speed": 1.0
    },
    "rule": {
        "formulas": ["v=v0+at", "s=vt"],
        "chain": 2,
        "speed": 1.2
    },
    "model": {
        "formulas": ["s=v0t+1/2at^2"],
        "chain": 3,
        "speed": 1.5
    },
    "reasoning": {
        "formulas": ["free_combination"],
        "chain": 5,
        "speed": 2.0
    }
}

def newton_day_update(student, day):
    stage = cognitive_stage(day)
    cfg = NEWTON_COGNITION[stage]
    student["allowed_formulas"] = cfg["formulas"]
    student["max_reasoning_chain"] = cfg["chain"]
    student["thinking_speed"] = cfg["speed"]

# ======================
# 3. 世界模型（教学行为）
# ======================

def apply_teaching_action(state, action):
    efficiency = state["attention"] * (1 - state["fatigue"])

    if action == "讲解":
        gain = 0.15
        state["fatigue"] += 0.05
    elif action == "例题":
        gain = 0.25
        state["fatigue"] += 0.08
    elif action == "反思":
        gain = 0.35
        state["fatigue"] -= 0.1
    elif action == "互动学习":
        gain = 0.45
        state["fatigue"] += 0.05
    elif action == "休息":
        state["fatigue"] -= 0.2
        state["attention"] += 0.1
        return
    else:
        gain = 0.1

    state["fatigue"] = min(max(state["fatigue"], 0), 1)
    state["attention"] = min(max(state["attention"], 0), 1)

    state["level"] += gain * efficiency * state["thinking_speed"]

# ======================
# 4. 奖励函数（对齐牛顿）
# ======================

def alignment_reward(student):
    # 奖励 = 能力 × 稳定性 × 进度
    return (
        student["level"]
        * student["thinking_speed"]
        * (1 - abs(student["fatigue"] - 0.3))
    )

# ======================
# 5. 教育版 MCTS
# ======================

ACTIONS = ["讲解", "例题", "反思", "互动学习", "休息"]

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0

    def ucb(self, c=1.4):
        if self.visits == 0:
            return float("inf")
        return self.value / self.visits + c * math.sqrt(
            math.log(self.parent.visits + 1) / self.visits
        )

def select(node):
    while node.children:
        node = max(node.children, key=lambda n: n.ucb())
    return node

def expand(node):
    for action in ACTIONS:
        s = copy.deepcopy(node.state)
        apply_teaching_action(s, action)
        node.children.append(Node(s, node, action))

def rollout(state, depth=5):
    s = copy.deepcopy(state)
    for _ in range(depth):
        apply_teaching_action(s, random.choice(ACTIONS))
    return alignment_reward(s)

def backprop(node, reward):
    while node:
        node.visits += 1
        node.value += reward
        node = node.parent

def mcts_decide(state, iterations=30):
    root = Node(copy.deepcopy(state))
    for _ in range(iterations):
        leaf = select(root)
        expand(leaf)
        child = random.choice(leaf.children)
        reward = rollout(child.state)
        backprop(child, reward)
    return max(root.children, key=lambda n: n.visits).action

# ======================
# 6. 4380 天主模拟
# ======================

def simulate_4380_days():
    student = create_blank_student()
    history = []

    for day in range(4380):
        newton_day_update(student, day)
        action = mcts_decide(student)
        apply_teaching_action(student, action)

        history.append({
            "day": day,
            "stage": cognitive_stage(day),
            "level": round(student["level"], 2),
            "action": action
        })

        if day % 500 == 0:
            print(f"Day {day} | Stage {cognitive_stage(day)} | Level {student['level']:.2f}")

    return history

# ======================
# 7. 运行入口
# ======================

if __name__ == "__main__":
    print("🚀 启动 AI 学生 4380 天认知生长模拟")
    history = simulate_4380_days()
    print("✅ 模拟完成，总天数：", len(history))
