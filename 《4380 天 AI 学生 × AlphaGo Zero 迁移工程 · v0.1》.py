import math
import random

# =========================
# 1. 空白学生体
# =========================
class BlankStudent:
    def __init__(self, name="学生"):
        self.name = name
        self.knowledge = {
            "v_at": 0.0,     # v = at
            "s_vt": 0.0,     # s = vt
            "s_at2": 0.0,   # s = 1/2 at^2
            "v2_2as": 0.0   # v^2 = 2as
        }
        self.experience = 0.0
        self.day = 0

    def mastery(self):
        return round(sum(self.knowledge.values()) / len(self.knowledge), 2)

# =========================
# 2. 牛顿榜样模型（理想认知）
# =========================
NEWTON_MODEL = {
    "v_at": 1.0,
    "s_vt": 1.0,
    "s_at2": 1.0,
    "v2_2as": 1.0
}

# =========================
# 3. 解题认知逻辑（核心）
# =========================
class PhysicsCognitionEngine:

    def solve(self, student, problem):
        thinking_log = []
        reward = 0.0

        thinking_log.append(f"【题目】{problem['desc']}")

        # Step 1：判断运动类型
        thinking_log.append("判断：是否为匀变速直线运动")
        is_uniform_acc = problem.get("a") is not None

        # Step 2：选择认知公式（仿生）
        if problem["type"] == "free_fall":
            thinking_log.append("识别为自由落体 → 使用 v=at, s=1/2at²")
            reward += self._use_formula(student, "v_at")
            reward += self._use_formula(student, "s_at2")

            t = problem["v"] / problem["g"]
            s_last = 0.5 * problem["g"] * (t**2 - (t-1)**2)

            thinking_log.append(f"计算：t={round(t,2)}s")
            thinking_log.append(f"最后1秒位移={round(s_last,2)}m")

        elif problem["type"] == "braking":
            thinking_log.append("识别为刹车问题 → 使用 v=at, s=vt")
            reward += self._use_formula(student, "v_at")
            reward += self._use_formula(student, "s_vt")

            a = (problem["v"] - problem["v0"]) / problem["t"]
            s6 = problem["v0"] * 6 + 0.5 * a * 36

            thinking_log.append(f"加速度 a={round(a,2)} m/s²")
            thinking_log.append(f"6秒位移={round(s6,2)}m")

        # Step 3：奖励与成长
        student.experience += reward
        thinking_log.append(f"获得经验值：{round(reward,2)}")
        thinking_log.append(f"当前掌握度：{student.mastery()}")

        return thinking_log

    def _use_formula(self, student, key):
        gain = 0.1 * (1 - student.knowledge[key])
        student.knowledge[key] += gain
        return gain

# =========================
# 4. 奖励函数（对齐牛顿）
# =========================
def alignment_reward(student):
    diff = 0
    for k in student.knowledge:
        diff += abs(student.knowledge[k] - NEWTON_MODEL[k])
    return round(1 - diff / len(student.knowledge), 3)

# =========================
# 5. 4380 天认知生长模拟器
# =========================
def simulate_learning(days=10):
    student = BlankStudent("空白学生体")
    engine = PhysicsCognitionEngine()

    problems = [
        {
            "type": "free_fall",
            "desc": "自由落体，落地速度10m/s，g=10",
            "v": 10,
            "g": 10
        },
        {
            "type": "braking",
            "desc": "汽车刹车，20m/s → 4m/s，用时4s",
            "v0": 20,
            "v": 4,
            "t": 4
        }
    ]

    for day in range(days):
        student.day += 1
        print(f"\n===== 第 {student.day} 天 =====")

        problem = random.choice(problems)
        logs = engine.solve(student, problem)

        for line in logs:
            print(line)

        print("对齐牛顿奖励：", alignment_reward(student))

# =========================
# 主程序
# =========================
if __name__ == "__main__":
    simulate_learning(days=10)

