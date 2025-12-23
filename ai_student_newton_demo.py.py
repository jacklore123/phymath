import math

# =========================
# 1. 空白学生体（只会思考，不会答案）
# =========================
class BlankStudent:
    def __init__(self):
        self.thought_log = []

    def think(self, action):
        self.thought_log.append(action)
        print(f"[空白学生体思考] {action}")


# =========================
# 2. 牛顿认知模型（示范轨迹）
# =========================
class NewtonCognition:
    def __init__(self, student):
        self.student = student

    def build_uniform_motion(self):
        self.student.think("建立匀变速直线运动模型")

    def decide_formula(self, formula):
        self.student.think(f"选择公式：{formula}")

    def check_reasonable(self):
        self.student.think("进行物理合理性判断，舍去不合理解")


# =========================
# 3. 物理求解器（真正算数的地方）
# =========================
class PhysicsSolver:
    def __init__(self, v0, a):
        self.v0 = v0
        self.a = a

    def velocity(self, t):
        return self.v0 + self.a * t

    def displacement(self, t):
        return self.v0 * t + 0.5 * self.a * t * t

    def stop_time(self):
        return -self.v0 / self.a if self.a != 0 else None

    def displacement_in_time(self, t):
        t_stop = self.stop_time()
        if t_stop and t > t_stop:
            return self.displacement(t_stop)
        return self.displacement(t)

    def displacement_before_stop(self, dt):
        t_stop = self.stop_time()
        v_start = self.velocity(t_stop - dt)
        return (v_start + 0) / 2 * dt

    def time_for_displacement(self, s):
        a = 0.5 * self.a
        b = self.v0
        c = -s
        delta = b * b - 4 * a * c
        if delta < 0:
            return None
        t1 = (-b + math.sqrt(delta)) / (2 * a)
        t2 = (-b - math.sqrt(delta)) / (2 * a)
        return min(t for t in [t1, t2] if t >= 0)


# =========================
# 4. 三道题统一求解流程
# =========================
def solve_problem(problem_name, solver, cognition):
    print("\n==============================")
    print(f"【开始解题】{problem_name}")
    print("==============================")

    cognition.build_uniform_motion()

    if problem_name == "自由落体":
        cognition.decide_formula("v = gt")
        t = solver.velocity(0) / 10
        print(f"（1）落地时间 t = {t:.2f} s")

        cognition.decide_formula("位移差法")
        s_last = 0.5 * 10 * (t*t - (t-1)*(t-1))
        print(f"（2）最后 1s 位移 = {s_last:.2f} m")

    elif problem_name == "刹车问题":
        cognition.decide_formula("v = v0 + at")
        s6 = solver.displacement_in_time(6)
        print(f"（1）6s 内位移 = {s6:.2f} m")

        cognition.decide_formula("平均速度法")
        s2 = solver.displacement_before_stop(2)
        print(f"（2）静止前 2s 位移 = {s2:.2f} m")

        cognition.decide_formula("位移方程反解时间")
        t1 = solver.time_for_displacement(16)
        print(f"（3）前进 16m 用时 t = {t1:.2f} s")

    cognition.check_reasonable()


# =========================
# 5. 主程序入口
# =========================
if __name__ == "__main__":
    # 创建空白学生体
    student = BlankStudent()
    cognition = NewtonCognition(student)

    # 题目一：自由落体
    free_fall_solver = PhysicsSolver(v0=10, a=-10)
    solve_problem("自由落体", free_fall_solver, cognition)

    # 题目二 & 三：汽车刹车
    brake_solver = PhysicsSolver(v0=20, a=-4)
    solve_problem("刹车问题", brake_solver, cognition)

    print("\n==============================")
    print("【空白学生体完整思考轨迹】")
    for i, t in enumerate(student.thought_log, 1):
        print(f"{i}. {t}")
