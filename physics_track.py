import random
import json

# ======================
# 学生状态（世界状态）
# ======================
student_state = {
    "age": 17,
    "subject": "physics",
    "module": "力学",
    "topic": "牛顿第二定律",
    "level": 2.6,
    "attention": 0.8,  # 专注度 0~1
    "fatigue": 0.2  # 疲劳度 0~1
}

# ======================
# 理想专家轨迹（牛顿）
# ======================
ideal_state = {
    "name": "牛顿",
    "subject": "physics",
    "module": "力学",
    "topic": "牛顿第二定律",
    "level": 4.5  # 理想掌握程度
}

# ======================
# 显示学习位置
# ======================
def show_learning_position(state):
    """
    打印当前学生的学习内容、进度、专注度和疲劳度。
    """
    topic = state["topic"]
    level = state["level"]
    attention = state["attention"]
    fatigue = state["fatigue"]

    print(f"学习内容：{topic}")
    print(f"当前轨迹位置：Level {round(level, 2)}")
    print(f"专注度：{round(attention, 2)} | 疲劳度：{round(fatigue, 2)}")

# ======================
# 进度条显示
# ======================
def show_progress_bar(state, max_level=5.0, bar_length=10):
    """
    显示学习进度条。
    """
    level = state["level"]
    if level < 0:
        level = 0
    if level > max_level:
        level = max_level

    filled_length = int(level / max_level * bar_length)
    empty_length = bar_length - filled_length
    bar = "■" * filled_length + "□" * empty_length
    print(f"学习进度：[{bar}] {round(level, 2)} / {max_level}")

# ======================
# 理想轨迹对齐对比
# ======================
def compare_with_ideal(student, ideal):
    """
    比较学生当前学习进度与理想轨迹之间的差距。
    """
    print("\n=== 学习轨迹对齐对比 ===")

    print(f"\n【理想轨迹 - {ideal['name']}】")
    show_progress_bar(ideal)

    print("\n【学生当前轨迹】")
    show_progress_bar(student)

    gap = ideal["level"] - student["level"]
    if gap > 0:
        print(f"\n⚠️ 学习差距：{round(gap, 2)}（尚未达到理想状态）")
    else:
        print("\n✅ 已达到或超过理想轨迹")

# ======================
# 模拟摄像头信号
# ======================
def simulate_camera_signal():
    """
    模拟摄像头信号，返回专注度与情绪波动。
    专注度：0~1，情绪波动：-0.5到0.5
    """
    attention_signal = random.uniform(0.7, 1)  # 随机生成专注度
    emotion_signal = random.uniform(-0.3, 0.3)  # 随机生成情绪波动
    return attention_signal, emotion_signal

# ======================
# 摄像头信号应用
# ======================
def apply_camera_signal(state):
    """
    根据摄像头信号调整学生的专注度和疲劳度。
    """
    attention_signal, emotion_signal = simulate_camera_signal()

    # 根据摄像头信号调整专注度和疲劳度
    state["attention"] = attention_signal  # 更新专注度
    state["fatigue"] += emotion_signal  # 情绪波动影响疲劳度（负值减轻疲劳，正值增加疲劳）

    # 确保疲劳度在[0, 1]之间
    state["fatigue"] = max(0, min(state["fatigue"], 1))

    # 输出调整后的结果
    print(f"摄像头信号 -> 专注度: {round(state['attention'], 2)} | 疲劳度: {round(state['fatigue'], 2)}")

# ======================
# 教学行为（世界演化规则）
# ======================
def apply_teaching_action(state, action):
    """
    根据教学行为调整学生的学习状态。
    """
    attention = state["attention"]
    fatigue = state["fatigue"]

    # 学习效率因子（越专注越好，越累越差）
    efficiency = attention * (1 - fatigue)

    if action == "讲解":
        base_gain = 0.2
        state["fatigue"] += 0.05
    elif action == "例题":
        base_gain = 0.3
        state["fatigue"] += 0.08
    elif action == "反思":
        base_gain = 0.4
        state["fatigue"] -= 0.1
    elif action == "休息":  # 添加休息行为
        base_gain = 0  # 休息不会增加学习进度
        state["attention"] += 0.1  # 恢复专注度
        state["fatigue"] -= 0.15  # 减少疲劳度
        state["attention"] = min(state["attention"], 1)  # 专注度最大为1
        state["fatigue"] = max(state["fatigue"], 0)  # 疲劳度最小为0
        print("休息中...")
        show_progress_bar(state)
        return  # 休息时直接返回，不增加学习进度

    # 疲劳范围控制
    state["fatigue"] = max(0, min(state["fatigue"], 1))

    # 实际学习增量
    real_gain = base_gain * efficiency
    state["level"] += real_gain

    print(f"学习效率系数：{round(efficiency, 2)}")
    show_progress_bar(state)

# ======================
# 模拟多个学生的学习状态
# ======================
students = [
    {"name": "学生A", "age": 17, "subject": "物理", "module": "力学", "topic": "牛顿第二定律", "level": 2.6, "attention": 0.8, "fatigue": 0.2},
    {"name": "学生B", "age": 16, "subject": "物理", "module": "力学", "topic": "牛顿第二定律", "level": 3.2, "attention": 0.9, "fatigue": 0.1},
    {"name": "学生C", "age": 18, "subject": "物理", "module": "力学", "topic": "牛顿第二定律", "level": 2.0, "attention": 0.6, "fatigue": 0.3}
]

# ======================
# 为每个学生进行一次完整学习过程
# ======================
for student in students:
    print(f"\n开始 {student['name']} 的学习过程：")
    for action in ["讲解", "例题", "反思", "休息"]:
        print(f"\n教学行为：{action}")
        apply_teaching_action(student, action)
        apply_camera_signal(student)  # 应用摄像头信号调整学生状态
        show_learning_position(student)
        compare_with_ideal(student, ideal_state)

# ======================
# 保存学习进度到文件（示例）
# ======================
with open("students_learning_progress.json", "w") as file:
    json.dump(students, file, indent=4)

print("\n学习数据已保存到 'students_learning_progress.json'")
