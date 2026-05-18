"""
知识图谱构建中"统计学生-知识点关系"的具体例子
"""
from collections import defaultdict

# ============ 原始数据 ============
# 学生做题记录
student_records = {
    "学生1": [
        {"exercise_id": "A", "is_correct": 1, "concept_ids": ["k1"]},
        {"exercise_id": "B", "is_correct": 0, "concept_ids": ["k1", "k2"]},
        {"exercise_id": "C", "is_correct": 1, "concept_ids": ["k2"]},
    ],
    "学生2": [
        {"exercise_id": "A", "is_correct": 1, "concept_ids": ["k1"]},
        {"exercise_id": "C", "is_correct": 1, "concept_ids": ["k2"]},
        {"exercise_id": "D", "is_correct": 1, "concept_ids": ["k3"]},
    ],
    "学生3": [
        {"exercise_id": "A", "is_correct": 0, "concept_ids": ["k1"]},
        {"exercise_id": "B", "is_correct": 1, "concept_ids": ["k1", "k2"]},
        {"exercise_id": "D", "is_correct": 0, "concept_ids": ["k3"]},
    ],
}

# ============ 统计过程 ============

# 1. 每个知识点的总尝试次数
concept_attempts = defaultdict(int)

# 2. 每个知识点的正确次数
concept_correct = defaultdict(int)

# 3. 尝试过某个知识点的学生集合
concept_attempt_students = defaultdict(set)

# 4. 答对某个知识点的学生集合
concept_success_students = defaultdict(set)

# 5. 每个学生尝试过的知识点集合
student_attempted_concepts = defaultdict(set)

# 6. 每个学生答对的知识点集合
student_correct_concepts = defaultdict(set)

# ========== 遍历所有记录进行统计 ==========
for student_id, records in student_records.items():
    for record in records:
        is_correct = record["is_correct"]
        for concept_id in record["concept_ids"]:
            # 统计尝试次数
            concept_attempts[concept_id] += 1
            concept_attempt_students[concept_id].add(student_id)
            student_attempted_concepts[student_id].add(concept_id)

            # 如果答对，统计正确次数
            if is_correct:
                concept_correct[concept_id] += 1
                concept_success_students[concept_id].add(student_id)
                student_correct_concepts[student_id].add(concept_id)

# 7. 计算答错某个知识点的学生集合
concept_failed_students = defaultdict(set)
for student_id, attempted in student_attempted_concepts.items():
    successes = student_correct_concepts.get(student_id, set())
    for concept_id in attempted - successes:  # 尝试过但没答对 = 答错
        concept_failed_students[concept_id].add(student_id)

# ============ 打印统计结果 ============
print("=" * 60)
print("知识点维度统计")
print("=" * 60)
print(f"{'知识点':<8} {'尝试次数':<8} {'正确次数':<8} {'正确率':<8} {'答对的学生':<15} {'答错的学生':<15}")
print("-" * 60)
for concept_id in sorted(concept_attempts.keys()):
    attempts = concept_attempts[concept_id]
    correct = concept_correct[concept_id]
    accuracy = correct / attempts if attempts > 0 else 0
    success_students = concept_success_students[concept_id]
    failed_students = concept_failed_students[concept_id]
    print(f"{concept_id:<8} {attempts:<8} {correct:<8} {accuracy:<8.2f} {str(success_students):<15} {str(failed_students):<15}")

print("\n" + "=" * 60)
print("学生维度统计")
print("=" * 60)
print(f"{'学生':<8} {'尝试的知识点':<20} {'答对的知识点':<20}")
print("-" * 60)
for student_id in sorted(student_attempted_concepts.keys()):
    attempted = student_attempted_concepts[student_id]
    correct = student_correct_concepts[student_id]
    print(f"{student_id:<8} {str(attempted):<20} {str(correct):<20}")

print("\n" + "=" * 60)
print("这些统计数据后续用途")
print("=" * 60)
print("""
1. concept_attempts / concept_correct → 计算每个知识点的正确率

2. concept_success_students → 用于判断先修关系
   例：答对k2的学生 {学生1, 学生2, 学生3}，其中 {学生1, 学生2} 也答对了k1
   → prereq_given_dependent = 2/3 = 0.67

3. concept_failed_students → 用于判断先修关系
   例：答错k1的学生 {学生3}，答错k2的学生 {}
   → 用于计算 backward_fail_ratio

4. student_correct_concepts → 统计知识点共现
   例：学生1答对了 {k1, k2}，学生2答对了 {k1, k2, k3}
   → joint_success_counts[(k1, k2)] = 2 (有2个学生同时答对k1和k2)
""")
