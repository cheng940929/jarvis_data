"""
Jarvis AI 眼镜助手 - 三层记忆数据生成器（三步流水线）

模拟真实系统的三层记忆架构：
  Step 0: 为每个用户生成 N 天事件日历（生活剧本作息 + 近期事件线 + 个性化随机日常事件）
  Step 1: 原子层 — 基于日历生成事件驱动的原子记录 {start, end, location, action}，模拟语义合并
  Step 2: 事件层 — 按每 2-3 小时分块，将原子层总结为事件块
  Step 3: 每日总结 — 从事件层提炼情绪、偏好、未完成目标（进入 System Prompt 的数据）

输入: life_scripts.json（生活剧本） + personas.py（用户画像）
输出: memory_data.json（三层记忆数据）

用法: python generate_memory.py
"""

import json
import os
import sys
import re
import random
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from chat_api import DS_v4
from personas import personas
from life_scripts import life_scripts

# Step 0/1 用强模型，Step 2 事件层调用量最大，可用同模型或切换更便宜的
llm = DS_v4()
# 如果想用不同模型做事件层，取消注释下面这行：
# llm_cheap = QWEN_MAX()

MEMORY_DAYS_MIN = 14  # 每个用户最少生成天数
MEMORY_DAYS_MAX = 30  # 每个用户最多生成天数

WEEKDAYS_CN = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]

# 事件层分块大小（小时）
CHUNK_HOURS_MIN = 2
CHUNK_HOURS_MAX = 3


# ============================================================
# Step 0: 生成事件日历（保留原有逻辑）
# ============================================================

def build_calendar_prompt(script, persona, day_count):
    prompt = f"""你是一个生活模拟专家。你的任务是为下面这个人，生成一份 {day_count} 天的事件日历。

事件日历的作用是：指导另一个大模型生成这个人每天的三层记忆数据（模拟AI眼镜记录的第一人称视角）。

## 这个人的生活剧本

{json.dumps(script, ensure_ascii=False, indent=2)}

## 这个人的画像信息

{json.dumps(persona, ensure_ascii=False, indent=2)}

## 生成规则

1. {day_count} 天的日期从 {day_count} 天前开始，到今天结束
2. **工作日**按生活剧本中的"工作日作息"走，**休息日**按"休息日作息"走（注意这个人的休息日是哪天）
3. "近期事件线"中的事件必须落在对应的日期上（根据时间描述推算）
4. 每天注入 1-3 个随机小事件——这些事件必须**贴合这个人的真实生活**，不能和画像/剧本矛盾
5. 每天的情绪要在合理范围内波动（不能每天都开心，也不能每天都低落）
6. 事件之间要有连续性（比如周一买了快递，周三可能到了；周二的面试结果可能周五才收到通知）

## 随机小事件的设计原则（非常重要）

你必须根据这个人的画像和生活剧本，为每一天设计贴合他/她真实生活的随机小事件。
以下是几类事件的参考方向，请根据这个人实际情况选择，也可以自由发挥：

- **日常琐碎**：吃饭踩雷/吃到好吃的、忘带东西、快递到了、手机没电等
- **身体感受**：睡不好、某个旧伤/慢性病不舒服、运动后状态好等（注意这个人实际有什么健康问题）
- **社交互动**：和这个人社交圈里的人发生的真实互动（参考生活剧本的"社交模式"）
- **职业相关**：和工作/学习相关的小事（参考这个人的职业特点）
- **家庭/伴侣**：和这个人的家人/伴侣/孩子的互动（如果有的话）
- **宠物**：和这个人的宠物的互动（如果有的话，注意宠物的名字和种类）
- **爱好相关**：和这个人爱好相关的小事（参考这个人的爱好列表）
- **情绪波动**：莫名的开心/焦虑/怀旧等（注意这个人的性格和近期事件线）
- **天气/环境**：天气变化对这个人出行的影响（参考这个人的城市和通勤方式）

**绝对禁止**：生成和这个人画像矛盾的事件（比如没养宠物的人出现宠物事件、有慢性病的人突然很健康、没车的人出现开车相关事件等）

## 输出格式

严格输出 JSON 数组，每个元素是一天：
[
    {{
        "day": 1,
        "date": "2024-05-27",
        "weekday": "星期一",
        "is_rest_day": false,
        "mood": "今天的心情（一句话）",
        "planned_events": ["生活剧本中这天会做的事"],
        "special_events": ["今天发生的特殊事件（1-3个），来自近期事件线或你设计的贴合这个人的随机小事件"],
        "weather": "今天的天气（简单描述）"
    }},
    ...
]

只输出 JSON，不要输出其他内容。共 {day_count} 天。"""

    return prompt


def generate_calendar(script, persona):
    day_count = random.randint(MEMORY_DAYS_MIN, MEMORY_DAYS_MAX)
    print(f"  Step 0: 生成 {day_count} 天事件日历...")
    prompt = build_calendar_prompt(script, persona, day_count)
    result, think = llm.get_complete([{"role": "user", "content": prompt}])

    # 解析 JSON
    try:
        calendar = json.loads(result)
    except json.JSONDecodeError:
        json_match = re.search(r'\[[\s\S]*\]', result)
        if json_match:
            calendar = json.loads(json_match.group())
        else:
            raise ValueError(f"无法解析事件日历 JSON: {result[:200]}")

    if not isinstance(calendar, list) or len(calendar) < MEMORY_DAYS_MIN:
        raise ValueError(f"事件日历天数不足: {len(calendar) if isinstance(calendar, list) else 'not a list'}")

    print(f"  ✓ 事件日历生成成功，共 {len(calendar)} 天")
    return calendar


# ============================================================
# Step 1: 生成原子层（事件驱动格式，模拟语义合并）
# ============================================================

def build_atomic_prompt(script, persona, day_info):
    prompt = f"""你是AI眼镜的VLM（视觉语言模型）模拟器。你需要模拟摄像头捕捉到的每一帧画面变化，将其转化为纯视觉描述。

原子层模拟的是：AI眼镜的摄像头 → 边缘截流（画面不变就不记录）→ VLM把画面转成文字 → 语义合并（连续做同一件事只记一条）。

## 用户画像
{json.dumps(persona, ensure_ascii=False, indent=2)}

## 这天的情况
日期: {day_info['date']} {day_info['weekday']}
{'休息日' if day_info.get('is_rest_day') else '工作日'}
心情: {day_info['mood']}
天气: {day_info['weather']}
计划中的事: {json.dumps(day_info['planned_events'], ensure_ascii=False)}
特殊事件: {json.dumps(day_info['special_events'], ensure_ascii=False)}

## 核心规则（非常重要，务必遵守）

### 1. 纯视觉，只有画面信息
你只能描述摄像头能"看到"的内容。**绝对不能出现：**
- 听觉信息："听到闹钟响"、"听到猫叫"、"听到播客"、"听到雷声"
- 内心活动："在想面试的事"、"心里焦虑"、"想起昨天的事"
- 手机/电脑屏幕内容："刷到一条新闻"、"看了B站视频"、"收到消息"

**只能写：**
- 人在做什么动作（可见的身体动作）
- 场景中有什么变化（物体移动、光线变化、其他人/动物的动作）
- 人去了哪里、拿了什么东西

### 2. 每条只描述一件事
一条记录 = 一次画面变化 = 一个动作/场景。**绝对不能把多件事合并到一条里。**

错误示例（合并了多件事）：
- "起床给猫加猫粮换水，蹲厕所刷手机"
- "走路去公司，戴耳机"
- "写代码和Review代码和参加站会"

正确示例（每条一件事）：
- "坐起身，拿手机看屏幕"
- "走进厨房，从柜子里拿出猫粮袋"
- "往猫碗里倒猫粮"
- "换猫碗里的水"
- "走进卫生间，关门"
- "从工位站起来，走向会议室"

### 3. 语义合并的规则
只有**完全相同的连续动作**才合并为一条。例如：
- 一个人坐在工位面对电脑屏幕，从10:00到12:00画面基本不变 → 合并为一条 {{start:"10:00", end:"12:00"}}
- 一个人走路去公司，画面持续变化但都是走路 → 合并为一条 {{start:"09:25", end:"09:40"}}
- 但"走路去公司"和"到公司接咖啡"是两件事 → 分成两条

### 4. 描述要简短客观
每条 action 控制在 15 个字以内，纯客观动作描述，不加任何修饰和判断。

## 输出格式

严格输出 JSON 数组：
[
    {{"start": "09:00", "end": "09:03", "location": "卧室", "action": "翻身拿手机按掉闹钟"}},
    {{"start": "09:03", "end": "09:08", "location": "卧室", "action": "侧躺看手机屏幕"}},
    {{"start": "09:08", "end": "09:10", "location": "卧室", "action": "猫跳上床走到枕头边"}},
    {{"start": "09:10", "end": "09:12", "location": "卧室", "action": "坐起来摸猫的头"}},
    {{"start": "09:12", "end": "09:14", "location": "卧室", "action": "下床穿拖鞋走向客厅"}},
    {{"start": "09:14", "end": "09:15", "location": "厨房", "action": "打开橱柜拿出猫粮袋"}},
    {{"start": "09:15", "end": "09:16", "location": "厨房", "action": "往猫碗里倒猫粮"}},
    {{"start": "09:16", "end": "09:17", "location": "厨房", "action": "端起猫水碗走到水槽换水"}},
    {{"start": "09:17", "end": "09:20", "location": "卫生间", "action": "走进卫生间关门坐到马桶上"}},
    {{"start": "09:20", "end": "09:30", "location": "卫生间", "action": "低头看手机屏幕"}},
    {{"start": "09:30", "end": "09:32", "location": "卫生间", "action": "站起身冲水洗手"}},
    ...
    {{"start": "10:00", "end": "12:00", "location": "公司工位", "action": "坐在工位面对电脑屏幕"}},
    {{"start": "12:00", "end": "12:02", "location": "公司工位", "action": "拿出手机点外卖"}},
    ...
]

只输出 JSON 数组，不要输出其他内容。覆盖从起床到睡觉的完整一天。一天大约 100-200 条。"""

    return prompt


def generate_atomic_layer(script, persona, day_info):
    prompt = build_atomic_prompt(script, persona, day_info)
    result, think = llm.get_complete([{"role": "user", "content": prompt}])

    try:
        atoms = json.loads(result)
    except json.JSONDecodeError:
        json_match = re.search(r'\[[\s\S]*\]', result)
        if json_match:
            atoms = json.loads(json_match.group())
        else:
            raise ValueError(f"无法解析原子层 JSON: {result[:200]}")

    if not isinstance(atoms, list):
        raise ValueError("原子层数据不是数组")

    return atoms


# ============================================================
# Step 2: 生成事件层（按 2-3 小时分块总结）
# ============================================================

def split_atoms_into_chunks(atoms, chunk_hours):
    """将原子层记录按时间分块，每块约 2-3 小时。

    策略：以一天的活动时间（如 07:00-02:00）均匀分为若干块。
    找到原子层的最早 start 和最晚 end，然后按 chunk_hours 小时切分。
    """
    if not atoms:
        return []

    def time_to_minutes(t):
        """'HH:MM' -> 分钟数，处理跨天（如 00:30 = 24*60+30）"""
        h, m = t.split(":")
        val = int(h) * 60 + int(m)
        # 如果时间小于 5:00，认为是第二天
        if val < 5 * 60:
            val += 24 * 60
        return val

    def minutes_to_time(m):
        """分钟数 -> 'HH:MM'"""
        day_offset = 0
        if m >= 24 * 60:
            day_offset = 24 * 60
            m -= day_offset
        return f"{m // 60:02d}:{m % 60:02d}"

    # 找到时间范围
    all_starts = [time_to_minutes(a["start"]) for a in atoms]
    all_ends = [time_to_minutes(a["end"]) for a in atoms]
    earliest = min(all_starts)
    latest = max(all_ends)

    # 生成分块边界
    total_minutes = latest - earliest
    chunk_minutes = chunk_hours * 60
    boundaries = []
    t = earliest
    while t < latest:
        boundaries.append((t, min(t + chunk_minutes, latest)))
        t += chunk_minutes

    # 将原子记录分配到各块
    chunks = []
    for chunk_start, chunk_end in boundaries:
        chunk_atoms = []
        for a in atoms:
            a_start = time_to_minutes(a["start"])
            a_end = time_to_minutes(a["end"])
            # 原子记录与分块有交集就纳入
            if a_start < chunk_end and a_end > chunk_start:
                chunk_atoms.append(a)
        if chunk_atoms:
            chunks.append({
                "time_range": f"{minutes_to_time(chunk_start)}-{minutes_to_time(chunk_end)}",
                "atoms": chunk_atoms,
            })

    return chunks


def build_event_prompt(chunk_time_range, chunk_atoms_json, day_info):
    prompt = f"""你是AI眼镜的事件层记忆生成器。你需要把一段时间的原子层记录总结为事件块。

事件层是 RAG 查询的主力数据，需要把零碎的原子记录整合成有意义的事件描述。

## 这天的基本信息
日期: {day_info['date']} {day_info['weekday']}
心情: {day_info['mood']}

## 这个时间段的原子层记录（{chunk_time_range}）

{chunk_atoms_json}

## 生成规则

1. 输入是上面这个时间段的原子层记录
2. 输出 2-5 个事件块，每个包含 time_range 和 event
3. event 描述可以包含简单的情绪和因果逻辑（如"因为...所以..."、"有点累但..."）
4. 合并相似行为，突出有意义的事件和变化
5. 语言自然流畅，像在跟自己回忆今天发生的事

## 输出格式

严格输出 JSON 数组：
[
    {{
        "time_range": "09:00-09:40",
        "event": "起床收拾出门。猫跳上床要吃的，给它加了猫粮换了水。蹲厕所刷了一会儿技术新闻。出门走路去公司。"
    }},
    ...
]

只输出 JSON 数组，不要输出其他内容。"""

    return prompt


def generate_event_layer(atoms, day_info):
    """将一天的原子层按 2-3 小时分块，每块调用一次 LLM 生成事件记录。"""
    chunk_hours = random.randint(CHUNK_HOURS_MIN, CHUNK_HOURS_MAX)
    chunks = split_atoms_into_chunks(atoms, chunk_hours)

    all_events = []
    for chunk in chunks:
        atoms_json = json.dumps(chunk["atoms"], ensure_ascii=False, indent=2)
        prompt = build_event_prompt(chunk["time_range"], atoms_json, day_info)
        result, think = llm.get_complete([{"role": "user", "content": prompt}])

        try:
            events = json.loads(result)
        except json.JSONDecodeError:
            json_match = re.search(r'\[[\s\S]*\]', result)
            if json_match:
                events = json.loads(json_match.group())
            else:
                print(f"      警告: 分块 {chunk['time_range']} 事件层解析失败，跳过")
                continue

        if isinstance(events, list):
            all_events.extend(events)

    return all_events


# ============================================================
# Step 3: 生成每日总结（从事件层提炼）
# ============================================================

def build_summary_prompt(day_info, events_json):
    prompt = f"""你是AI眼镜的每日总结生成器。你需要从今天的事件层记录中提炼出每日总结。

每日总结是唯一会进入 System Prompt 的数据，是 AI 助手了解用户今天状态的核心信息。

## 这天的基本信息
日期: {day_info['date']} {day_info['weekday']}
{'休息日' if day_info.get('is_rest_day') else '工作日'}
心情: {day_info['mood']}

## 今天的事件层记录

{events_json}

## 生成规则

1. **聚焦三个方面**：情绪状态、偏好/习惯变化、未完成的目标或待办
2. 2-3 句话，高度压缩
3. 突出对 AI 助手最有用的信息（比如：用户今天心情不好因为什么、用户在等什么结果、用户明天有什么计划）
4. 不要复述今天做了什么，而是提炼出"意义"和"状态"
5. 语言简洁直接，不要抒情

## 输出格式

直接输出一段文字（2-3句话），不要输出 JSON，不要加引号。"""

    return prompt


def generate_daily_summary(day_info, events):
    events_json = json.dumps(events, ensure_ascii=False, indent=2)
    prompt = build_summary_prompt(day_info, events_json)
    result, think = llm.get_complete([{"role": "user", "content": prompt}])

    # 清理可能的引号包裹
    summary = result.strip()
    if summary.startswith('"') and summary.endswith('"'):
        summary = summary[1:-1]
    if summary.startswith("'") and summary.endswith("'"):
        summary = summary[1:-1]

    return summary


# ============================================================
# 辅助函数
# ============================================================

def parse_json_with_fallback(text):
    """尝试解析 JSON，支持多种格式容错。"""
    # 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试从 markdown 代码块中提取
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试匹配最外层 [] 或 {}
    for pattern in [r'\[[\s\S]*\]', r'\{[\s\S]*\}']:
        json_match = re.search(pattern, text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                continue

    return None


# ============================================================
# 主流程
# ============================================================

def main(target_pid=None, max_days=None):
    output_file = "memory_data.json"

    # 直接从 life_scripts.py 导入
    all_scripts = life_scripts
    if target_pid:
        all_scripts = {target_pid: life_scripts[target_pid]}
    print(f"已加载 {len(all_scripts)} 个生活剧本")

    # 加载已有的记忆数据（断点续传）
    existing = {}
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"已加载 {len(existing)} 个用户的记忆数据")

    # 构建 persona 查找表
    persona_map = {p["id"]: p for p in personas}

    # 逐个用户生成
    for pid, script in all_scripts.items():
        if pid in existing and not existing[pid].get("_error"):
            print(f"\n跳过 {pid}（已存在）")
            continue

        persona = persona_map.get(pid, {})
        print(f"\n{'='*60}")
        print(f"处理: {pid} - {persona.get('姓名', '未知')}, {persona.get('年龄', '?')}岁, {persona.get('城市', '?')}")
        print(f"{'='*60}")

        try:
            # Step 0: 生成事件日历
            calendar = generate_calendar(script, persona)

            # 调试模式：只跑前 N 天
            if max_days:
                calendar = calendar[:max_days]
                print(f"  [调试模式] 只跑前 {max_days} 天")

            # 逐天生成三层记忆
            user_memory = {
                "persona_id": pid,
                "days": [],
            }

            total_days = len(calendar)
            for idx, day_info in enumerate(calendar):
                day_num = day_info.get("day", 0)
                date_str = day_info.get("date", "?")
                print(f"\n  [{idx+1}/{total_days}] Day {day_num}: {date_str} - {day_info.get('special_events', [])}")

                try:
                    # Step 1: 生成原子层
                    print(f"    Step 1: 生成原子层...")
                    atoms = generate_atomic_layer(script, persona, day_info)
                    print(f"    ✓ 原子层: {len(atoms)} 条")

                    # Step 2: 生成事件层
                    print(f"    Step 2: 生成事件层...")
                    events = generate_event_layer(atoms, day_info)
                    print(f"    ✓ 事件层: {len(events)} 条")

                    # Step 3: 生成每日总结
                    print(f"    Step 3: 生成每日总结...")
                    summary = generate_daily_summary(day_info, events)
                    print(f"    ✓ 每日总结: {summary[:50]}...")

                    # 组装这一天
                    day_memory = {
                        "day": day_num,
                        "date": date_str,
                        "weekday": day_info.get("weekday", ""),
                        "mood": day_info.get("mood", ""),
                        "weather": day_info.get("weather", ""),
                        "原子层": atoms,
                        "事件层": events,
                        "每日总结": summary,
                    }
                    user_memory["days"].append(day_memory)

                except Exception as e:
                    print(f"    ✗ Day {day_num} 生成失败: {e}")
                    user_memory["days"].append({
                        "day": day_num,
                        "date": date_str,
                        "_error": str(e),
                    })

            existing[pid] = user_memory
            print(f"\n  ✓ {pid} 记忆数据生成完成，共 {len(user_memory['days'])} 天")

        except Exception as e:
            print(f"  ✗ {pid} 生成失败: {e}")
            existing[pid] = {"_error": str(e)}

        # 每个用户完成后保存
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        print(f"  已保存到 {output_file}")

    print(f"\n全部完成！共 {len(existing)} 个用户，保存在 {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", default=None, help="只生成指定用户，如 P01")
    parser.add_argument("--days", type=int, default=None, help="只跑前 N 天，用于调试")
    args = parser.parse_args()
    main(target_pid=args.pid, max_days=args.days)
