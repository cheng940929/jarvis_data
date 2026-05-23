import re
from typing import List, Dict, Union


def parse_dialog_history(text: str) -> List[Dict[str, Union[str, List[str]]]]:
    """
    解析对话文本，按标签顺序输出 list of dicts。
    支持标签：
      - User
      - Assistant Thought
      - Assistant Tool Call  -> 在输出中使用键 "Assistant Tool" (值为 list)
      - Tool Output          -> 在输出中使用键 "Tool Output"    (值为 list)
      - Assistant Final Response

    规则：
      - 保持内容不变（仅去除块首尾多余空行），工具调用/输出按空行分块或按行分割成多个条目。
      - 返回顺序与文本中标签出现顺序一致。
    """
    labels = [
        "User",
        "Assistant Thought",
        "Assistant Tool Call",
        "Tool Output",
        "Assistant Final Response",
    ]
    # 用来匹配标签（行首）
    pattern = re.compile(r'(?m)^(User|Assistant Thought|Assistant Tool Call|Tool Output|Assistant Final Response)\s*:\s*')

    matches = list(pattern.finditer(text))
    if not matches:
        return []

    def split_top_level_commas(s: str) -> List[str]:
        """
        在顶层（不在 () 或 {} 内）按逗号切分字符串
        """
        parts = []
        buf = []
        depth_paren = 0
        depth_brace = 0

        for ch in s:
            if ch == '(':
                depth_paren += 1
            elif ch == ')':
                depth_paren -= 1
            elif ch == '{':
                depth_brace += 1
            elif ch == '}':
                depth_brace -= 1

            if ch == ',' and depth_paren == 0 and depth_brace == 0:
                parts.append(''.join(buf).strip())
                buf = []
            else:
                buf.append(ch)

        if buf:
            parts.append(''.join(buf).strip())

        return [p for p in parts if p]
    def split_into_items(block: str) -> List[str]:
        blk = block.strip('\n')
        if blk.strip() == "":
            return []

        # 1️⃣ 优先：按空行切
        if re.search(r'\n\s*\n', blk):
            return [p.rstrip() for p in re.split(r'\n\s*\n', blk) if p.strip()]

        # 2️⃣ 单行但包含多个工具 / 输出（顶层逗号）
        if '\n' not in blk and ',' in blk:
            parts = split_top_level_commas(blk)
            if len(parts) > 1:
                return parts

        # 3️⃣ 普通多行：按行
        if '\n' in blk:
            return [line.rstrip() for line in blk.splitlines() if line.strip()]

        return [blk]

    out = []
    for i, m in enumerate(matches):
        label = m.group(1)
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        content = text[start:end]
        # 去除两端空白行，但保留内部换行和缩进
        content = content.strip('\n')
        if label == "Assistant Tool Call":
            items = split_into_items(content)
            out.append({"Assistant Tool": items})
        elif label == "Tool Output":
            items = split_into_items(content)
            out.append({"Tool Output": items})
        else:
            # 其他标签，保持为单个字符串（去掉首尾空行）
            out.append({label: content.strip()})
    return out


import json
from copy import deepcopy

def split_to_sharegpt(raw_str: str):
    """
    将包含 system + 多轮对话 + 多工具调用的数据
    拆分为多条 ShareGPT 格式数据
    """
    data = json.loads(raw_str)

    system_info = json.dumps(data[0], ensure_ascii=False)
    rounds = data[1:]

    results = []

    history_final_only = []

    for round_msgs in rounds:
        user_msg = None
        thought_buffer = ""
        current_convs = []

        for msg in round_msgs:
            # ---------- User ----------
            if "User" in msg:
                user_msg = msg["User"]

                # 新一轮开始：历史只保留 final
                current_convs = deepcopy(history_final_only)
                current_convs.append({
                    "from": "human",
                    "value": user_msg
                })

            # ---------- Assistant Thought ----------
            elif "Assistant Thought" in msg:
                thought_buffer = msg["Assistant Thought"]

            # ---------- Assistant Tool ----------
            elif "Assistant Tool" in msg:
                for tool_call in msg["Assistant Tool"]:
                    value = ""
                    if thought_buffer:
                        value += f"<think>\n{thought_buffer}\n</think>\n\n"
                        thought_buffer = ""

                    value += self_normalize_tool_call(tool_call)

                    current_convs.append({
                        "from": "function_call",
                        "value": value
                    })

                    results.append({
                        "conversations": deepcopy(current_convs),
                        "system": system_info
                    })

            # ---------- Tool Output ----------
            elif "Tool Output" in msg:
                for out in msg["Tool Output"]:
                    current_convs.append({
                        "from": "observation",
                        "value": json.dumps({"Tool Output": [out]}, ensure_ascii=False)
                    })

            # ---------- Assistant Final ----------
            elif "Assistant Final Response" in msg:
                final_clean = msg["Assistant Final Response"]

                value_with_thought = ""
                if thought_buffer:
                    value_with_thought += f"<think>\n{thought_buffer}\n</think>\n\n"
                    thought_buffer = ""

                value_with_thought += final_clean

                # 当前样本（带 thought）
                current_convs.append({
                    "from": "gpt",
                    "value": value_with_thought
                })

                results.append({
                    "conversations": deepcopy(current_convs),
                    "system": system_info
                })

                # ✅ history：只保留 clean final
                history_final_only.append({
                    "from": "human",
                    "value": user_msg
                })
                history_final_only.append({
                    "from": "gpt",
                    "value": final_clean
                })

    return results


def self_normalize_tool_call(tool_call_str: str):
    """
    把类似：
    query_memory("a":1,"b":2)
    变成标准 JSON 字符串
    """
    name, args = tool_call_str.split("(", 1)
    args = args.rsplit(")", 1)[0]

    return json.dumps({
        "name": name.strip(),
        "arguments": json.loads("{" + args + "}")
    }, ensure_ascii=False)



