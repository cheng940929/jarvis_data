import json
from chat_api import R1,QWEN_MAX
import random
import threading
from utiles import parse_dialog_history
from user_profile_config import config
from user_seed_info import user_seed_info,generate_hobby_profile,weights
ds_r1 = QWEN_MAX()
conflict = ['无冲突：用户请求与状态表无关',
            '软冲突（偏好）：用户请求违背了设定的偏好',
            '硬冲突（生理/安全）：用户请求严重违背生理状态',
            '互补增强：用户状态能辅助回答']
conflict_weight = [0.4,0.15,0.15,0.3]
tool_use = ['无需调用工具：纯闲聊或基于状态表回复。',
            '单工具：只调用一个工具就能回复',
            '多工具并行：需要同时调用工具，且工具间互不影响',
            '多工具依赖：需要按顺序调用多个工具，后面工具调用的输入需要前面调用工具的结果',
            '工具调用结果不全:成功调用了结果，但是结果的信息不足或不完善，导致没办法回答用户的问题',
            '工具无效/缺失：正常调用了工具，但工具因为某种原因没有返回结果，Tool Output那里直接填ERROR']
tool_use_weight = [0.25,0.25,15,0.15,0.1,0.1]
tool_use_weight_loss_info = [0.30,0.35,0,15,0.1,0.1]
intent = ['事实查询（百科、新闻）。',
          '生活决策（穿衣、饮食、运动）。',
          '情感宣泄（抱怨、开心、焦虑）。',
          '记忆回溯（“我上次...”）。',
          '跟jarvis分享趣闻']
intent_weight = [0.25,0.25,0.25,0.25,0.25]
def make_user_message(message):
    return {"role": "user", "content": message}
def make_ai_message(message):
    return {"role": "assistant", "content": message}
def make_prompt(result):
    conflict_tmp = random.choices(conflict, conflict_weight, k=1)[0]
    tool_use_tmp = random.choices(tool_use, tool_use_weight, k=1)[0]
    intent_tmp = random.choices(intent, intent_weight, k=1)[0]
    print('####################')
    print(conflict_tmp+'\n'+tool_use_tmp+'\n'+intent_tmp)
    print('####################')

    prompt = f'''请生成一段用户和它的人工智能助手jarvis的对话:
        背景:jarvis是一个会察言观色，能看到用户状态表(一个描述用户当前状态的信息表）并且能调用三个工具的AI人工助手(三个工具都是信息查询类的工具），类似于钢铁侠的jarvis，不过更偏向帮助用户的生活学习，没有除了查询信息，保存记忆(jarvis的基础功能，无需调用任何工具就能保存) ，和同用户交流之外的其他功能，用户的第一人称视频总结都存在jarvis的记忆中，jarvis可以通过工具进行查看，jarvis还可以调用查询天气的工具和互联网搜索的工具。
        用户会对jarvis说各种各样的话，jarvis会基于用户说的话，基于当前用户的状态表，判断是否要调用工具，最终给出一个很智能的回答，jarvis所有的回答都是基于已知信息的，不会出现幻觉而编造或推理出任何没有出现的信息

        具体怎么做?你需要基于用户状态表和下方的要求生成一个如下结构的数据
        User: user query
        Assistant Thought: (专注于思考如何回复user query)
        Assistant Tool Call (optional，如果不需要调用工具，这一行直接省略): ...
        Tool Output (optional，如果不需要调用工具，这一行直接省略): ...
        Assistant Final Response: ...

        生成的User Query的要求，必须全部满足:
        1：user query 和用户状态表的冲突程度:{conflict_tmp}
        2:工具调用复杂度:{tool_use_tmp}
        3:用户意图类型:{intent_tmp}
        4:用户默认jarvis知道他的当前状态，所以说的话不会包含自己的状态信息
        
        Assistant(jarvis)回复的要求:
        1:实事求是，jarvis的已知信息只有用户状态表，jarvis和用户的聊天记录，和jarvis调用工具的结果。禁止出现幻觉说出上面信息源之外的有关用户的信息

        Available Tools：[query_memory, query_weather, query_internet]
        query_memory:query_memory("detail_level":"有三种不同的数据源可以选，1:原子级(每30s记录用户正在做的事情) 2:事件级(记录用户做的单个完整事件) 3:每日总结(记录用户每一天的行动总结)","查询关键词":"要查询的关键信息"，"time_range":"要查询数据的时间范围:例如yyyy-mm-dd hh:mm:ss-yyyy-mm-dd hh:mm:ss，也可以为空")
        query_weather:query_weather("city": "要查询天气的城市名，比如深圳")
        query_internet:query_internet("query":"要搜索的内容")
        

        用户状态表:{result}
        '''

    return prompt


def make_missed_prompt(result):
    conflict_tmp = random.choices(conflict, conflict_weight, k=1)[0]
    tool_use_tmp = random.choices(tool_use, tool_use_weight_loss_info, k=1)[0]
    intent_tmp = random.choices(intent, intent_weight, k=1)[0]
    print('####################')
    print('无冲突'+ '\n' + tool_use_tmp + '\n' + intent_tmp)
    print('####################')

    prompt = f'''请生成一段用户和它的人工智能助手jarvis的对话:
        背景:jarvis是一个会察言观色，能看到用户状态表(一个描述用户当前状态的信息表）并且能调用三个工具的AI人工助手(三个工具都是信息查询类的工具），类似于钢铁侠的jarvis，不过更偏向帮助用户的生活学习，没有除了查询信息，保存记忆(jarvis的基础功能，无需调用任何工具就能保存)，和同用户交流之外的其他功能，用户的第一人称视频总结都存在jarvis的记忆中，jarvis可以通过工具进行查看，jarvis还可以调用查询天气的工具和互联网搜索的工具。
        用户会对jarvis说各种各样的话，jarvis会基于用户说的话，基于当前用户的状态表，判断是否要调用工具，最终给出一个很智能的回答，jarvis所有的回答都是基于已知信息的，不会出现幻觉而编造或推理出任何没有出现的信息

        具体怎么做?你需要基于用户状态表和下方的要求生成一个如下结构的数据
        User: user query
        Assistant Thought: (专注于思考如何回复user query)
        Assistant Tool Call (optional，如果不需要调用工具，这一行直接省略): ...
        Tool Output (optional，如果不需要调用工具，这一行直接省略): ...
        Assistant Final Response: ...

        生成的User Query的要求，必须全部满足:
        1:jarvis回复时要用到用户状态表中缺失的数据
        2:工具调用复杂度:{tool_use_tmp}
        3:用户意图类型:{intent_tmp}
        4:用户默认jarvis知道他的当前状态，所以说的话不会包含自己的状态信息

        Assistant(jarvis)回复的要求:
        1:实事求是，jarvis的已知信息只有用户状态表，jarvis和用户的聊天记录，和jarvis调用工具的结果。禁止出现幻觉说出上面信息源之外的有关用户的信息
        2:如果jarvis回复必须用到状态表中缺失的信息，那么jarvis会先询问用户这些必要的信息

        Available Tools：[query_memory, query_weather, query_internet]
        query_memory:query_memory("detail_level":"有三种不同的数据源可以选，1:原子级(每30s记录用户正在做的事情) 2:事件级(记录用户做的单个完整事件) 3:每日总结(记录用户每一天的行动总结)","查询关键词":"要查询的关键信息"，"time_range":"要查询数据的时间范围:例如yyyy-mm-dd hh:mm:ss-yyyy-mm-dd hh:mm:ss，也可以为空")
        query_weather:query_weather("city": "要查询天气的城市名，比如深圳")
        query_internet:query_internet("query":"要搜索的内容")

        用户状态表:{result}
        '''

    return prompt


def generate_sft_data():
    chat_dict = []
    constructed_data = []
    user_info = random.choices(user_seed_info,weights,k=1)[0]
    hobby = generate_hobby_profile()
    prompt_full = f"""{user_info},爱好是{hobby},请你发挥想象力，先想象出一个鲜活的人，再构建出人物信息表，表的结构如下所示:没必要所有的key都有对应的值，但是请不要额外增加key,\n{config}"""
    numbers = [i for i in range(10, 91, 10)]
    miss_percent = random.choice(numbers)
    prompt_miss = f"""{user_info},爱好是{hobby},请你发挥想象力，先想象出一个鲜活的人，再构建出人物信息表，表的结构如下所示,但是因为某种原因，这个任务信息表损坏了(当前时间的信息没哟损坏），其中{miss_percent}%的信息都缺失了,请返回{miss_percent}%缺失后的人物信息表\n{config}"""
    prompt_list = [prompt_full, prompt_miss]
    type = random.choices([0,1])[0]
    result,think = ds_r1.get_complete([make_user_message(prompt_list[type])])
    if type == 0:
        prompt = make_prompt(result)
    else:
        prompt = make_missed_prompt(result)
    result_, think_ = ds_r1.get_complete([make_user_message(prompt)])
    constructed_data.append(json.loads(result))
    constructed_data.append(parse_dialog_history(result_))
    chat_dict.append(make_user_message(prompt))
    chat_dict.append(make_ai_message(result_))
    for i in range(random.randint(5,10)):
        chat_dict.append(make_user_message('请猜测一下用户下一句会说什么，用第一人称视角，只输出用户会说的话'))
        result_, think_ = ds_r1.get_complete(chat_dict)
        chat_dict.pop(-1)
        chat_dict.append(make_user_message(result_))
        result_, think_ = ds_r1.get_complete(chat_dict)
        constructed_data.append(parse_dialog_history(result_))
        chat_dict.append(make_ai_message(result_))
    json_line = json.dumps(constructed_data, ensure_ascii=False)
    with open("jarvis_chat.jsonl", 'a', encoding='utf-8') as f:
        f.write(json_line + '\n')


for i in range(200):
    generate_sft_data()


