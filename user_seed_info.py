import random

# 定义爱好列表 (基于上一轮的数据)
tier_1_hobbies = [
    "电子游戏 (手游/PC/主机)",
    "短视频与流媒体刷剧",
    "搞钱与副业研究"
]

tier_2_hobbies = [
    "养宠 (吸猫撸狗)",
    "身体管理与新式运动 (健身/帕梅拉)",
    "探店与 City Walk (城市漫步)"
]

tier_3_hobbies = [
    "户外露营与山系生活",
    "特调咖啡与微醺文化",
    "潮流运动 (飞盘/陆冲/滑雪)"
]

tier_4_hobbies = [
    "盲盒与潮玩收藏 (谷圈)",
    "泛二次元与ACG文化 (Cosplay/三坑)",
    "客制化与极客装备 (键盘/摄影/HiFi)",
    "玄学与身心灵 (塔罗/水晶/冥想)"
]

def generate_hobby_profile():
    """
    创建一个兴趣表，根据特定概率从不同Tier中随机抽取爱好。
    """
    my_hobbies = []

    # 1. Tier 1: 50% 概率
    if random.random() < 0.4:
        selected = random.choice(tier_1_hobbies)
        my_hobbies.append(selected)

    # 2. Tier 2: 70% 概率
    if random.random() < 0.6:
        selected = random.choice(tier_2_hobbies)
        my_hobbies.append(selected)

    # 3. Tier 3: 50% 概率
    if random.random() < 0.4:
        selected = random.choice(tier_3_hobbies)
        my_hobbies.append(selected)

    # 4. Tier 4: 20% 概率
    if random.random() < 0.2:
        selected = random.choice(tier_4_hobbies)
        my_hobbies.append(selected)

    # 5. 保底机制：如果列表为空，则从所有爱好中随机选一个
    if not my_hobbies:
        all_hobbies = tier_1_hobbies + tier_2_hobbies + tier_3_hobbies + tier_4_hobbies
        selected = random.choice(all_hobbies)
        my_hobbies.append(selected)

    return my_hobbies



user_seed_info = ['科技极客/开发者 (The Techie)',
                  '银发族/退休老人 (The Senior)',
                  '忙碌的带娃家长 (The Busy Parent)',
                  '户外探险/运动达人 (The Athlete/Explorer)',
                  'Z世代大学生/潮流青年 (The Gen Z)',
                  '商务精英/高管 (The Professional)']

weights = [0.25,0.05,0.1,0.2,0.3,0.1]