import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import random
import time


def getNormRvs(mu, sigma, step):
    """
    :param mu: int, 正态分布的平均值
    :param sigma: int, 正态分布的标准差
    :param step: int, 从这个正态分布中随机取多少个值
    :return: 返回取出来的值序列(四舍五入取整, 因为假设真实情况下老虎机给的Reward是整数)
    """
    rvs = stats.norm.rvs(loc=mu, scale=sigma, size=step)
    return [int(0.5 + data) for data in rvs]


def getQ(Reward, first=True):  # 也可以叫calculate_Q, 第二个参数first的含义会在3.2详细解释
    """
    :param Reward: 反馈序列
    :param first: 初始值是否参与计算, 默认True, 即参与
    :return: 该序列的Q值
    """
    if first:  # 带初始值计算
        return np.mean(Reward)  # 直接返回平均值即可
    else:  # 不带初始值计算
        if len(Reward) == 1:  # 目前Reward序列中只有初始值
            return Reward[0]  # 那就只能返回初始值
        else:  # 有很多值了
            return sum([Reward[i] for i in range(1, len(Reward))])//(len(Reward) - 1)  # 计算除去初始值的序列平均值


def pickHigherOne():
    """
    r取值在[1, 10]
    0.3 的概率不取Q值更高者
    :return:
        r <= 3:False
        3 <= r <= 10:True
    """
    r = random.randint(1, 10)
    if r <= 3:
        return False
    else:
        return True


def print_step_info(step, name, rewardValue, rewardSeq, curQ):
    """
    :param step: 当前第step步
    :param name: 老虎机名称
    :param rewardValue: 老虎机提供的反馈值
    :param rewardSeq: 之前的反馈序列
    :return: 无返回值
    """
    print('当前是第{0}次, 选择了老虎机 {1}, 它的Reward是{2}, 当前Q值为{3}'.format(step + 1, name, rewardValue, int(curQ + 0.5)))
    print('----------------------')
    rewardSeq.append(rewardValue)  # 顺便把reward存入序列
    time.sleep(1)
