from utils import *

if __name__ == '__main__':
    # 我们的目标是从有限的步骤中尽量最大化自己的收益
    A = getNormRvs(500, 50, 20)  # 老虎机 A 的概率分布 N(500, 50), 假设取20步
    B = getNormRvs(550, 100, 20)  # 老虎机 B 的概率分布 N(550, 100), 假设取20步

    Reward_A = [998]  # 存放从 A 获得的奖励, 假设有初值998(初值设置很大是为了避免只能选一个的尴尬情况, 可以选择参与Q值计算)
    Reward_B = [998]  # 存放从 B 获得的奖励, 假设有初值998(初值设置很大是为了避免只能选一个的尴尬情况, 可以选择参与Q值计算)

    # ξ-Greedy, 假设 ξ = 0.3
    for i in range(20):  # 在有限的步数之内，每一步都做个决策, 决策的策略是使用Q值更大的那个，但是保持一定的概率会选取另一个
        Q_A = getQ(Reward_A, False)  # Reward_A 目前的Q值
        Q_B = getQ(Reward_B, False)  # Reward_B 目前的Q值
        print('老虎机A的Q值:{0}   老虎机B的Q值:{1}'.format(Q_A, Q_B))
        if Q_A == 998:  # 998 那肯定得先用它
            print_step_info(i, 'A', A[i], Reward_A, Q_A)

        elif Q_B == 998:  # 998 那肯定得先用它
            print_step_info(i, 'B', B[i], Reward_B, Q_B)

        else:
            pick_higher = pickHigherOne()
            if Q_A >= Q_B:  # Q_A 其实是要选择 A 的, 但因为 ξ, 所以可能会选B
                if pick_higher:  # 我们要选择更高的那个
                    print_step_info(i, 'A', A[i], Reward_A, Q_A)
                else:
                    print_step_info(i, 'B', B[i], Reward_B, Q_B)
            else:
                if pick_higher:  # 我们要选择更高的那个
                    print_step_info(i, 'B', B[i], Reward_B, Q_B)
                else:
                    print_step_info(i, 'A', A[i], Reward_A, Q_A)
