import pandas as pd
import numpy as np
import warnings
from newmodel_v26 import Display
from Genome import Genome
warnings.filterwarnings(action='ignore')
np.random.seed(777)
from load_data_v16 import load_data
import math
from datetime import datetime, date, time, timedelta

sheet = ["DEMAND", "BOM", "MC", "최대로딩시간", "DEVICE", "TAT", "ASSIGN", "Regular PM", "LineCapa", "Stock", "Frozen"]
data = pd.read_excel("생산계획_data.xlsx", sheet_name = sheet, engine = 'openpyxl')
num_gene = 15

env = Display(data=data)
GA_agent = Genome(num_gene = num_gene, population_size = 100, crossover_rate = 0.7, mutation_rate = 0.1, iteration = 100, stopping = 10, theta = 0.2, env = env)

max_fitenss, best_sol = GA_agent.run()

load = load_data(data)
demand = load.create_demand('성형2공장')
device = load.device_load('성형2공장')
using_device, using_name = load.u_device_load(demand, device)
action_dict = {index: device[index] for index in using_device}

make_action_name = {act_num: f'Device {device[device_name]} set up' for act_num, device_name in
                    enumerate(using_device)}
make_action_name[len(make_action_name)] = 'Device Not set up'

# 위 코드를 통해 action_name 지정
action_name = make_action_name

obs = env.reset()
cumul_reward = 0
count_abandon = 0
count_success = 0
count_excess = 0
count_set_up = 0
count_failed = 0
count_MC = 0
production_ready = pd.DataFrame(columns=['require_mol', 'mol_name', 'amount_setup', 'device', 'time',
                                         'Action'])  # 학습 결과를 출력하기 위해 state와 action 들어갈 수 있는 틀 만들어놓기

# STAY 제외 시
for i in range(2000):
    print("state: ", [round(x, 1) for x in np.array(env.state, dtype=float)])
    if obs[2] == 0:
        action = best_sol[i % num_gene]
    else:
        action = 3

    obs, rewards, dones, info = env.step(action)

    if info.get('abandon_amount'):
        count_abandon += info.get('abandon_amount')
    if info.get('success'):
        count_success += 1
    if info.get('excess'):
        count_excess += 1
    if info.get('set_up'):
        count_set_up += 1
    if info.get('failed_use'):
        count_failed += 1
    if info.get('MC'):
        count_MC += 1
    cumul_reward += rewards

    if dones:
        break

    total_time = info['usetime']
    setup_times = sum(info['setup_times'])
    mc_times = sum(info['mc_times'])
    excess_amount = sum(info['excess_amount'])
    load_amount = sum(info['load'])
    print("action: ", action_name[action])
    print("reward this step: ", "{:.1f}".format(float(rewards)))
    print("total reward: ", "{:.1f}".format(float(cumul_reward)))
    print("=" * 50)

    # 위에서 만들어 놓은 틀에 실제 state랑 action 넣기
    production_ready = production_ready.append({
        'require_mol': [round(x, 1) for x in np.array([env.state[0]], dtype=float)],
        'mol_name': [round(x, 1) for x in np.array([env.state[1]], dtype=float)],
        'amount_setup': [round(x, 1) for x in np.array([env.state[2]], dtype=float)],
        'device': [round(x, 1) for x in np.array([env.state[3]], dtype=float)],
        'time': [round(x, 1) for x in np.array([env.state[4]], dtype=float)],
        'Action': action_name[action],
    }, ignore_index=True)

# # STAY 포함
# for i in range(1000):
#     print("state: ", [round(x, 1) for x in np.array(env.state, dtype=float)])
#     action = best_sol[i % num_gene]
#     obs, rewards, dones, info = env.step(action)
#
#     cumul_reward += rewards
#     if info.get('abandon_amount'):
#         count_abandon += info.get('abandon_amount')
#     if info.get('success'):
#         count_success += 1
#     if info.get('excess'):
#         count_excess += 1
#     if info.get('set_up'):
#         count_set_up += 1
#     if info.get('failed_use'):
#         count_failed += 1
#
#     if dones:
#         break

    # total_time = info['usetime']
    # print("action: ", action_name[action])
    # print("reward this step: ", "{:.1f}".format(float(rewards)))
    # print("total reward: ", "{:.1f}".format(float(cumul_reward)))
    # print("=" * 50)
    # #
    # # 위에서 만들어 놓은 틀에 실제 state랑 action 넣기
    # production_ready = production_ready.append({
    #     'require_mol': [round(x, 1) for x in np.array([env.state[0]], dtype=float)],
    #     'mol_name': [round(x, 1) for x in np.array([env.state[1]], dtype=float)],
    #     'amount_setup': [round(x, 1) for x in np.array([env.state[2]], dtype=float)],
    #     'device': [round(x, 1) for x in np.array([env.state[3]], dtype=float)],
    #     'time': [round(x, 1) for x in np.array([env.state[4]], dtype=float)],
    #     'Action': action_name[action],
    # }, ignore_index=True)

print("Number of successful processing: ", count_success)
print("Total abandon amount", count_abandon)
print("Total excess count", count_excess)
print("Total MC count", count_MC)
print("Number of set up", count_set_up)
print('Less than 80% usage', count_failed)
print('Total time required: ', "{:.1f}".format(float(total_time / 24)), 'days')
# mc 합, 셋업 합
print('Sum of mc', "{:.1f}".format(float(mc_times / 24)), 'days')
print('Sum of set up', "{:.1f}".format(float(setup_times / 24)), 'days')
print('Sum of excess amount', float(excess_amount))
print('Sum of load amount', float(load_amount))
print('utilization', ((total_time - mc_times - setup_times - count_abandon) / total_time) * 100)
print('Total Reward: ', cumul_reward)

print(best_sol)
# GA 학습 곡선
GA_agent.curve()