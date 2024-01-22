import warnings
warnings.filterwarnings("ignore")
import copy
import gym
import pandas as pd
from gym import spaces
import numpy as np
from datetime import datetime, date, time, timedelta
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import time
from calendar import monthrange
from gym.utils import seeding  # random seed control 위해 import
# 납기 코드 수정(주문양만큼 비례)
# 버려지는 소자의 TERM 바꾸기 MAX(버려지는 소자 -20, 0)

import os
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization, DummyVecEnv
from typing import Union, List, Optional, Tuple

from stable_baselines3 import DQN, PPO
from stable_baselines3.common import callbacks
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from load_data_v16 import load_data
import math


def evaluate_policy_Display(
        model,
        env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        return_episode_rewards: bool = False,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    epi_rewards_discounted, epi_success_order, epi_abandon_amount, epi_excess_order, epi_setup, epi_times = [], [], [], [], [], []
    for i in range(n_eval_episodes):
        if not isinstance(env, VecEnv) or i == 0:
            obs = env.reset()
        done, state = False, None
        epi_reward_discounted = 0.0
        epi_success = 0
        epi_abandon = 0
        epi_excess = 0
        epi_set = 0
        epi_time = 0
        num_steps = 0
        while not done:
            action, state = model.predict(obs, state=state)
            obs, reward, done, _info = env.step(action)
            epi_reward_discounted += np.power(model.gamma, num_steps) * reward
            num_steps += 1

            if _info[0].get('abandon_amount'):
                epi_abandon += _info[0].get('abandon_amount')
            if _info[0].get('success'):
                epi_success += 1
            if _info[0].get('excess'):
                epi_excess += 1
            if _info[0].get('set_up'):
                epi_set += 1

            epi_time = float(_info[0]['usetime'] / 24)

            if render:
                env.render()
        epi_rewards_discounted.append(epi_reward_discounted)
        epi_success_order.append(epi_success)
        epi_abandon_amount.append(epi_abandon)
        epi_excess_order.append(epi_excess)
        epi_setup.append(epi_set)
        epi_times.append(epi_time)
    mean_discounted_reward = np.mean(epi_rewards_discounted)
    std_discounted_reward = np.std(epi_rewards_discounted)
    if return_episode_rewards:
        return epi_rewards_discounted, epi_success_order, epi_abandon_amount, epi_excess_order, epi_setup, epi_times
    return mean_discounted_reward, std_discounted_reward


class EvalCallback_Display(EvalCallback):
    def __init__(self, eval_env: Union[gym.Env, VecEnv],
                 callback_on_new_best: Optional[BaseCallback] = None,
                 n_eval_episodes: int = 5,
                 eval_freq: int = 10000,
                 log_path: str = None,
                 best_model_save_path: str = None,
                 deterministic: bool = True,
                 render: bool = False,
                 verbose: int = 1):
        super(EvalCallback_Display, self).__init__(eval_env,
                                                   callback_on_new_best,
                                                   n_eval_episodes,
                                                   eval_freq,
                                                   log_path,
                                                   best_model_save_path,
                                                   deterministic,
                                                   render,
                                                   verbose)
        self.results_discounted = []
        self.results_success_order = []
        self.results_abandon = []
        self.results_excess = []
        self.results_setup = []
        self.best_mean_abandon = 1000000

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            epi_rewards_discounted, epi_success_order, epi_abandon_amount, epi_excess_order, epi_setup, epi_times \
                = evaluate_policy_Display(self.model, self.eval_env,
                                          n_eval_episodes=self.n_eval_episodes,
                                          render=self.render,
                                          deterministic=self.deterministic,
                                          return_episode_rewards=True)

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.results_discounted.append(epi_rewards_discounted)
                self.results_success_order.append(epi_success_order)
                self.results_abandon.append(epi_abandon_amount)
                self.results_excess.append(epi_excess_order)
                self.results_setup.append(epi_setup)
                self.evaluations_length.append(epi_times)
                np.savez(self.log_path, timesteps=self.evaluations_timesteps,
                         results_discounted=self.results_discounted,
                         results_success_order=self.results_success_order,
                         results_abandon=self.results_abandon,
                         results_excess=self.results_excess,
                         results_setup=self.results_setup,
                         ep_lengths=self.evaluations_length)

            mean_reward_discounted, std_reward_discounted = np.mean(epi_rewards_discounted), np.std(
                epi_rewards_discounted)
            mean_success, std_success = np.mean(epi_success_order), np.std(epi_success_order)
            mean_abandon, std_abandon = np.mean(epi_abandon_amount), np.std(epi_abandon_amount)
            mean_excess, std_excess = np.mean(epi_excess_order), np.std(epi_excess_order)
            mean_setup, std_setup = np.mean(epi_setup), np.std(epi_setup)
            mean_ep_length, std_ep_length = np.mean(epi_times), np.std(epi_times)
            # Keep track of the last evaluation, useful for classes that derive from this callback
            self.last_mean_reward = mean_reward_discounted

            if self.verbose > 0:
                print("Eval num_timesteps={}, "
                      "episode_discounted_reward={:.2f} +/- {:.2f}".format(self.num_timesteps, mean_reward_discounted,
                                                                           std_reward_discounted),
                      "episode_success={:.2f} +/- {:.2f}".format(mean_success, std_success),
                      "episode_abandon={:.2f} +/- {:.2f}".format(mean_abandon, std_abandon),
                      "episode_excess={:.2f} +/- {:.2f}".format(mean_excess, std_excess),
                      "episode_setup={:.2f} +/- {:.2f}".format(mean_setup, std_setup))
                print("Episode day: {:.2f} +/- {:.2f}".format(mean_ep_length, std_ep_length))

            # if mean_success < 1.0e-4 and self.n_calls % (self.eval_freq*5):
            #     self.model.setup_model()

            # if mean_reward_discounted > self.best_mean_reward:
            if mean_abandon < self.best_mean_abandon:
                if mean_success > 108:
                    if self.verbose > 0:
                        # print("New best mean reward!")
                        print("New best mean abandon!")
                    if self.best_model_save_path is not None:
                        self.model.save(os.path.join(self.best_model_save_path, 'best_model'))
                    self.model.save(os.path.join(self.best_model_save_path, 'model' + str(self.num_timesteps)))
                    self.best_mean_abandon = mean_abandon
                    # self.best_mean_reward = mean_reward_discounted
                    # Trigger callback if needed
                    if self.callback is not None:
                        return self._on_event()

        return True


class Display(gym.Env):
    def __init__(self, data):

        # load_data 통해 필요한 변수들 준비해놓기
        self.load = load_data(data)

        # order 파일
        self.demand = self.load.create_demand('성형2공장')
        self.d_col = self.demand.columns.tolist()
        # 주문 총개수
        self.demand_num = self.demand[self.demand != 0].count().sum()

        # 소자 종류 정보
        self.general_device = self.load.device_load('성형2공장')

        # 해당 모델에서 사용할 소자만 가져오기
        self.device_num, self.device = self.load.u_device_load(self.demand, self.general_device)

        # 셋업 소요 시간 가져오기
        self.pm = self.load.make_array_PM('성형2공장')
        # 모델 체인지 시간 가져오기
        self.mc = self.load.make_array_MC('성형2공장')
        # 원형 조달 시간
        self.TAT = self.load.make_dict_TAT('성형2공장')
        # 성형 & 원형 제작 개당 소요 시간
        self.tt = self.load.make_dict_AS('성형2공장')
        # 현재 주문을 충족시키기 위해 필요한 PRT 종류 개수
        self.PRT_num = len(self.load.create_prt('성형2공장'))
        # 주문 충족을 위해 필요한 PRT 종류 불러오기
        self.prt = self.load.create_prt('성형2공장')
        # 소자 최대 가동시간 정보
        self.load_time = self.load.max_load('성형2공장')
        # PRT 초기 재고량
        self.stock = self.load.stock_prt(self.prt)

        # 주문 충족을 위해 필요한 PRT 종류 불러오기
        self.mol = self.load.create_mol('성형2공장')
        # MOL 초기 재고량
        self.mol_stock = self.load.stock_mol(self.mol)

        self.action_space = spaces.Discrete(len(self.device) + 1)  # 소자 종류 수 만큼 액션 + stay 액션
        self.observation_space = spaces.MultiDiscrete(
            [math.ceil(self.demand.max().max() + 1), len(self.demand.columns) + 1, max(self.load_time) + 1,
             len(self.device) + 1, 250000])

        # 하나의 mol 성형이 끝났을 때, 원래는 1
        self.reward_per_success = 1

        # # 세팅한 소자를 모두 사용했을 때
        # self.reward_operation_max = 100

        # 낮은 가동률에 대한 패널티
        self.reward_operation_rate1 = 10
        self.reward_operation_rate2 = 5

        # 잘못된 셋업에 대한 패널티
        self.reward_per_miss = 50

        # 원형에서 생산해야하는 PRT 수 count
        self.required_PRT = []

        self.p_due_time = 0

        self.viewer = None

        self.reset()

    def excess_penalty(self, due_time, c_time, amount):  # 현재 성형해야 하는 mol의 납기가 지났을 때, 패널티
        penalty1 = 0
        coef1 = 0  # 0.000001

        # processing_date = datetime(year, month, day)
        # due_date = datetime(year, due_month, due_day)

        difference = c_time - due_time
        # difference = delta.days
        penalty1 += coef1 * amount * (difference ** 2)

        return penalty1


    def abandon_penalty(self, p_amount_set_up, p_device):  # 세팅한 소자를 다 쓰지 않고 버렸을 때 패널티
        penalty2 = 0
        coef2 = 0.4
        y = 0

        y = max(p_amount_set_up - self.load_time[p_device] * 0.2, 0)
        penalty2 += coef2 * (y ** 2)

        return penalty2

    def check_feasibility(self, line, amount, time, tat, PRT_name):
        # PRT_demand와 현재 시간(날짜)를 가지고 원형 공정에서 납기가 가능한지 체크
        feasibility = True
        start_PRT = time - tat
        capa = self.load.check_linecapa(line, start_PRT)
        stock = self.stock.get(PRT_name, 0)
        # stock = self.stock(PRT_name)
        # stock = stock.get(PRT_name, 0)

        # 현재 다루고 있는 목표 모델 line의 원형공정 모델 생산 리스트 불러오기
        if not self.required_PRT:
            required_capa = capa
        else:
            line_PRT = [lst for lst in self.required_PRT if lst[0] == line]

            # 현재 다루고 있는 목표 모델의 time-tat보다 작은 모델 생산 리스트 불러오기
            constraint1 = [lst for lst in line_PRT if lst[2] < start_PRT]

            # constraint1에서 현재 사용하고 있는 capa 합
            constraint2 = sum(lst2[1] for lst2 in constraint1)

            # required_capa는 이용할 수 있는 capa 량
            required_capa = capa - constraint2

        # 제약 확인 불가능 시 False
        if required_capa < amount - stock:
            feasibility = False
        # 만들 수 있다는 판단이면 초기 재고 업데이트 필요
        else:
            # 초기 재고 소진
            if amount >= stock:
                self.stock[PRT_name] = 0
            # 초기 재고 잔량 업데이트
            else:
                self.stock[PRT_name] = abs(amount - stock)

        return feasibility

    # 원형공정 제약 o
    def load_model(self, p_device, selected_device, demand, p_mol_name):
        found_solution = False
        info = pd.DataFrame()
        p_mol_name = self.demand.columns[p_mol_name]

        if selected_device == len(self.device):
            info = pd.DataFrame({'time': 0,
                                 'mol_name': [0],
                                 'amount': 0})
        else:

            if p_device == selected_device:  # 이전과 같은 소자를 셋업한 경우
                name = self.device[selected_device]
                columns = list(map(str, demand.columns))
                mol_cols = [col for col in columns if name in col]
                new_df = demand[mol_cols]  # 셋업된 소자를 사용하는 mol들로 구성된 새로운 df 생성

                if (new_df == 0).all().all():  # 셋업된 소자를 사용하는 mol 모두 주문이 없을 때
                    info = pd.DataFrame({'time': 0,
                                         'mol_name': [0],
                                         'amount': 0})
                else:  # 주문이 있을 때
                    for i in range(len(new_df)):
                        if (new_df.loc[new_df.index[i], p_mol_name] != 0).any():  # 이전의 mol종류 먼저 고려, 연속생산
                            time = new_df.index[i]
                            amounts = new_df.loc[time, p_mol_name]
                            selected_PRT = demand.columns.get_loc(p_mol_name)
                            PRT_name = self.prt.iloc[selected_PRT]
                            line_num = self.load.check_line(PRT_name)
                            ratio = self.load.check_ratio(p_mol_name)
                            tat = self.TAT[PRT_name]

                            if self.check_feasibility(line_num, amounts * ratio, time, tat,
                                                      PRT_name):  # 첫번째 load_model 우선순위 상 가능한지 확인
                                self.required_PRT.append([line_num, amounts * ratio, time - tat])  # PRT 필요량 업데이트
                                info = pd.DataFrame({'time': [time],
                                                     'mol_name': p_mol_name,
                                                     'amount': amounts})
                                break

                            else:  # 원형공정 제약 만족하지 못할 때
                                continue  # 다음으로 우선순위에 해당하는 mol 찾기


                        elif (new_df.loc[:, p_mol_name] == 0).all():  # 이전 mol의 주문이 더 이상 없을 때
                            if (new_df.iloc[i] != 0).any():  # 납기일 빠른 거 먼저 생산
                                time = new_df.index[i]
                                mol_names = new_df.iloc[i][
                                    new_df.iloc[i] != 0].index.tolist()  # 납기일이 동일한 mol들이 존재할 때 해당 mol_name 기록
                                amounts = new_df.iloc[i][new_df.iloc[i] != 0].tolist()  # mol_name에 해당하는 amount도 기록
                                sorted_amounts = sorted(range(len(amounts)), key=lambda k: amounts[k],
                                                        reverse=True)  # amounts 큰 순서로 정렬
                                amounts = [amounts[i] for i in sorted_amounts]
                                mol_names = [mol_names[i] for i in sorted_amounts]  # mol_nams도 그에 맞게 재정렬
                                if len(mol_names) > 1:  # 납기일이 동일한 mol들이 존재 시
                                    for j in range(len(mol_names)):
                                        selected_PRT = demand.columns.get_loc(mol_names[j])
                                        PRT_name = self.prt.iloc[selected_PRT]
                                        line_num = self.load.check_line(PRT_name)
                                        ratio = self.load.check_ratio(mol_names[j])
                                        tat = self.TAT[PRT_name]
                                        amount = amounts[mol_names.index(mol_names[j])]
                                        if self.check_feasibility(line_num, amount * ratio, time, tat, PRT_name):
                                            self.required_PRT.append(
                                                [line_num, amount * ratio, time, tat])  # PRT 필요량 업데이트
                                            info = pd.DataFrame({'time': [time],
                                                                 'mol_name': mol_names[j],
                                                                 'amount': amount})
                                            found_solution = True
                                            break

                                        else:
                                            continue  # mol_names 중 제약 만족하는 mol 찾기

                                    if found_solution:
                                        break

                                else:  # 납기일이 동일한 mol이 없고 한 행에 한 mol의 주문만 존재 시
                                    selected_PRT = demand.columns.get_loc(mol_names[0])
                                    PRT_name = self.prt.iloc[selected_PRT]
                                    line_num = self.load.check_line(PRT_name)
                                    ratio = self.load.check_ratio(mol_names[0])
                                    tat = self.TAT[PRT_name]

                                    if self.check_feasibility(line_num, amounts[0] * ratio, time, tat,
                                                              PRT_name):  # 첫번째 load_model 우선순위 상 가능한지 확인
                                        self.required_PRT.append(
                                            [line_num, amounts[0] * ratio, time - tat])  # PRT 필요량 업데이트
                                        info = pd.DataFrame({'time': [time],
                                                             'mol_name': mol_names[0],
                                                             'amount': amounts[0]})
                                        break

                                    else:
                                        continue

            else:  # 이전 소자와 다른 소자 셋업한 경우
                name = self.device[selected_device]
                columns = list(map(str, demand.columns))
                mol_cols = [col for col in columns if str(name) in col]
                new_df = demand[mol_cols]  # 셋업된 소자를 사용하는 mol들로 구성된 새로운 df 생성

                if (new_df == 0).all().all():  # 셋업된 소자를 사용하는 mol 모두 주문이 없을 때
                    info = pd.DataFrame({'time': 0,
                                         'mol_name': [0],
                                         'amount': 0})
                else:  # 주문이 있을 때
                    for i in range(len(new_df)):
                        if (new_df.iloc[i] != 0).any():  # 납기일 빠른 거 먼저 생산
                            time = new_df.index[i]
                            mol_names = new_df.iloc[i][
                                new_df.iloc[i] != 0].index.tolist()  # 납기일이 동일한 mol들이 존재할 때 해당 mol_name 기록
                            amounts = new_df.iloc[i][new_df.iloc[i] != 0].tolist()  # mol_name에 해당하는 amount도 기록
                            sorted_amounts = sorted(range(len(amounts)), key=lambda k: amounts[k],
                                                    reverse=True)  # amounts 큰 순서로 정렬
                            amounts = [amounts[i] for i in sorted_amounts]
                            mol_names = [mol_names[i] for i in sorted_amounts]  # mol_nams도 그에 맞게 재정렬
                            if len(mol_names) > 1:  # 납기일이 동일한 mol들이 존재 시
                                for j in range(len(mol_names)):
                                    selected_PRT = demand.columns.get_loc(mol_names[j])
                                    PRT_name = self.prt.iloc[selected_PRT]
                                    line_num = self.load.check_line(PRT_name)
                                    ratio = self.load.check_ratio(mol_names[j])
                                    tat = self.TAT[PRT_name]
                                    amount = amounts[mol_names.index(mol_names[j])]
                                    if self.check_feasibility(line_num, amount * ratio, time, tat, PRT_name):
                                        self.required_PRT.append([line_num, amount * ratio, time, tat])  # PRT 필요량 업데이트
                                        info = pd.DataFrame({'time': [time],
                                                             'mol_name': mol_names[j],
                                                             'amount': amount})
                                        found_solution = True
                                        break

                                    else:
                                        continue  # mol_names 중 제약 만족하는 mol 찾기

                                if found_solution:
                                    break

                            else:  # 납기일이 동일한 mol이 없고 한 행에 한 mol의 주문만 존재 시
                                selected_PRT = demand.columns.get_loc(mol_names[0])
                                PRT_name = self.prt.iloc[selected_PRT]
                                line_num = self.load.check_line(PRT_name)
                                ratio = self.load.check_ratio(mol_names[0])
                                tat = self.TAT[PRT_name]

                                if self.check_feasibility(line_num, amounts[0] * ratio, time, tat,
                                                          PRT_name):  # 첫번째 load_model 우선순위 상 가능한지 확인
                                    self.required_PRT.append([line_num, amounts[0] * ratio, time - tat])  # PRT 필요량 업데이트
                                    info = pd.DataFrame({'time': [time],
                                                         'mol_name': mol_names[0],
                                                         'amount': amounts[0]})
                                    break

                                else:
                                    continue

        return info

    def update_model(self, demand, order_info, required_mol):

        demand.loc[demand.index == order_info.loc[0, 'time'], order_info['mol_name']] = required_mol

        return demand

    def generate_excel(self, total_time, production_ready, demand, load, assign):
        total_time_u = math.ceil(total_time)  # 총 소요 시간 올림처리
        production_ready = production_ready.applymap(
            lambda x: str(x).lstrip('[').rstrip(']'))  # state랑 action 저장해놓은거에서 필요없는 괄호 없애주기
        rein_result = production_ready[~production_ready.duplicated(keep='first')]  # prudoction_ready 파일에서 중복된 행 없애기

        # rein_result 프레임에서 'Action'열 제외하고 float으로 변환
        exclude_columns = ['Action']
        for column in rein_result.columns:
            if column not in exclude_columns:
                rein_result.loc[:, column] = rein_result[column].astype(float)

        # rein_result 프레임에서 'require_mol, amount_set_up, Action' 빼고 정수로 변환
        exclude_columns2 = ['require_mol', 'amount_setup', 'Action']
        for column in rein_result.columns:
            if column not in exclude_columns2:
                rein_result.loc[:, column] = np.ceil(rein_result[column]).astype(int)
        duplicates = rein_result[rein_result.duplicated(subset='time', keep='first')]
        while len(duplicates) != 0:
            duplicates = rein_result[rein_result.duplicated(subset='time', keep='first')]
            for index in duplicates.index:
                rein_result.loc[index, 'time'] += 1

        # 생산계획 출력할 파일 틀 만들어놓기
        production_plan = pd.DataFrame(columns=['Time'] + list(demand.columns))

        # 생산계획 파일에 시간을 추가하기 위해 형식 만들어놓기
        date = load.order['일자'][0]
        base_time = pd.Timestamp(date)
        production_plan['Time'] = [base_time + pd.Timedelta(hours=i) for i in range(total_time_u)]

        # 이제 생산계획 파일에 Time 열에 값 넣어주기, 위에서 보여줬던 total time_u 까지 1시간씩 오름차순으로 넣음
        production_plan['Time'] = [date + timedelta(hours=i) for i in range(total_time_u)]

        # 생산계획 초기화
        production_columns = [col for col in demand.columns if col != 'Time']

        # production_plan 데이터프레임에 선택한 열들을 0으로 초기화하여 추가
        production_plan[production_columns] = 0

        # 값 초기화
        p_mol_name, p_device = None, None

        # 이제 모델 열들에 있는 0 값들을 생산계획으로 채워 넣어주기 위해 for 문을 돌린다
        for i in range(len(rein_result)):
            mol_name = rein_result.iloc[i, 1]
            device = rein_result.iloc[i, 3]
            time = rein_result.iloc[i, 4]
            action = rein_result.iloc[i, 5]
            p_amount_setup = rein_result.iloc[i - 1, 2]
            amount_setup = rein_result.iloc[i, 2]

            use_time = time - rein_result.iloc[i - 1, 4] if i != 0 else time

            if use_time > 0:
                if action in ['Device A set up', 'Device B set up', 'Device C set up', 'Device D set up']:
                    # action 숫자 파악
                    action_num = {'Device A set up': 0, 'Device B set up': 1, 'Device C set up': 2}[action]
                    # 새로 생산하는 케이스
                    if p_device is None and p_mol_name is None:
                        pm_time = self.pm[-1][self.device_num[action_num]]
                        start_index = time - use_time
                        end_index = start_index + math.ceil(pm_time)
                        production_plan.iloc[start_index: end_index, mol_name + 1] = 's'
                        production_plan.iloc[end_index: time, mol_name + 1] = 3600 // assign[mol_name]

                    # 소자를 바꿔야 할 때
                    elif p_device != device and p_device is not None:
                        pm_time = self.pm[device][self.device_num[action_num]]
                        idle_time = math.ceil(p_amount_setup) # 폐기 시간 계산
                        idle_start_index = time - use_time
                        idle_end_index = idle_start_index + idle_time
                        set_start_index = idle_end_index
                        set_end_index = math.ceil(set_start_index + pm_time)
                        production_plan.iloc[idle_start_index: idle_end_index, p_mol_name + 1] = 'i'
                        production_plan.iloc[set_start_index: set_end_index, mol_name + 1] = 's'
                        production_plan.iloc[set_end_index: time, mol_name + 1] = 3600 // assign[mol_name]

                    # 몰네임이 같은데 소자를 다 소진한 케이스 같은 소자를 투입할 때
                    elif p_device == device and p_device is not None:
                        pm_time = self.pm[p_device][self.device_num[action_num]]
                        idle_time = math.ceil(p_amount_setup)
                        idle_start_index = time - use_time
                        idle_end_index = idle_start_index + idle_time
                        set_start_index = idle_end_index
                        set_end_index = math.ceil(set_start_index + pm_time)
                        production_plan.iloc[idle_start_index: idle_end_index, p_mol_name + 1] = 'i'
                        production_plan.iloc[set_start_index: set_end_index, mol_name + 1] = 's'
                        production_plan.iloc[set_end_index: time, mol_name + 1] = 3600 // assign[mol_name]

                    # 마지막 소자 처리
                    if i == len(rein_result) - 1:
                        pm_time = self.pm[device][self.device_num[action_num]]
                        idle_start_index = time - use_time
                        idle_end_index = math.ceil(idle_start_index + p_amount_setup)
                        set_start_index = idle_end_index
                        set_end_index = set_start_index + math.ceil(pm_time)
                        prod_start_index = set_end_index
                        prod_end_index = set_end_index + math.ceil(use_time - p_amount_setup - pm_time - amount_setup)
                        production_plan.iloc[idle_start_index: idle_end_index, p_mol_name + 1] = 'i'
                        production_plan.iloc[set_start_index: set_end_index, mol_name + 1] = 's'
                        production_plan.iloc[prod_start_index: prod_end_index, mol_name + 1] = 3600 // assign[mol_name]
                        production_plan.iloc[prod_end_index:time+1, mol_name + 1] = 'i'


                elif action == 'Device Not set up':
                    if mol_name != p_mol_name and rein_result.iloc[i, 2] !=0 and p_mol_name is not None:
                        p_mol_name = rein_result.iloc[i - 1, 1]
                        mc_time = self.mc[p_mol_name][mol_name]
                        start_index = time - use_time
                        end_index = math.ceil(start_index + mc_time)
                        production_plan.iloc[start_index:end_index, mol_name + 1] = 'c'
                        production_plan.iloc[end_index:time, mol_name + 1] = 3600 // assign[mol_name]

                    elif mol_name == 0 and rein_result.iloc[i, 2] == 0:
                        idle_time = math.ceil(p_amount_setup)
                        prod_start_index = time - use_time
                        prod_end_index = prod_start_index + math.ceil(use_time - idle_time)
                        idle_start_index = prod_end_index
                        idle_end_index = idle_start_index + idle_time
                        production_plan.iloc[prod_start_index: prod_end_index, p_mol_name + 1] = 3600 // assign[p_mol_name]
                        production_plan.iloc[idle_start_index: idle_end_index, p_mol_name + 1] = 'i'

                    else:
                        production_plan.iloc[time - use_time: time, mol_name + 1] = 3600 // assign[mol_name]

                    # 마지막 소자 처리
                    if i == len(rein_result) - 1:
                        if mol_name != p_mol_name:
                            p_mol_name = rein_result.iloc[i - 1, 1]
                            mc_time = self.mc[p_mol_name][mol_name]
                            start_index = time - use_time
                            end_index = start_index + math.ceil(mc_time)
                            prod_start_index = end_index
                            prod_end_index = time - math.ceil(amount_setup)
                            production_plan.iloc[start_index: end_index, mol_name + 1] = 'c'
                            production_plan.iloc[prod_start_index: prod_end_index, mol_name + 1] = 3600 // assign[mol_name]
                            production_plan.iloc[prod_end_index: time, mol_name + 1] = 'i'
                        else:
                            start_index = time - use_time
                            end_index = start_index + math.ceil(use_time - amount_setup)
                            production_plan.iloc[start_index: end_index, mol_name + 1] = 3600 // assign[mol_name]
                            production_plan.iloc[end_index: time+1, mol_name + 1] = 'i'


                p_mol_name = mol_name
                p_device = device

        # self.automate_excel(production_plan)

        # 생산계획 출력
        # rein_result.to_excel('C:/Users/admin/Desktop/AGV, INVENTORY 코드/rein_result_성형2공장.xlsx')
        # production_plan.to_excel('C:/Users/admin/Desktop/AGV, INVENTORY 코드/output1.xlsx', index=False)
        # 상태 전이 파일
        rein_result.to_excel('rein_result_성형2공장.xlsx')
        # 생산 계획
        production_plan.to_excel('output_성형2공장.xlsx', index=False)


    def automate_excel(self, production_plan):
        production_plan["Time"] = pd.to_datetime(production_plan["Time"])

        # Year, Month, Day, Date 열 추가
        production_plan["Year"] = production_plan["Time"].dt.year
        production_plan["Month"] = production_plan["Time"].dt.month
        production_plan["Day"] = production_plan["Time"].dt.day
        production_plan["Date"] = production_plan["Time"].dt.date

        columns_to_loop = production_plan.columns[production_plan.columns.str.startswith('MOL_')]

        # Setup 열 추가
        production_plan['Setup'] = 0
        for i in range(len(production_plan)):
            for j in columns_to_loop:
                if production_plan.loc[i, j] == 's':
                    production_plan.loc[i, 'Setup'] += 1

        # MC 열 추가
        production_plan['MC'] = 0
        for i in range(len(production_plan)):
            for j in columns_to_loop:
                if production_plan.loc[i, j] == 'c':
                    production_plan.loc[i, 'MC'] += 1

        # Idle 열 추가
        production_plan['Idle'] = 0
        for i in range(len(production_plan)):
            for j in columns_to_loop:
                if production_plan.loc[i, j] == 'i':
                    production_plan.loc[i, 'Idle'] += 1

        # Category 열 추가
        production_plan['Category'] = ''
        for i in range(len(production_plan)):
            if production_plan['Setup'][i] != 0:
                production_plan.loc[i, 'Category'] = 'Setup'
            elif production_plan['MC'][i] != 0:
                production_plan.loc[i, 'Category'] = 'MC'
            elif production_plan['Idle'][i] != 0:
                production_plan.loc[i, 'Category'] = 'Idle'
            else:
                for col in columns_to_loop:
                    if production_plan[col][i] > 0:
                        production_plan.loc[i, 'Category'] = col
                        break

        # Val 열 추가
        production_plan['Val'] = 1

    def calc_ut(self, plan):
        df = plan
        total_time = df.shape[0]  # 행 수를 통해 전체 시간 계산
        count_s = df.applymap(lambda x: x == 's').sum().sum()  # 전체 데이터 프레임에 s 개수
        count_mc = df.applymap(lambda x: x == 'c').sum().sum()  # 전체 데이터 프레임에 mc 개수
        count_i = df.applymap(lambda x: x == 'i').sum().sum()  # 전체 데이터 프레임에 i 개수
        ut_percent = (total_time - count_s - count_mc - count_i) / total_time  # 생산시간 / 전체 소요시간
        return print('가동률 :', ut_percent * 100, '%\nmc :', count_mc, 'hr\nset_up :', count_s, 'hr\nidle_time :',
                     count_i, 'hr')


    def step(self, action):
        # 현재 생산해야 하는 mol의 성형에 소요되는 시간 / 현재 셋업된 소자 잔량 / 현재 세팅된 소자 종류 / 현재 날짜
        required_mol, mol_name, amount_set_up, device, c_time = self.state

        # 셋업하기 이전의 소자 잔량 저장
        p_amount_set_up = amount_set_up
        p_device = device
        p_mol_name = mol_name
        p_time = c_time


        reward = 0

        done = False
        self.steps += 1
        info = {}

        if action == len(self.device):
            amount_set_up += 0
            device = p_device
            c_time = p_time
        elif action < len(self.device):
            amount_set_up = self.load_time[self.device_num[action]]
            device = action
            c_time = p_time
            c_time += self.pm[p_device][self.device_num[action]]
            c_time += p_amount_set_up
            self.setup_times.append(self.pm[p_device][self.device_num[action]])
            self.load_amount.append(amount_set_up)
            info['set_up'] = True
        else:
            raise Exception('bad action {}'.format(action))

        # 모델 투입 룰에 따라 모델 가져오기
        order_info = self.load_model(p_device, device, self.demand, p_mol_name)

        required_mol = order_info['amount'].values[0]

        # 초기재고 있으면 required_mol 줄이기
        if required_mol > 0:
            if self.mol_stock.iloc[p_mol_name].empty:
                pass
            else:
                if required_mol > int(self.mol_stock.iloc[p_mol_name]):
                    required_mol -= int(self.mol_stock.iloc[p_mol_name])
                    self.mol_stock.iloc[p_mol_name] = 0
                else:
                    required_mol = 0
                    self.mol_stock.iloc[p_mol_name] -= required_mol

        d_columns = list(map(str, self.demand.columns))
        mol_str = order_info['mol_name'].values[0]
        if mol_str == 0:
            mol_name = 0
        else:
            mol_name = d_columns.index(mol_str)

        # 현재 셋업된 소자량으로 생산가능한 mol
        time_per = self.tt[mol_name] / 3600
        possible_mol = amount_set_up / time_per

        # 모델 체인지 시간 반영. 만약 STAY라면 setup과 별개로 모델 체인지 발생
        if action == len(self.device):
            if p_mol_name == mol_name:
                c_time += 0
                self.model_change_times.append(0)
            elif p_mol_name != mol_name:
                c_time += self.mc[p_mol_name][mol_name]
                self.model_change_times.append(self.mc[p_mol_name][mol_name])


        # 만약 새로운 셋업 액션을 했더라면,모델 체인지 시간은 setup 내에서 이루어지기 때문에 미포함..
        else:
            self.model_change_times.append(0)
            c_time += 0

        # 버려지는 소자량에 대한 패널티 발생
        if p_amount_set_up > 0 and action != len(self.device):
            reward -= self.abandon_penalty(p_amount_set_up, p_device)
            info['abandon_amount'] = p_amount_set_up

            # 패널티가 존재한다는 것은 80% 보다 못 쓴 것
            if self.abandon_penalty(p_amount_set_up, p_device) > 0:
                info['failed_use'] = True



        # stay 한번에 그냥 다 낭비라고 보기
        if required_mol == 0 and action == len(self.device):
            if amount_set_up > 0:
                c_time += amount_set_up
                info['abandon_amount'] = amount_set_up
                reward -= self.abandon_penalty(amount_set_up, device)
                amount_set_up = 0


        # 소자 태우는 상태 전이 표현
        if required_mol < possible_mol and required_mol > 0:
            # self.running_times.append(required_mol / 6)
            reward += self.reward_per_success * required_mol
            c_time += required_mol * time_per
            possible_mol -= required_mol
            required_mol = 0
            info['success'] = True
            amount_set_up = possible_mol * time_per



        elif required_mol == possible_mol and required_mol > 0:
            self.running_times.append(amount_set_up)
            reward += self.reward_per_success * required_mol  # + self.reward_operation_max
            c_time += required_mol * time_per
            required_mol = 0
            info['success'] = True
            possible_mol = 0
            amount_set_up = possible_mol * time_per


        elif required_mol > possible_mol and required_mol > 0 and possible_mol > 0:
            self.running_times.append(amount_set_up)
            reward += self.reward_per_success * possible_mol  # + self.reward_operation_max
            c_time += possible_mol * time_per
            required_mol -= possible_mol
            possible_mol = 0
            amount_set_up = possible_mol * time_per

        due_time = order_info['time']  # 납기일.

        previous_step = [self.p_due_time, p_mol_name]  # 전 스텝 생산 제품의 납기, 몰이름 정보 저장
        c_step = [due_time, mol_name]  # 현 스텝 생산 제품의 납기, 몰이름 정보 저장


        if (due_time == 0).all():
            pass
        else:
            if (due_time < c_time).any():  # 납기 초과건에 대해
                if previous_step is c_step:  # 같으면 즉, 주문이 여러 스텝에 걸쳐 생산되면
                    reward -= self.excess_penalty(due_time, c_time, order_info['amount'].values[0])  # 납기 초과 패널티 적용
                else:  # 같지 않은 경우, 이전 스텝이 없던 경우
                    reward -= self.excess_penalty(due_time, c_time, order_info['amount'].values[0])  # 납기 초과 패널티 적용
                    info['excess'] = True  # 초과 건수 적용
                    self.excess_amount.append(required_mol)
                    # print("초과건수") # 20회 넘으면 안됨!
                    self.p_due_time = due_time

        info['usetime'] = c_time
        info['setup_times'] = self.setup_times
        info['mc_times'] = self.model_change_times
        info['load_amount'] = self.load_amount
        info['excess_amount'] = self.excess_amount

        # required_mol = order_info['result']
        if (order_info['mol_name'] == 0).all() and (order_info['amount'] == 0).all() and (
                order_info['time'] == 0).all():
            pass
        else:
            self.demand = self.update_model(self.demand, order_info, required_mol)

        # order에 주어진 모든 주문을 처리한 경우, 종료
        if (self.demand[self.demand.columns.tolist()] == 0).all().all():
            if action < len(self.device):
                info['abandon_amount'] = p_amount_set_up + amount_set_up
            elif action == len(self.device):
                info['abandon_amount'] = amount_set_up
            c_time += amount_set_up
            info['usetime'] = c_time
            done = True

        if self.steps == 3000:
            failed_order = (self.demand != 0).sum().sum()
            reward -= failed_order * 10000
            if action < len(self.device):
                info['abandon_amount'] = p_amount_set_up + amount_set_up
            elif action == len(self.device):
                info['abandon_amount'] = amount_set_up
            c_time += amount_set_up
            info['usetime'] = c_time
            done = True

        self.state = (required_mol, mol_name, amount_set_up, device, c_time)
        return np.array(self.state), reward, done, info



    def reset(self):
        self.state = (0, 0, 0, len(self.device),) + (0,)

        self.steps = 0
        self.running_times = []
        self.setup_times = []
        self.model_change_times = []
        self.load_amount = []
        self.excess_amount = []

        self.PRT = [0] * self.PRT_num

        self.demand = self.load.create_demand('성형2공장')

        self.stock = self.load.stock_prt(self.prt)

        # MOL 초기 재고량
        self.mol_stock = self.load.stock_mol(self.mol)

        # 필요한 PRT 수 reset
        self.required_PRT = []  # PRT_num = PRT 종류 수

        return np.array(self.state)

    def render(self):
        pass

    def close(self):
        pass


if __name__ == '__main__':

    # 데이터 불러오기
    sheet = ["DEMAND", "BOM", "MC", "최대로딩시간", "DEVICE", "TAT", "ASSIGN", "Regular PM", "LineCapa", "Stock", "Frozen"]
    data = pd.read_excel("생산계획_data.xlsx", sheet_name=sheet, engine='openpyxl')

    # 아래 코드 뭉치는 사용하는 소자 종류가 달라져도 그에 맞는 action_name 산출하도록 한다.
    load = load_data(data)
    assign = load.make_dict_AS('성형2공장')
    demand = load.create_demand('성형2공장')
    device = load.device_load('성형2공장')
    using_device, using_name = load.u_device_load(demand, device)
    action_dict = {index: device[index] for index in using_device}

    make_action_name = {act_num: f'Device {device[device_name]} set up' for act_num, device_name in
                        enumerate(using_device)}
    make_action_name[len(make_action_name)] = 'Device Not set up'

    # 위 코드를 통해 action_name 지정
    action_name = make_action_name

    # 디스플레이 클래스 환경 가져오기
    env = Display(data=data)
    eval_env = Display(data=data)

    # cb = EvalCallback_Display(eval_env=eval_env, n_eval_episodes=10, eval_freq=3000,
    #                           log_path="./model",
    #                           best_model_save_path="./best_model"
    #                           )

    # # log_dir = "ppo_log/"

    # # PPO 알고리즘에 환경불러와 모델 생성
    # model = PPO('MlpPolicy', env, verbose=0)  # 여기서 learning rate 수정 (1.0으로 2.5로?, -4는 임의로 수정하기, -1은 비추천)
    # # 얼마나 학습할 것인지
    # total_timesteps = int(1.0e4)
    # # 위에서 정해준 스텝만큼 학습 진행
    # model.learn(total_timesteps=total_timesteps, callback=cb)  # callback 사용 안할거면 cb 지우면 됨
    # 모델 저장
    # model.save("DSRL_model")

    # # TEST # #
    # production_ready = pd.DataFrame(columns=['require_mol', 'mol_name', 'amount_setup', 'device', 'time',
    #                                          'Action'])
    # count_abandon = 0
    # count_success = 0
    # count_excess = 0
    # count_failed = 0
    # count_set_up = 0
    # cumul_reward = 0
    # obs = env.reset()
    # for iter in range(3000):
    #     env.render()
    #     print("state: ", [round(x, 1) for x in np.array(env.state, dtype=float)])
    #     action = env.action_space.sample()
    #     obs, reward, done, info = env.step(action)
    
    #     cumul_reward += reward
    #     if info.get('abandon_amount'):
    #         count_abandon += info.get('abandon_amount')
    #     if info.get('success'):
    #         count_success += 1
    #     if info.get('excess'):
    #         count_excess += info.get('excess')
    #     if info.get('set_up'):
    #         count_set_up += 1
    
    #     print("action: ", action_name[action])
    #     print("reward this step: ", "{:.1f}".format(float(reward)))
    #     print("total reward: ", "{:.1f}".format(float(cumul_reward)))
    #     print("=" * 50)
    #     total_time = info['usetime']
    
    #     production_ready = production_ready.append({
    #         'require_mol': [round(x, 1) for x in np.array([env.state[0]], dtype=float)],
    #         'mol_name': [round(x, 1) for x in np.array([env.state[1]], dtype=float)],
    #         'amount_setup': [round(x, 1) for x in np.array([env.state[2]], dtype=float)],
    #         'device': [round(x, 1) for x in np.array([env.state[3]], dtype=float)],
    #         'time': [round(x, 1) for x in np.array([env.state[4]], dtype=float)],
    #         'Action': action_name[action],
    #     }, ignore_index=True)
    
    #     if done:
    #         break
    
    # print("Total successful move: ", count_success)
    # print("Total abandon amount", count_abandon)
    # print("Total excess count", count_excess)
    # print("Total set_up count", count_set_up)
    
    # env.generate_excel(total_time, production_ready, demand, load, assign)

    # #
    # # EVAL ENV # #
    # eval_env = DummyVecEnv([lambda: eval_env])
    # epi_rewards_discounted, epi_success_order, epi_abandon_amount, epi_excess_order, epi_setup, epi_times \
    #     = evaluate_policy_Display(model, eval_env,
    #                               n_eval_episodes=10,
    #                               render=False,
    #                               deterministic=False,
    #                               return_episode_rewards=True)
    # mean_reward_discounted, std_reward_discounted = np.mean(epi_rewards_discounted), np.std(epi_rewards_discounted)
    # mean_success, std_success = np.mean(epi_success_order), np.std(epi_success_order)
    # mean_abandon, std_abandon = np.mean(epi_abandon_amount), np.std(epi_abandon_amount)
    # mean_excess, std_excess = np.mean(epi_excess_order), np.std(epi_excess_order)
    # mean_setup, std_setup = np.mean(epi_setup), np.std(epi_setup)
    # mean_ep_length, std_ep_length = np.mean(epi_times), np.std(epi_times)
    # #
    # print(
    #     "episode_discounted_reward={:.2f} +/- {:.2f}".format(mean_reward_discounted,
    #                                                          std_reward_discounted),
    #     "episode_success={:.2f} +/- {:.2f}".format(mean_success, std_success),
    #     "episode_abandon={:.2f} +/- {:.2f}".format(mean_abandon, std_abandon),
    #     "episode_excess={:.2f} +/- {:.2f}".format(mean_excess, std_excess),
    #     "episode_setup={:.2f} +/- {:.2f}".format(mean_setup, std_setup))
    # # "Episode day: {:.2f} +/- {:.2f}".format(mean_ep_length, std_ep_length))
    # print("Episode day: {:.2f} +/- {:.2f}".format(mean_ep_length, std_ep_length))

    # Enjoy trained agent
    model = PPO.load("best_model/best_model.zip")

    # 학습 끝냈으니, 확인해보자
    obs = env.reset()
    count_abandon = 0
    count_success = 0
    count_excess = 0
    count_failed = 0
    count_set_up = 0
    cumul_reward = 0
    production_ready = pd.DataFrame(columns=['require_mol', 'mol_name', 'amount_setup', 'device', 'time',
                                             'Action'])  # 학습 결과를 출력하기 위해 state와 action 들어갈 수 있는 틀 만들어놓기
    for i in range(3000):
        print("state: ", [round(x, 1) for x in np.array(env.state, dtype=float)])
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
    
        cumul_reward += rewards
        if info.get('abandon_amount'):
            count_abandon += info.get('abandon_amount')
        if info.get('success'):
            count_success += 1
        if info.get('excess'):
            count_excess += 1
        if info.get('failed_use'):
            count_failed += 1
        if info.get('set_up'):
            count_set_up += 1
    
        total_time = info['usetime']
        setup_times = sum(info['setup_times'])
        mc_times = sum(info['mc_times'])
    
        print("action: ", action_name[action])
        print("reward this step: ", "{:.1f}".format(float(rewards)))
        print("total reward: ", "{:.1f}".format(float(cumul_reward)))
        print("=" * 50)
        #
        # 위에서 만들어 놓은 틀에 실제 state랑 action 넣기
        production_ready = production_ready.append({
            'require_mol': [round(x, 1) for x in np.array([env.state[0]], dtype=float)],
            'mol_name': [round(x, 1) for x in np.array([env.state[1]], dtype=float)],
            'amount_setup': [round(x, 1) for x in np.array([env.state[2]], dtype=float)],
            'device': [round(x, 1) for x in np.array([env.state[3]], dtype=float)],
            'time': [round(x, 1) for x in np.array([env.state[4]], dtype=float)],
            'Action': action_name[action],
        }, ignore_index=True)
        #
        if dones:
            break
    
    print("Number of successful processing: ", count_success)
    print("Total abandon amount", count_abandon)
    print("Total excess count", count_excess)
    print("Number of set up", count_set_up)
    print('Less than 80% usage', count_failed)
    print('Total time required: ', "{:.1f}".format(float(total_time / 24)), 'days')
    # mc 합, 셋업 합
    print('Sum of mc', "{:.1f}".format(float(mc_times / 24)), 'days')
    print('Sum of set up', "{:.1f}".format(float(setup_times / 24)), 'days')
    print('utilization', ((total_time - mc_times - setup_times - count_abandon) / total_time) * 100)
    
    env.generate_excel(total_time, production_ready, demand, load, assign)

    # 러닝커브 출력
    # loaded_model = PPO2.load("DSRL_model")
    # eval_env = DummyVecEnv([lambda: env])
    # eval_rewards = []
    # x_values = []
    # eval_freq = 1000
    #
    # for t in range(total_timesteps // eval_freq):
    #     # 모델 평가
    #     # mean_reward, _ = evaluate_policy_Display(loaded_model, eval_env, n_eval_episodes=10, render=False)
    #     # eval_rewards.append(mean_reward)
    #     epi_rewards_discounted, epi_success_order, epi_abandon_amount, epi_excess_order, epi_setup, epi_times = evaluate_policy_Display(
    #         loaded_model, eval_env, n_eval_episodes=10, render=False, return_episode_rewards=True)
    #     avg_abandon_amount = np.mean(epi_abandon_amount)
    #     eval_rewards.append(avg_abandon_amount)
    #
    #     #loaded_model.learn(total_timesteps=eval_freq)
    #
    #     # 학습 곡선 그리기
    #     x_values.append((t + 1) * eval_freq)
    #     print("x_values:", x_values)
    #     print("eval_rewards:", eval_rewards)
    #     plt.plot(x_values, eval_rewards)
    #     plt.xlabel('Timesteps')
    #     # plt.ylabel('Mean Reward')
    #     plt.ylabel('Mean Abandon Amount')
    #     plt.title('Learning Curve')
    #     plt.pause(0.01)  # 딜레이
    #
    # # 최종 학습 곡선 그리기
    # print("after all this years")
    # #x_values.append(total_timesteps)
    # #eval_rewards.append(avg_abandon_amount)
    # plt.plot(x_values, eval_rewards)
    # plt.xlabel('Timesteps')
    # # plt.ylabel('Mean Reward')
    # plt.ylabel('Mean Abandon Amount')
    # plt.title('Learning Curve')
    # plt.show()