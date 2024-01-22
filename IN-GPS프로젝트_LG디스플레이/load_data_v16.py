import pandas as pd
import numpy as np
from datetime import datetime

class load_data:
    def __init__(self, data):
        # 파일 내 시트들 불러오기
        self.order = data['DEMAND']
        self.bom = data['BOM']
        self.MC = data['MC']
        self.loading = data['최대로딩시간']
        self.device = data['DEVICE']
        self.TAT = data['TAT']
        self.assign = data['ASSIGN']
        self.pm = data['Regular PM']
        self.linecapa = data['LineCapa']
        self.stock = data['Stock']
        self.frozen = data['Frozen']


    # 주문 파일 읽기 쉽도록 변환
    def create_demand(self, factory_name, line_number=None):
        order = self.order.copy()
        assign = self.assign.copy()

        if line_number is not None:
            order = order.merge(assign, on='모델명', how='left')
            order.columns = order.columns.str.replace('_x', '')
            selected_columns = ['공장명', '모델명', '수량', '일자', '라인번호']  # 필요한 열만 선택
            order = order[selected_columns]

        order['일자'] = pd.to_datetime(order['일자'])
        order['일자'] = pd.to_datetime(order['일자'].dt.date) + pd.to_timedelta('18:00:00')
        base_time = datetime(2020, 6, 1, 18)
        order['일자'] = order['일자'].apply(lambda x: int((x - base_time).total_seconds() / 3600))
        factories = order['공장명'].unique()  # 공장명 고유값 뽑기

        if line_number == None:
            demand_data = {}
            for factory in factories:  # 필요한 형태로 데이터 가공
                df_name = factory.replace(" ", "")  # 변수 이름에 공백이 포함되지 않도록 조정
                df = order[order['공장명'] == factory].copy()
                demand_data[df_name] = df.pivot_table(index='일자', columns='모델명', values='수량')
                demand_data[df_name] = demand_data[df_name].fillna(0)
            return demand_data[factory_name]

        else:
            for factory in factories:
                if factory_name == factory:
                    df = order[(order['공장명'] == factory) & (order['라인번호'] == line_number)]
                    df = df.pivot_table(index='일자', columns='모델명', values='수량')
                    df = df.fillna(0)
                    return df

    # 해당 계획을 완성하기 위해 필요한 PRT들 종류 뽑기
    def create_prt(self, factory_name, line_number=None):
        order = self.order.copy()
        bom = self.bom.copy()
        assign = self.assign.copy()

        order_dup = order[order['공장명'] == factory_name].drop_duplicates(subset='모델명')
        bom = bom[bom['소비공정'] == '원형']

        if line_number is not None:
            order_dup = order_dup.merge(assign, on='모델명')
            order_dup.columns = order_dup.columns.str.replace('_y', '')
            selected_columns = ['공장명', '모델명', '수량', '일자', '라인번호']
            order_dup = order_dup[selected_columns]
            order_dup = order_dup[order_dup['라인번호'] == line_number]

        model_name = order_dup['모델명'].values
        bom_demand = bom[bom['모델명'].isin(model_name)]
        prt_dict = bom_demand['소비모델']

        return prt_dict

    # 현재 투입하고자 하는 PRT는 어느 라인에서 만들어지는지
    def check_line(self, prt_name):
        assign = self.assign.copy()

        line_num = assign[assign['모델명'] == prt_name]['라인번호'].item()

        return line_num

    # 우리가 만들고자 하는 mol 1개를 생산하는데 필요한 PRT 단위수량
    def check_ratio(self, mol_name):
        bom = self.bom.copy()

        check_bom = bom.drop_duplicates(subset='모델명')

        ratio = check_bom[check_bom['모델명'] == mol_name]['단위수량']

        return ratio.item()

    # 소자 종류 확인
    def device_load(self, factory_name, line_number=None):
        device = self.device.copy()

        if line_number == None:
            find_d = device[device['공장명'] == factory_name]

        else:
            find_d = device[(device['공장명'] == factory_name) & (device['라인번호'] == line_number)]

        device_types = sorted(find_d['점검타입'].unique())

        return device_types

    def u_device_load(self, demand, device_types):
        # 우리가 해당 모델에서 사용할 device만 demand에서 추출
        columns = demand.columns.tolist()
        device_set = set(col.replace('MOL_', '').strip('_0123456789') for col in columns)
        device_list = sorted(device_set)

        using_device = [device_types.index(item) for item in device_list]

        return using_device, device_list

    # 최대가동시간 불러오기
    def max_load(self, factory_name):

        loading = self.loading.copy()

        result = loading.groupby(['공장명', '점검그룹타입'])['최대로딩시간'].max()
        result = pd.DataFrame(result).reset_index()

        factory_dfs = {}
        factory_groups = result.groupby('공장명')
        for factory_n, factory_df in factory_groups:
            factory_dfs[factory_n] = factory_df

        # 공장 개수만큼 [0,0,0,0,0]을 갖는 변수 생성
        for i in factory_dfs:
            globals()["f_" + str(i)] = [0, 0, 0, 0, 0]

        fac_list = factory_dfs[factory_name]['점검그룹타입'].tolist()

        var_name = "f_" + str(factory_name)
        var_value = globals()[var_name]
        if 'A' in fac_list:
            var_value[0] = factory_dfs[factory_name].loc[factory_dfs[factory_name]['점검그룹타입'] == 'A', '최대로딩시간'].max()
        if 'B' in fac_list:
            var_value[1] = factory_dfs[factory_name].loc[factory_dfs[factory_name]['점검그룹타입'] == 'B', '최대로딩시간'].max()
        if 'C' in fac_list:
            var_value[2] = factory_dfs[factory_name].loc[factory_dfs[factory_name]['점검그룹타입'] == 'C', '최대로딩시간'].max()
        if 'D' in fac_list:
            var_value[3] = factory_dfs[factory_name].loc[factory_dfs[factory_name]['점검그룹타입'] == 'D', '최대로딩시간'].max()
        if 'E' in fac_list:
            var_value[4] = factory_dfs[factory_name].loc[factory_dfs[factory_name]['점검그룹타입'] == 'E', '최대로딩시간'].max()

        var_value = tuple(var_value)

        return var_value

    def make_array_MC(self, factory, line_number=None):
        order = self.order.copy()
        mc = self.MC.copy()
        demand = order[order['공장명'] == factory]['모델명'].values
        sorted_demand = sorted(np.unique(demand))

        if line_number == None:
            # 공장명에 따라 데이터프레임 구분
            data = mc[(mc['공장명'] == factory)]

        else:
            # 공장명과 라인에 따라 데이터프레임 구분
            data = mc[(mc['공장명'] == factory) & (mc['라인번호'] == line_number)]

        # demand에 있는 걸로 mol 데이터 정리
        data = data[data['FROM'].isin(sorted_demand)]
        data = data[data['TO'].isin(sorted_demand)]

        # array의 행/열 요소를 각 공장 라인에 따라 뽑아내기
        from_values = np.unique(data['FROM'])
        to_values = np.unique(data['TO'])

        # 빈 array 생성
        array = np.zeros((len(from_values), len(to_values)), dtype=float)

        # data 돌려보면서 FROM과 TO에 따른 모델변경시간 정리 후 array에 넣기
        for _, row in data.iterrows():
            from_index = np.where(from_values == row["FROM"])[0][0]
            to_index = np.where(to_values == row["TO"])[0][0]
            array[from_index, to_index] = row["모델변경시간"]

        # 결측값 채우기
        non_zero_values = array[array != 0]
        total_mean = np.mean(non_zero_values)
        array[array == 0] = int(total_mean)

        return array


    # 셋업시간 정보 불러오기
    def make_array_PM(self, factory, line_number=None):
        order = self.order.copy()
        pm = self.pm.copy()
        demand = order[order['공장명'] == factory]['모델명'].copy()
        demand['ELEMENT'] = demand.str.split('_').str[1]
        element = sorted(np.unique(demand['ELEMENT']))
        element.append('Shutdown')


        # FROM 에 속한 값, TO열에 속한 값 정리
        from_values = np.array(element)
        to_values = np.array(element)

        if line_number == None:
            # 공장명에 따라 데이터프레임 구분
            data = pm[pm['공장명'] == factory]
        else:
            # 공장명과 라인에 따라 데이터프레임 구분
            data = pm[(pm['공장명'] == factory) & (pm['라인번호'] == line_number)]
            unique_data = data['FROM'].unique()

            intersec = set(from_values).intersection(set(unique_data))
            from_values = np.array(sorted(list(intersec)))
            to_values = np.array(sorted(list(intersec)))

        data = data[data['FROM'].isin(from_values)]
        data = data[data['TO'].isin(to_values)]

            # 빈 array 생성
        array = np.zeros((len(from_values), len(to_values)), dtype=int)

        # data 돌려보면서 FROM과 TO에 따른 소요시간 정리 후 array에 넣기
        for _, row in data.iterrows():
            from_index = np.where(from_values == row["FROM"])[0][0]
            to_index = np.where(to_values == row["TO"])[0][0]
            array[from_index, to_index] = row["소요시간"]

        return array

    # 원형에 대한 TAT 정보 가져오기
    def make_dict_TAT(self, factory, line_number=None):
        order = self.order.copy()
        tat = self.TAT.copy()
        bom = self.bom.copy()
        assign = self.assign.copy()

        merged_df = order.merge(bom, on='모델명', how='left')
        merged_df = merged_df.merge(assign, on='모델명', how='left')
        merged_df = merged_df.drop(columns=['공장명_y', '공정_y']).loc[~merged_df['소비모델'].str.startswith('MOL')]

        if line_number == None:
            demand = merged_df[merged_df['공장명_x'] == factory]['소비모델'].values

        else:
            demand = merged_df[(merged_df['공장명_x'] == factory) & (merged_df['라인번호'] == line_number)]['소비모델'].values

        if demand.size == 0:
            return None
        else:
            prt_demand = sorted(np.unique(demand))
            tat_data = tat[tat['모델명'].isin(prt_demand)]
            tat_data = tat_data[['모델명', 'TAT']]
            tat_dict = tat_data.set_index('모델명').T.squeeze().to_dict()

        return tat_dict

    # 생산시간(소요시간) 가져오기(성형, 원형)
    def make_dict_AS(self, factory, line_number=None):
        order = self.order.copy()
        assign = self.assign.copy()

        if line_number == None:
            # 성형공장과 성형공장 라인에 따라 구분
            assign_data = assign[assign['공장명'] == factory]
            # 딕셔너리로 만듦
            assign_data = assign_data[['모델명', 'T/T (s)']]

            if assign_data.empty:
                return None
            assign_dict = assign_data['T/T (s)'].to_list()

            return assign_dict

        else:
            order = order[(order['공장명'] == factory)]
            models = sorted(order['모델명'].unique())

            assign_list = []

            # 성형공장과 성형공장 라인에 따라 구분
            assign_data = assign[(assign['공장명'] == factory) & (assign['라인번호'] == line_number)]

            for model in models:
                assign_data_model = assign_data[assign_data['모델명'] == model]
                for value in assign_data_model['T/T (s)']:
                    assign_list.append(float(value))

            return assign_list

    # 시간 t까지 생산 가능한 원형 한도 계산
    def check_linecapa(self, line, start_prt):
        linecapa = self.linecapa.copy()

        start_date = '2020-06-01'
        start_date = pd.to_datetime(start_date)
        linecapa['일자'] = pd.to_datetime(linecapa['일자'])
        linecapa['일자'] = (linecapa['일자'] - start_date).dt.days + 1
        linecapa1 = linecapa[linecapa['라인번호'] == line]
        index_num = linecapa1.index[linecapa1['일자'] <= start_prt][-1]
        capa = sum(i for i in linecapa1.loc[:index_num]['한도'])

        return capa

    # prt stock 딕셔너리로 만듦
    def stock_prt(self, prt_dict):
        stock = self.stock.copy()
        prt_dict_0 = {}
        for i, r in stock.iterrows():
            k = r['소비모델명']
            v = r['재고량']
            if k.startswith('PRT'):
                prt_dict_0[k] = v

        select_key = prt_dict.tolist()
        select_dict = {key: prt_dict_0[key] for key in select_key}

        return select_dict

    # 생산할 몰 파악 후 시리즈로 만듦
    def create_mol(self, factory):
        order = self.order.copy()
        order = order[order['공장명'] == factory]['모델명'].unique()
        order = list(order)

        return pd.Series(sorted(order))

    # mol stock 데이터프레임으로 만듦(frozen 시트의 것)
    def stock_mol(self, mol_df):
        stock = self.frozen.copy()

        common_prt_models = mol_df[mol_df.isin(stock['모델명']).values].dropna()

        stock_mol_dict = {}
        for model in mol_df:
            stock_mol_dict[model] = 0

        for model in common_prt_models:
            stock_value = stock.loc[stock['모델명'] == model, '수량'].iloc[0]
            if not pd.isna(stock_value):
                stock_mol_dict[model] = stock_value

        result_df = pd.DataFrame.from_dict(stock_mol_dict, orient='index', columns=['수량'])

        return result_df
    

if __name__ == '__main__':

    sheet = ["DEMAND", "BOM", "MC", "최대로딩시간", "DEVICE", "TAT", "ASSIGN", "Regular PM", "LineCapa", "Stock", "Frozen"]
    data = pd.read_excel("생산계획_data.xlsx", sheet_name=sheet, engine='openpyxl')

    load = load_data(data)
    pm = load.make_array_PM('성형2공장')
    print(pm)
