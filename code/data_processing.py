import pandas as pd
from test_basic import get_risk_level
from test_blood import evaluate_blood
def transform_data(data):
    cat_columns = ['gender', 'is_coronary', 'is_diabetes', 'is_hypertension', 'is_smoking', 'is_drinking',
                   'is_family_history']
    for index, row in data.iterrows():
        # 1.处理类别变量
        for col in cat_columns:
            if row[col] == '是':
                data.at[index, col] = 1
            elif row[col] == '否':
                data.at[index, col] = 0
            elif row[col] == '男':
                data.at[index, col] = 1
            elif row[col] == '女':
                data.at[index, col] = 2
            elif row[col].strip() == '':
                data.at[index, col] = 0
        # 2.处理缺失值
        if row['years_of_smoking']== '':
            data.at[index, 'years_of_smoking'] = 0
    return data
def preprocess_data(data):
    data = data.drop('patient_sn', axis=1)
    data = data.drop('birth_date', axis=1)
    data = data.drop('height', axis=1)
    data = data.drop('weight', axis=1)
    data = transform_data(data)
    # 填补缺失值
    num_columns = ['lab_biochemical_tests_' + item + '_test_result' for item in
                   ['TC', 'TG', 'HDL-C', 'LDL-C', 'LPa', 'apoAI', 'apoB']]
    num_columns.append('heart_rate')
    num_columns.extend(
        ['age', 'years_of_smoking', 'respiratory_rate', 'diastolic_pressure', 'systolic_pressure', 'body_mass_index'])

    for col in num_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        data[col] = data[col].fillna(data[col].median())

    # 选择需要拼接的列
    columns_to_concatenate = ['exam_conclusion', 'exam_conclusion.1', 'exam_conclusion.2']

    # 使用 concat 函数拼接列
    concatenated_column = pd.concat([data[col] for col in columns_to_concatenate], axis=1)

    # 将拼接后的列添加到原始数据中
    data['concatenated_column'] = concatenated_column.apply(
        lambda row: '超声报告的结果是：' + row[0] + '心电图报告的结果是：' + row[1] + '。CT报告的结果是：' + row[2],
        axis=1)

    # 删除原始的拼接列
    data.drop(columns_to_concatenate, axis=1, inplace=True)
    # 添加通过 get_risk_level 方法得出的结果列
    basic_risk_prefix = "根据WHO/ISH风险预测图由性别、年龄、收缩压、血总胆固醇、吸烟状况和有无糖尿病估测发生致死性或非致死性心血管事件的年风险概率为："
    blood_risk_prefix = ("根据2024中国血脂管理指南中的中国成年人ASCVD总体风险评估流程图，通过血液中固定TC总胆固醇、LDL-C低密度脂蛋白胆固醇、年龄以及是是否是糖尿病患者得出的"
                         "10年ASCVD（动脉粥样硬化心血管疾病）风险概率为：")
    data['basic_risk'] = data.apply(lambda row: get_risk_level(row), axis=1)
    data['basic_risk'] = basic_risk_prefix + data['basic_risk'].astype(str)
    data['blood_risk'] = data.apply(lambda row: evaluate_blood(row), axis=1)
    data['blood_risk'] = blood_risk_prefix + data['blood_risk'].astype(str)
    return data