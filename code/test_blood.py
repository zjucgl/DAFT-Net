
def evaluate_blood(row_data):
    high_level = '高危'
    mid_level = '中危'
    low_level = '低危'
    ldc_c = row_data.get('lab_biochemical_tests_LDL-C_test_result',0)
    tc = row_data.get('lab_biochemical_tests_TC_test_result',0)
    risk_factor = 0

    if ldc_c is not None and ldc_c >= 4.9:
        return high_level
    if tc is not None and tc >= 7.2:
        return high_level

    # 风险因素累加
    if row_data.get('is_diabetes') == 1 and row_data.get('age', 0) >= 40:
        return high_level
    risk_factor += row_data.get('is_diabetes', 0)
    risk_factor += row_data.get('is_smoking', 0)
    risk_factor += 1 if row_data.get('body_mass_index', 0) >= 30 else 0
    risk_factor += row_data.get('is_hypertension', 0)

    # 根据不同血脂范围判断风险等级
    if row_data.get('is_hypertension') == 0:
        if 3.1 <= tc < 4.1 or 1.8 <= ldc_c < 2.6:
            return low_level
        if 4.1 <= tc < 5.2 or 2.6 <= ldc_c < 3.4:
            return mid_level if risk_factor > 2 else low_level
        if 5.2 <= tc < 7.2 or 3.4 <= ldc_c < 4.9:
            return mid_level if risk_factor > 1 else low_level
    else:
        if 3.1 <= tc < 4.1 or 1.8 <= ldc_c < 2.6:
            if risk_factor > 1:
                return low_level
            elif risk_factor == 2:
                return mid_level
            elif risk_factor >= 3:
                return high_level
        if 4.1 <= tc < 5.2 or 2.6 <= ldc_c < 3.4:
            if risk_factor == 0:
                return low_level
            elif risk_factor == 1:
                return mid_level
            elif risk_factor > 2:
                return high_level
        if 5.2 <= tc < 7.2 or 3.4 <= ldc_c < 4.9:
            if risk_factor == 0:
                return low_level
            elif risk_factor == 1:
                return mid_level
            elif risk_factor > 2:
                return high_level
    return low_level
if __name__ == '__main__':
    data = {
        "lab_biochemical_tests_LDL-C_test_result": 2.5,
        "lab_biochemical_tests_TC_test_result": 2.5,
        "is_diabetes": 1,
        "is_smoking": 1,
        "is_hypertension": 1,
        "body_mass_index": 30,
        "age": 40,
    }
    print(evaluate_blood(data))