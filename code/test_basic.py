import pymysql

def connect_to_database():
    return pymysql.connect(host='mysql2.sqlpub.com', port=3307, user='tmeduser', password='7bOROJyy9IM7Q6hN', database='tmed_db')
# 取舍年龄
def calculate_age(age):
    if age <=40:
        return 40
    if age >=70 :
        return 70
    return int(age / 10)*10
# 取舍胆固醇
def calculate_cholesterol(cholesterol):
    if cholesterol < 4:
        return 4
    if cholesterol > 8 :
        return 8
    return int(cholesterol)
# 取舍收缩压
def calculate_systolic_pressure(systolic_pressure):
    if systolic_pressure < 120:
        return 120
    if systolic_pressure >= 180 :
        return 180
    if systolic_pressure < 140:
        return 120
    if systolic_pressure < 160:
        return 140
    if systolic_pressure < 180:
        return 160


def get_risk_level(data):
    age = calculate_age(data.get('age', 0))
    cholesterol = calculate_cholesterol(data.get('cholesterol', 0))
    systolic_pressure = calculate_systolic_pressure(data.get('systolic_pressure', 0))
    gender = data.get('gender', 0)
    is_diabetes = data.get('is_diabetes', 0)
    is_smoking = data.get('is_smoking', 0)
    connection = connect_to_database()
    cursor = connection.cursor()
    sql = f"""
    SELECT DISTINCT risk_level from cvd_risk 
        where age = {age} and gender = {gender} 
        and is_diabetes = {is_diabetes} and is_smoking = {is_smoking}
        and systolic_pressure={systolic_pressure} and cholesterol = {cholesterol}
    """
    cursor.execute(sql)
    result = cursor.fetchone()[0]
    return result

if __name__ == '__main__':
    data = {
        'age': 40,
        'cholesterol': 4,   # 总胆固醇
        'systolic_pressure': 120,  # 收缩压
        'gender': 1,
        'is_diabetes': 0, # 糖尿病
        'is_smoking': 0  # 吸烟
    }
    print(get_risk_level(data))