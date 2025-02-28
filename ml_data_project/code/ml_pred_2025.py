import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, rc
import seaborn as sns
from sklearn.model_selection import train_test_split

################## 데이터 전처리 파트 #################

# 폰트 설정
font_path = "H2GTRM.TTF"
font_name=font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

mos = pd.read_csv("./code/data/모기지수.csv" ,encoding="cp949")
rain = pd.read_csv("./code/data/(일별)서울강수량.csv", encoding="cp949")
hum = pd.read_csv("./code/data/(일별)서울시습도.csv", encoding="cp949")
tem = pd.read_csv("./code/data/(일별)서울기온분석.csv", encoding="cp949")

# 종합 모기지수 (수부지 주거지 공원의 합)
mos['종합모기지수'] = mos['모기지수(수변부)'] + mos['모기지수(주거지)'] + mos['모기지수(공원)']

# 데이터를 가공해서 각 연도별로 5월~ 10월 까지의 데이터만 남겨야함
rain['일시'] = rain['일시'].str.strip() # 공백제거
mos['모기지수 발생일'] = mos['모기지수 발생일'].str.strip()
tem['날짜'] = tem['날짜'].str.strip()

rain['일시'] = pd.to_datetime(rain['일시'],format='%Y-%m-%d')
mos['모기지수 발생일'] = pd.to_datetime(mos['모기지수 발생일'],format='%Y-%m-%d')
rain = rain[rain['일시'].dt.month.between(5,10)]

rain = rain.reset_index(drop=True)

mos = mos.sort_index(ascending=False)
mos = mos[mos['모기지수 발생일']>='2021-01-01']
mos = mos.drop_duplicates(subset=["모기지수 발생일"], keep='first') # 날짜중복 제거
mos = mos.reset_index(drop=True)

mos_weather = tem
mos_weather['강수량(mm)'] = rain['강수량(mm)']
mos_weather['평균습도(%rh)'] = hum['평균습도(%rh)']
mos_weather['종합모기지수'] = mos['종합모기지수']

mos_weather = mos_weather.dropna() # 결측값이 있는 행 날림
mos_weather.reset_index(drop=True, inplace=True)
print(mos_weather)
# 모기예보단계 열 생성 ( 분류 준비 )
mos_weather['모기예보 단계'] = round(mos_weather['종합모기지수']/3,1)

# 단계 분류함수
def categorize(p):
    if 0 <= p < 25:
        return 1
    elif 25 <= p < 50:
        return 2
    elif 50 <= p < 75:
        return 3
    else:
        return 4
    
# 단계 분류
mos_weather['모기예보 단계'] = mos_weather['모기예보 단계'].apply(categorize)
# 날짜를 인덱스로 만들기
mos_weather = mos_weather.set_index('날짜')

# 평기 최저기 최고기 강수 평습 -> data / 모기예보 -> 레이블 === 분류데이터
mos_weather_data = mos_weather.drop(['종합모기지수','지점','모기예보 단계'],axis=1)
mos_weather_lable = mos_weather['모기예보 단계'].copy()

###################### 2025년도 예측 파트 ######################## 총 184일
from fbprophet import Prophet

