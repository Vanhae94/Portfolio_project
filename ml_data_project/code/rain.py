import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np

rain = pd.read_csv('./data/(월별)서울강수량.csv',encoding='cp949')
mos = pd.read_csv('./data/모기지수.csv',encoding='cp949')
mosDf = pd.DataFrame(mos)

# 종합 모기지수 (수부지 주거지 공원의 합)
mosDf['종합모기지수'] = mosDf['모기지수(수변부)'] + mosDf['모기지수(주거지)'] + mosDf['모기지수(공원)']
mosDf['모기지수 발생일'] = mosDf['모기지수 발생일'].str.strip()
mosDf['모기지수 발생일'] = pd.to_datetime(mosDf['모기지수 발생일'],format='%Y-%m-%d')

mosDf_reversed = mosDf.sort_index(ascending=False)


# 2021년 밑으로는 삭제
mosDf_reversed = mosDf_reversed[mosDf_reversed['모기지수 발생일'] >= '2021-01-01']

# 인덱스번호 재설정
mosDf_reversed = mosDf_reversed.reset_index(drop=True)

mosDf_reversed = mosDf_reversed.rename(columns={'모기지수 발생일':'날짜'})

mosDf_reversed.set_index('날짜',inplace=True)

# 월별 평균
total_mos_av = mosDf_reversed.resample('M').mean()
total_mos_av = total_mos_av.reset_index()
total_mos_av = total_mos_av[total_mos_av['날짜'].dt.month.between(5,10)]
total_mos_av = total_mos_av.sort_index(ascending=True) # 인덱스 번호 재조정해야함

total_mos_av = total_mos_av.reset_index(drop=True)

rain_mos = total_mos_av
rain_mos['강수량(mm)'] = rain['강수량(mm)']

# 상관계수
cor = rain_mos['종합모기지수'].corr(rain_mos['강수량(mm)'])
print('상관계수 :',cor) 
sns.scatterplot(x='강수량(mm)', y='종합모기지수', data=rain_mos)
plt.show()

print(rain_mos)

# 선형회귀
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


