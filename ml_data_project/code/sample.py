import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np

mos = pd.read_csv('./data/모기지수.csv',encoding='cp949')
tem = pd.read_csv('./data/(일별)서울기온분석.csv',encoding='cp949')
rain = pd.read_csv('./data/(월별)서울강수량.csv',encoding='cp949')

mosDf = pd.DataFrame(mos)
print(mosDf)
print(mosDf.info())

# 종합 모기지수 (수부지 주거지 공원의 합)
mosDf['종합모기지수'] = mosDf['모기지수(수변부)'] + mosDf['모기지수(주거지)'] + mosDf['모기지수(공원)']

print(mosDf)
print(tem)
print(tem.info())

# 기온데이터를 가공해서 각 연도별로 5월~ 10월 까지의 데이터만 남겨야함
tem['날짜'] = tem['날짜'].str.strip() # 공백제거
mosDf['모기지수 발생일'] = mosDf['모기지수 발생일'].str.strip()

tem['날짜'] = pd.to_datetime(tem['날짜'], format='%Y-%m-%d')
tem_filtered = tem[tem['날짜'].dt.month.between(5,10)]
mosDf['모기지수 발생일'] = pd.to_datetime(mosDf['모기지수 발생일'],format='%Y-%m-%d')

mosDf_reversed = mosDf.sort_index(ascending=False)

# 2021년 밑으로는 삭제
mosDf_reversed = mosDf_reversed[mosDf_reversed['모기지수 발생일'] >= '2021-01-01']
tem_filtered = tem_filtered[tem_filtered['날짜'] >= '2021-01-01']

# 인덱스번호 재설정
mosDf_reversed = mosDf_reversed.reset_index(drop=True)
tem_filtered = tem_filtered.reset_index(drop=True)

mosDf_reversed = mosDf_reversed.rename(columns={'모기지수 발생일':'날짜'})

# 데이터 병합
temMos = pd.merge(mosDf_reversed,tem_filtered, how='inner', on='날짜') # 기온과 모기 일별 데이터
print(temMos)
print(temMos['종합모기지수'])
print(temMos['평균기온(℃)'])

## 상관계수 ##
correlation = temMos['평균기온(℃)'].corr(temMos['종합모기지수'])
print("상관계수 :",correlation)

# 산점도 그래프 -> 그래프를 그리기 위해서는 두 데이터프레임이 병합해야함
sns.scatterplot(x = '평균기온(℃)', y = '종합모기지수', data=temMos)
plt.title('평균기온의 변화와 모기지수의 상관관계')
plt.xlabel('평균기온(℃)')
plt.ylabel('종합모기지수')
plt.show()
# 선형회귀선 추가(추세 확인용)
sns.lmplot(x = '평균기온(℃)', y = '종합모기지수', data=temMos)



# 머신러닝 레이블 -> 모기지수 / 데이터 나누기

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

temMosLable = temMos['모기지수(주거지)'] # 열을 따로 떼오면 이건 1차원 시리즈임 
temMosData = temMos['평균기온(℃)'] # data는 종속 변수이므로 2차원으로 가공후에 진행해야 함 
temMosData = pd.DataFrame(temMosData)
print(temMosData)

x_train, x_test, y_train, y_test = train_test_split(temMosData, temMosLable, random_state=0, test_size=0.2)
model = LinearRegression()
model.fit(x_train, y_train)

test_predict = model.predict(x_test)
print(test_predict)
print("훈련 정확도 평가 점수:", model.score(x_train, y_train))
print("테스트 정확도 평가 점수 : ", model.score(x_test, y_test) )

# 점수가 과소적합 -> 라쏘회귀 모델 사용
from sklearn.linear_model import Lasso
lasso_model = Lasso(alpha=0.0001, max_iter=10000)
lasso_model.fit(x_train, y_train)

print("훈련 정확도 평가 점수:", lasso_model.score(x_train, y_train))
print("테스트 정확도 평가 점수 : ", lasso_model.score(x_test, y_test) )
print("특성의 개수: ", lasso_model.coef_.size)
print("사용한 특성의 개수: ", np.sum(lasso_model.coef_ != 0))
print("특성 값이 0인 개수: ", np.sum(lasso_model.coef_ == 0))
#plt.show()

# 라쏘 부적격 / 최근접이웃모델 사용
from sklearn.neighbors import KNeighborsRegressor
maxi = 0
num = 0
for i in range(1,101) :
    knn_model = KNeighborsRegressor(n_neighbors=i)
    knn_model.fit(x_train, y_train)
    knn_result = knn_model.score(x_test,y_test)
    if ( knn_result > maxi ) :
        maxi = knn_result
        num = i

knn_model = KNeighborsRegressor(n_neighbors=num)
knn_model.fit(x_train, y_train)
print("-----------------------------------------------------------------------------------------------------")
print("훈련 정확도 평가 점수:", knn_model.score(x_train, y_train))
print("테스트 정확도 평가 점수 : ", maxi )
print("가장 높은점수 값을 갖는 이웃 수 : ",num)

# 시각화
# 산점도 
plt.figure(figsize=(10,6))
sns.scatterplot(x=temMosData['평균기온(℃)'], y=temMosLable, color='blue', label="실제 데이터")
# 선형 회귀선
sns.regplot(x=temMosData['평균기온(℃)'], y=temMosLable, scatter=False, color='red', label='선형 회귀선')
# 2차 다항회귀 선
sns.regplot(x=temMosData['평균기온(℃)'], y=temMosLable, scatter=False, color='green', label='2차 다항 회귀선', order=2 )

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# 2차 다항 회귀 모델 사용
poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

# 다항 회귀 모델 훈련 및 평가
model_poly = LinearRegression()
model_poly.fit(x_train_poly, y_train)

print("다항 회귀 훈련 정확도 평가 점수:", model_poly.score(x_train_poly, y_train))
print("다항 회귀 테스트 정확도 평가 점수:", model_poly.score(x_test_poly, y_test))

# plt.xlabel('평균기온 (℃)')
# plt.ylabel('종합 모기지수')
# plt.title('평균기온과 종합 모기지수의 관계')
# plt.legend()
# plt.show()


from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=5, random_state=1)
forest.fit(x_train, y_train)
print(forest.score(x_test,y_test))