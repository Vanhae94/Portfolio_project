import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, rc
import seaborn as sns
from sklearn.model_selection import train_test_split

################## 전처리 파트 #################

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

rain['일시'] = pd.to_datetime(rain['일시'],format='%Y-%m-%d')
mos['모기지수 발생일'] = pd.to_datetime(mos['모기지수 발생일'],format='%Y-%m-%d')
rain = rain[rain['일시'].dt.month.between(5,10)]

rain = rain.reset_index(drop=True)

mos = mos.sort_index(ascending=False)
mos = mos[mos['모기지수 발생일']>='2021-01-01']
mos = mos.reset_index(drop=True)

mos_weather = tem
mos_weather['강수량(mm)'] = rain['강수량(mm)']
mos_weather['평균습도(%rh)'] = hum['평균습도(%rh)']
mos_weather['종합모기지수'] = mos['종합모기지수']

mos_weather = mos_weather.dropna() # 결측값이 있는 행 날림
mos_weather.reset_index(drop=True, inplace=True)

# 모기예보단계 열 생성 ( 분류 준비 )
mos_weather['모기예보 단계'] = round(mos_weather['종합모기지수']/3,1)

def categorize(p):
    if 0 <= p < 25:
        return 1
    elif 25 <= p < 50:
        return 2
    elif 50 <= p < 75:
        return 3
    else:
        return 4

mos_weather['모기예보 단계'] = mos_weather['모기예보 단계'].apply(categorize)

# 평기 최저기 최고기 강수 평습 -> data / 모기예보 -> 레이블 === 분류데이터
mos_weather_data = mos_weather.drop(['종합모기지수','날짜','지점','모기예보 단계'],axis=1) # << 날짜는 인덱스로 보내자
mos_weather_lable = mos_weather.pop('모기예보 단계')

# 전처리 완료 된 데이터와 레이블
print(mos_weather_data)
print(mos_weather_lable)

# 훈련세트 테스트세트 분리
x_train, x_test, y_train, y_test = train_test_split(mos_weather_data, mos_weather_lable,random_state=18, test_size=0.2)

############## 모델 선정과 훈련 / 검증 파트 ###################

# 그래디언트 부스팅 분류 모델 선정
from sklearn.ensemble import GradientBoostingClassifier
gbc_model = GradientBoostingClassifier(random_state=18, n_estimators=100, learning_rate=0.1, max_depth=1, max_features= 'sqrt')
gbc_model.fit(x_train,y_train)

print("그래디언트 분류 훈련 세트 정확도: {:.2f}".format(gbc_model.score(x_train, y_train)))
print("그래디언트 분류 테스트 세트 정확도: {:.2f}".format(gbc_model.score(x_test, y_test)))

# 성능 향상을 위한 정규화
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 그리드 서치로 최적의 파라미터값을 탐색
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
grid_param = {
    'n_estimators': [ 100, 200, 300 ],
    'learning_rate': [ 0.1, 0.13, 0.15, 0.2 ],
    'max_depth': [ 1, 2 , 3]
}
grid_search = GridSearchCV(estimator=gbc_model, param_grid=grid_param, cv=5, scoring='accuracy')
grid_search.fit(x_train_scaled,y_train)

print("최적의 하이퍼파라미터:", grid_search.best_params_)
print("최적의 교차 검증 점수:", grid_search.best_score_)

best_gbc_model = grid_search.best_estimator_
rf_y_pred = best_gbc_model.predict(x_test_scaled)
print("베스트모델 정확도 :", best_gbc_model.score(x_test_scaled,y_test) )
print("베스트모델 정확도 accuracy 활용 :", accuracy_score(y_test,rf_y_pred))

y_test_pred = gbc_model.predict(x_test)

# 그래프로 시각화하여 비교하고 추이보기
comparison_df = pd.DataFrame({
    '실제값': y_test,
    '예측값': y_test_pred
    }).reset_index(drop=True)
plt.figure(figsize=(12, 6))
plt.plot(comparison_df.index, comparison_df['실제값'], marker='o', color='b', label='Actual value')
plt.plot(comparison_df.index, comparison_df['예측값'], marker='x', color='r', label='Predict value')

# 그래프 설정
plt.title('compare')
plt.xlabel('data index')
plt.ylabel('mos level')
plt.legend()
plt.grid(True)


# 25년도 예측

# 데이터를 23년도 기후데이터 x / 24년도 기후데이터 y
print(mos_weather.info())
mos_weather['날짜']=mos_weather['날짜'].str.strip()
mos_weather['날짜'] = pd.to_datetime(mos_weather['날짜'], format='%Y-%m-%d')
print(mos_weather)
print(mos_weather.info())
mos_weather23 = mos_weather[mos_weather['날짜'].dt.year==2023]
mos_weather24 = mos_weather[mos_weather['날짜'].dt.year==2024]
mos_weather23.reset_index(drop=True, inplace=True)
mos_weather24.reset_index(drop=True, inplace=True)
# 23년도 기후데이터 데이터
mos_weather23 = mos_weather23.drop(['종합모기지수','날짜','지점','종합모기지수'],axis=1)
# 24년도 기후데이터 레이블 -> 중요도를 뽑아보니 최저기온이 가장높음 최저기온을 y로 사용
mos_weather24y = mos_weather24.pop('최저기온(℃)')

# 기후와 모기지수 분석에서 특성의 중요도 뽑기
feature_importances = gbc_model.feature_importances_

# 중요도 시각화
plt.figure(figsize=(10,6))
plt.barh(mos_weather_data.columns, feature_importances,color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Feature Importances in Gradient Boosting Model')
plt.gca().invert_yaxis()
# 최저기온이 압도적으로 중요함



# 회귀모델 채택
from sklearn.ensemble import GradientBoostingRegressor
x25_train, x25_test, y25_train, y25_test = train_test_split(mos_weather23,mos_weather24y,random_state=18,test_size=0.2)

gbr25 = GradientBoostingRegressor()
gbr25.fit(x25_train,y25_train)

print(gbr25.score(x25_test,y25_test)) # 0.82 점

# 25년도 예측에 사용할 24년도 기후데이터
mos_weather24x = mos_weather[mos_weather['날짜'].dt.year==2024]
mos_weather24x = mos_weather24x.drop(['종합모기지수','날짜','지점','종합모기지수'],axis=1)

# 25년도 기후 회귀모델로 예측
mos_weather25 = gbr25.predict(mos_weather24x)
mos_weather25 = pd.DataFrame(mos_weather25)
pd.options.display.float_format = '{:.1f}'.format
print(mos_weather25)

# 25년도 모기예보 1~4단계 예측 
mos25 = gbc_model.predict(x_train_scaled)
print("2025년도 모기예보단계 예측\n",mos25)

# 그래프 확인
plt.show()
