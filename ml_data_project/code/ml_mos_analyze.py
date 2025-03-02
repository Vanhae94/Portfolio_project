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
print(mos_weather)

# 1차적으로 전처리가 완료된 데이터와 레이블
print(mos_weather_data)
print(mos_weather_lable)

# 훈련세트 테스트세트 분리
x_train, x_test, y_train, y_test = train_test_split(mos_weather_data, mos_weather_lable,random_state=18, test_size=0.2)

############## 모델 선정과 훈련 / 검증 파트 ###################

# 그래디언트 부스팅 분류 모델 선정
from sklearn.ensemble import GradientBoostingClassifier

gbc_model = GradientBoostingClassifier(random_state=18, learning_rate=0.1, max_depth=1, n_estimators=100)

# 성능 향상을 위한 정규화
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 모델 학습
gbc_model.fit(x_train_scaled, y_train)
# 예측 진행
y_pred = gbc_model.predict(x_test_scaled)

# 정확도 평가
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.2f}")
'''
# 그리드 서치로 최적의 파라미터 탐색
from sklearn.model_selection import GridSearchCV
grid_param = {
    'n_estimators': [ 100, 200, 300 ],
    'learning_rate': [ 0.1, 0.13, 0.15, 0.2 ],
    'max_depth': [ 1, 2 , 3]
}
grid_search = GridSearchCV(estimator=gbc_model, param_grid=grid_param, cv=5, scoring='accuracy')
grid_search.fit(x_train_scaled, y_train)

print("최적의 하이퍼파라미터:", grid_search.best_params_)
print("최적의 교차 검증 점수:", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test_scaled)

accuracy = accuracy_score(y_test,y_pred)
print(f"베스트 모델의 점수:{accuracy:.2f}")
'''
# 기후와 모기지수 분석에서 특성의 중요도 뽑기
feature_importances = gbc_model.feature_importances_

# 중요도 시각화
plt.figure(figsize=(10,6))
plt.barh(mos_weather_data.columns, feature_importances,color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Feature Importances in Gradient Boosting Model')
plt.gca().invert_yaxis()
plt.show()


