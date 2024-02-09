#!/usr/bin/env python
# coding: utf-8

# # 영업 성공 여부 분류 경진대회

# ## 1. 데이터 확인

# ### 필수 라이브러리

# In[206]:


import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


# ### 데이터 셋 읽어오기

# In[207]:


df_train = pd.read_csv("train.csv") # 학습용 데이터
df_test = pd.read_csv("submission.csv") # 테스트 데이터(제출파일의 데이터)


# In[208]:


df_train.head() # 학습용 데이터 살펴보기


# ## 2. 데이터 전처리

# ### 레이블 인코딩

# In[209]:


def label_encoding(series: pd.Series) -> pd.Series:
    """범주형 데이터를 시리즈 형태로 받아 숫자형 데이터로 변환합니다."""

    my_dict = {}

    # 모든 요소를 문자열로 변환
    series = series.astype(str)

    for idx, value in enumerate(sorted(series.unique())):
        my_dict[value] = idx
    series = series.map(my_dict)

    return series


# In[210]:


# 레이블 인코딩할 칼럼들
label_columns = [
    "customer_country",
    "business_subarea",
    "business_area",
    "business_unit",
    "customer_type",
    "enterprise",
    "customer_job",
    "inquiry_type",
    "product_category",
    "product_subcategory",
    "product_modelname",
    "customer_country.1",
    "customer_position",
    "response_corporate",
    "expected_timeline",
]

df_all = pd.concat([df_train[label_columns], df_test[label_columns]])

for col in label_columns:
    df_all[col] = label_encoding(df_all[col])


# 다시 학습 데이터와 제출 데이터를 분리합니다.

# In[211]:


for col in label_columns:  
    df_train[col] = df_all.iloc[: len(df_train)][col]
    df_test[col] = df_all.iloc[len(df_train) :][col]


# In[218]:


get_ipython().system('pip install --upgrade pip')
get_ipython().system('pip install impyute')
get_ipython().system('python -m pip install --upgrade pip')

from impyute.imputation.cs import mice
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

df_impute1 = pd.DataFrame(IterativeImputer(random_state=1234).fit_transform(df_train))
df_impute1.columns = df_train.columns
df_train = df_impute1

df_impute2 = pd.DataFrame(IterativeImputer(random_state=1234).fit_transform(df_test))
df_impute2.columns = df_test.columns
df_test = df_impute2


# ### 2-2. 학습, 검증 데이터 분리

# In[219]:


x_train, x_val, y_train, y_val = train_test_split(
    df_train.drop("is_converted", axis=1),
    df_train["is_converted"],
    test_size=0.2,
    shuffle=True,
    random_state=400,
)


# ## 3. 모델 학습

# ### 모델 정의 

# In[225]:


get_ipython().system('pip install xgboost')
get_ipython().system('pip install lightgbm')

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier

lgbm_model = LGBMClassifier(n_estimators=1000, learning_rate=0.1, max_depth=15, random_state=1234)
xgb_model = XGBClassifier(n_estimators=1000, learning_rate=0.6, max_depth=4, random_state=1234)

ensemble_model = VotingClassifier(estimators=[
    ('lgbm', lgbm_model),
    ('xgb', xgb_model)
], voting='soft')


# ### 모델 학습

# In[226]:


ensemble_model.fit(x_train, y_train)


# ### 모델 성능 보기

# In[227]:


def get_clf_eval(y_test, y_pred=None):
    confusion = confusion_matrix(y_test, y_pred, labels=[True, False])
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, labels=[True, False])
    recall = recall_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred, labels=[True, False])

    print("오차행렬:\n", confusion)
    print("\n정확도: {:.4f}".format(accuracy))
    print("정밀도: {:.4f}".format(precision))
    print("재현율: {:.4f}".format(recall))
    print("F1: {:.4f}".format(F1))


# In[228]:


pred = ensemble_model.predict(x_val)
get_clf_eval(y_val, pred)


# ## 4. 제출하기

# ### 테스트 데이터 예측

# In[203]:


# 예측에 필요한 데이터 분리
x_test = df_test.drop(["is_converted", "id"], axis=1)


# In[204]:


test_pred = ensemble_model.predict(x_test)
sum(test_pred) # True로 예측된 개수


# ### 제출 파일 작성

# In[205]:


# 제출 데이터 읽어오기 (df_test는 전처리된 데이터가 저장됨)
df_sub = pd.read_csv("submission.csv")
df_sub["is_converted"] = test_pred

# 제출 파일 저장
df_sub.to_csv("submission.csv", index=False)


# **우측 상단의 제출 버튼을 클릭해 결과를 확인하세요**
