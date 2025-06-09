# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

class RandomForestModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state
        )
        self.label_encoders = {}
        
    def preprocess_data(self, df, is_train=True):  # 데이터 전처리 함수
        # 불필요한 컬럼 제거
        df = df.drop(columns=['UID'], errors='ignore')

        # 파생 변수 생성 (0으로 나누는 문제 방지)
        df['부채 비율'] = np.where(df['최대 신용한도'] == 0, np.nan, df['현재 미상환 신용액'] / df['최대 신용한도'])
        df['신용 점수 대비 부채 비율'] = np.where(df['신용 점수'] == 0, np.nan, df['부채 비율'] / df['신용 점수'])
        df['연체 리스크 지표'] = df['신용 문제 발생 횟수'] * df['마지막 연체 이후 경과 개월 수']
        df['월 소득 대비 부채 비율'] = np.where(df['연간 소득'] == 0, np.nan, df['월 상환 부채액'] / (df['연간 소득'] / 12))
        df['총 부채 대비 월 상환액'] = np.where((df['현재 대출 잔액'] + df['현재 미상환 신용액']) == 0, np.nan, df['월 상환 부채액'] / (df['현재 대출 잔액'] + df['현재 미상환 신용액']))
        df['연간 소득 대비 최대 신용한도 비율'] = np.where(df['연간 소득'] == 0, np.nan, df['최대 신용한도'] / df['연간 소득'])

        # 무한대 값(NaN) 변환 후 결측치 처리
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = ['주거 형태', '현재 직장 근속 연수', '대출 목적', '대출 상환 기간']

        # 수치형 변수: 중앙값 대체
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        # 범주형 변수: 'Unknown'으로 대체
        for col in categorical_cols:
            df[col] = df[col].fillna('Unknown')

        # 범주형 변수 인코딩
        if is_train:
            self.label_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        else:
            for col in categorical_cols:
                if col in self.label_encoders:
                    df[col] = df[col].map(lambda x: self.label_encoders[col].transform([x])[0] if x in self.label_encoders[col].classes_ else -1)

        return df

    def train(self, X_train, y_train):  # 모델 학습
        self.model.fit(X_train, y_train)
        
    def predict_proba(self, X):  # 예측 확률 반환
        return self.model.predict_proba(X)[:, 1]
        
    def predict(self, X):  # 예측 레이블 반환
        return self.model.predict(X)
        
    def get_feature_importance(self):  # 특성 중요도 반환
        return self.model.feature_importances_ 