# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from model_rf import RandomForestModel
from model_lr import LogisticRegressionModel

def main():
    # 데이터 경로 설정
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    submission_path = 'data/sample_submission.csv'
    
    # 데이터 로드
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # 데이터 전처리 (RandomForest 모델의 전처리 함수 사용)
    # Note: custom preprocess_data returns a DataFrame suitable for sklearn models
    rf_preprocessing_model = RandomForestModel()
    df = rf_preprocessing_model.preprocess_data(train.copy(), is_train=True)
    
    # 데이터 분할
    X = df.drop(columns=['채무 불이행 여부'])
    y = df['채무 불이행 여부']
    
    # 클래스 불균형 해결 (SMOTE 사용)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Train-Test Split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )

    # --- 개별 베이스 모델 학습 및 성능 보고서 출력 ---
    print("\n--- Individual Base Model Performance ---")

    # 1. LightGBM Classifier
    lgb_model = lgb.LGBMClassifier(random_state=42)
    lgb_model.fit(X_train, y_train)
    lgb_preds = lgb_model.predict(X_valid)
    print("\n1. LightGBM Classifier:\n")
    print(classification_report(y_valid, lgb_preds, digits=4))

    # 2. XGBoost Classifier
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_valid)
    print("\n2. XGBoost Classifier:\n")
    print(classification_report(y_valid, xgb_preds, digits=4))

    # 3. CatBoost Classifier
    cat_model = cb.CatBoostClassifier(verbose=0, random_state=42)
    cat_model.fit(X_train, y_train)
    cat_preds = cat_model.predict(X_valid)
    print("\n3. CatBoost Classifier:\n")
    print(classification_report(y_valid, cat_preds, digits=4))

    print("------------------------------------------")

    # --- Stacking 앙상블 구성 및 학습 ---
    base_learners = [
        ('lgb', lgb.LGBMClassifier(random_state=42)),
        ('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)),
        ('cat', cb.CatBoostClassifier(verbose=0, random_state=42))
    ]
    
    meta_learner = LogisticRegression()
    stacking_model = StackingClassifier(
        estimators=base_learners, 
        final_estimator=meta_learner, 
        passthrough=True
    )
    
    # 모델 학습
    stacking_model.fit(X_train, y_train)
    
    # 검증 데이터로 성능 평가
    stacking_preds = stacking_model.predict_proba(X_valid)[:, 1]
    stacking_auc = roc_auc_score(y_valid, stacking_preds)
    stacking_pred_labels = stacking_model.predict(X_valid)
    stacking_acc = accuracy_score(y_valid, stacking_pred_labels)

    # 테스트 데이터 전처리
    test_processed = rf_preprocessing_model.preprocess_data(test.copy(), is_train=False)
    test_processed = test_processed[X_train.columns]
    
    # 최적 모델을 사용하여 예측
    stacking_test_preds = stacking_model.predict_proba(test_processed)[:, 1]
    
    # 제출 파일 생성
    submission = pd.read_csv(submission_path)
    submission.iloc[:, 1] = stacking_test_preds
    submission.to_csv('submission.csv', index=False)
    
    # 결과 출력 (요청된 형식)
    print(f"\nStacking Ensemble ROC-AUC: {stacking_auc:.4f}")
    print(f"Stacking Ensemble Accuracy: {stacking_acc:.4f}")
    print("제출 파일이 생성되었습니다: submission.csv")

if __name__ == "__main__":
    main()
