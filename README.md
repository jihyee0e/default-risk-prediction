# 💳 채무 불이행 여부 예측: 불이행의 징후를 찾아라!

## 📁 프로젝트 구조

```
miniproject/
├── main.py              # 메인 실행 파일 (스태킹 앙상블)
├── model_rf.py          # Random Forest 모델 클래스
├── model_lr.py          # Logistic Regression 모델 클래스
├── requirements.txt     # 필요한 패키지 목록
├── data/               # 데이터 파일들
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
└── README.md           # 프로젝트 설명서
```

## 🚀 실행 방법

1. **환경 설정**
   ```bash
   pip install -r requirements.txt
   ```

2. **모델 실행**
   ```bash
   python main.py
   ```

3. **예상 출력**
   ```
   Stacking Ensemble ROC-AUC: 0.6247
   제출 파일이 생성되었습니다: submission.csv
   ```

## 📊 모델 아키텍처

### 스태킹 앙상블 구성
- **Base Learners**: LightGBM, XGBoost, CatBoost, Random Forest, Logistic Regression
- **Meta Learner**: Logistic Regression
- **특징**: Passthrough 활성화로 원본 특성 보존

### 개별 모델 파일
- **`model_rf.py`**: Random Forest 모델 클래스 (특성 중요도 분석 포함)
- **`model_lr.py`**: Logistic Regression 모델 클래스 (스케일링 자동 처리)

## 배경
채무 불이행은 금융 기관과 개인 모두에게 큰 영향을 미치며, 경제 시스템에 부담 UP
<br>효과적으로 채무 불이행 여부를 예측할 수 있는 기술은 금융 안정성을 높이고, 금융 서비스 제공자와 고객 간의 신뢰를 강화하는 데 큰 기여하기

## 주제
채무 불이행 가능성을 예측하는 AI 알고리즘 개발 

## 📊 데이터 특성
- 19개 독립변수 (수치형 15개, 범주형 4개)
- 채무 불이행 여부: 0(정상 상환), 1(채무 불이행)
- 클래스 불균형: 1:1.8 → 소수 클래스(불이행) 예측 정확도가 핵심
<img src="https://github.com/user-attachments/assets/29e3b239-d277-43f5-bf1a-2b3623f2d965" width="800px">

## 데이터 전처리 전략
#### 1. 데이터 정제  
| **단계** | **방법** | **근거** |  
|------|------|------|  
| UID 제거 | UID 컬럼 삭제 | 고유 식별자는 모델 예측력에 기여하지 않으며 과적합 유발 가능성 |  
| 이상치 처리 | IQR 기반 Winsorizing 대신 중앙값 대체 선택 | 트리 기반 모델의 이상치 Robustness 특성 활용 |  

#### 2. 파생변수 공학  
| **파생변수** | **금융적 해석** | **수학적 표현** | **예상 영향력** |  
|----------|------------|--------------|--------------|  
| 신용 점수 대비 부채 비율 | 신용등급 대비 실제 부담률 | `부채 비율 / 신용 점수` | 음의 상관관계 예상 |  
| 연체 리스크 지표 | 과거 이력 기반 미래 위험 예측 | `발생 횟수 × 경과 개월` | 양의 상관관계 예상 |  
| 월 소득 대비 부채 비율 | 현금흐름 안정성 평가 | `월 상환액 / (연소득 / 12)` | 임계값 초과 시 위험 증가 |  
#### 3. 결측치 처리 체계
수치형 변수:
- 중앙값 대체 이유:
    - 로버스트 통계량으로 극단값 영향 최소화
    - `신용 점수` 등의 변수에서 Skewness 관찰
범주형 변수:
- 'Unknown'처리 장점:
    - 신규 범주 출현시 대응 가능
    - 모델 범주 고정으로 안전성 확보 
#### 4. 인코딩 전략 비교
| **방법** | **장점** | **단점** | **선택 이유** |
| --- | --- | --- | --- |
| Label Encoding | 차원 증가 없음 | 순서 의미 부여 가능성 | 트리 모델의 비선형성 대응 |
| One-Hot | 순서 영향 제거 | 차원 폭발 문제 | 대규모 데이터에 비효율적 |
| Target Encoding | 타겟 정보 반영 | 과적합 위험 | 검증 데이터 누수 가능성 |

## 클래스 불균형 대응
<img src="./images/target.png" width="400px">

#### 1. SMOTE 작동 메커니즘

```
graph LR
    A[기존 소수 클래스] --> B[K-NN 기반 이웃 탐색]
    B --> C[임의 보간법으로 합성 샘플 생성]
    C --> D[균형 잡힌 데이터 분포]
```

**적용 세부사항**:

- **k_neighbors=5** (기본값) 유지: 과도한 샘플링 방지
- **분할 전 적용**: 검증 세트에 정보 누수 방지
- **성능 비교**: SMOTE vs ADASYN vs BorderlineSMOTE 실험 계획

#### 2. 대체 기법 비교표

| **방법** | **원리** | **장점** | **단점** |
| --- | --- | --- | --- |
| Undersampling | 다수 클래스 삭제 | 계산 효율 | 정보 손실 |
| Class Weight | 손실 함수 가중치 | 데이터 변형 없음 | 모델 특정 구현 필요 |
| GAN | 생성적 적대 신경망 | 복잡한 분포 학습 | 훈련 불안정성 |


## 앙상블 모델 설계

#### 1. 스태킹 아키텍처

```
graph TD
    A[Raw Features] --> B[LightGBM]
    A --> C[XGBoost]
    A --> D[CatBoost]
    B --> E[Meta Features]
    C --> E
    D --> E
    E --> F[Logistic Regression]
    A -->|Passthrough| F
```

#### 2. 개별 모델 최적화 포인트

**LightGBM**:

- `num_leaves=31`: 복잡도 제어
- `feature_fraction=0.9`: 과적합 방지

**XGBoost**:

- `gamma=0`: 가지치기 완화
- `max_depth=6`: 적절한 복잡도

**CatBoost**:

- `grow_policy='SymmetricTree'`: 균형 트리 구조
- `one_hot_max_size=10`: 자동 범주형 처리

#### 3. 메타 학습기 선택 근거

- **로지스틱 회귀**의 선형 결합 특성:
    - Base Learner 예측값의 가중 합 최적화
    - 저차원 데이터에서 고성능
- **Passthrough 활성화**: 원본 특성 보존으로 메타 모델의 설명력 향상


## 검증 방법

#### 1. 계층화 분할 프로세스

```
X_train, X_valid, y_train, y_valid = train_test_split(
    X_resampled, y_resampled,
    test_size=0.2,
    stratify=y_resampled,  # SMOTE 적용 후 분포 반영
    random_state=42
)
```

#### 2. 평가 지표 분석

**ROC-AUC 선택 이유**:

- 임계값 불문 전반적 성능 평가
- Precision-Recall 대비 불균형 데이터에 Robust
- 금융 리스크 관리자의 의사결정 유연성 지원


## 실행 결과

- **검증 AUC**: 0.6247


## 개선 가능 방향

#### 1. 하이퍼파라미터 튜닝

| **도구** | **장점** | **적용 계획** |
| --- | --- | --- |
| Optuna | Bayesian 최적화 | 100회 Trial + 5-Fold CV |
| Hyperopt | Tree 구조 탐색 | 모델별 개별 튜닝 |

#### 2. 심층 특성 공학

- **상호작용 항**: `신용 점수 × 연체 리스크`
- **비선형 변환**: `log(연간 소득 + 1)`
- **시간 가중치**: 최근 6개월 데이터 강조

#### 3. 모델 확장

- **Deep Learning**: Wide & Deep 아키텍처
- **확률 보정**: Platt Scaling 적용
- **다단계 앙상블**: 1단계에서 Hard Sample 선별 후 2단계 집중 학습


## 결론

본 프로젝트는 **스태킹 앙상블**을 중심으로 한 종합적 신용위험 모델링 프레임워크를 제시했습니다. 데이터 전처리 단계에서 도메인 지식을 반영한 파생변수 생성, SMOTE를 활용한 불균형 대응, 다양한 트리 모델의 예측력을 결합한 앙상블 기법이 핵심 성공 요인입니다. 향후 하이퍼파라미터 튜닝과 심층적 특성 분석을 통해 AUC 0.9+ 달성이 기대됩니다.


### 📚 분석 및 시각화
<img src="./images/heatmap.png" width="600px">
<img src="./images/roc_curve.png" width="600px">


### ⚒️ Libraries & Tools
![pandas](https://img.shields.io/badge/pandas-150458.svg?&style=for-the-badge&logo=pandas&logoColor=white)
![numpy](https://img.shields.io/badge/numpy-013243.svg?&style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikitlearn-F7931E.svg?&style=for-the-badge&logo=scikitlearn&logoColor=white)
![xgboost](https://img.shields.io/badge/xgboost-3CC131.svg?&style=for-the-badge&logo=xgboost&logoColor=white)
![lightgbm](https://img.shields.io/badge/lightgbm-9C4A2D.svg?&style=for-the-badge&logo=lightgbm&logoColor=white)
![catboost](https://img.shields.io/badge/catboost-00A3FF.svg?&style=for-the-badge&logo=catboost&logoColor=white)
![optuna](https://img.shields.io/badge/optuna-4A90E2.svg?&style=for-the-badge&logo=optuna&logoColor=white)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-00A1E4.svg?&style=for-the-badge&logo=python&logoColor=white)
<br>
![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00.svg?&style=for-the-badge&logo=googlecolab&logoColor=white)
![GitHub](https://img.shields.io/badge/github-181717.svg?&style=for-the-badge&logo=github&logoColor=white)

---

### 🔧 트러블슈팅 회고
본 프로젝트는 데이콘 채무 불이행 예측 해커톤에 참여하기 위해 진행된 모델 개발 과정에서 여러 가지 문제와 어려움을 겪었습니다. 아래는 주요 이슈와 그에 대한 해결 시도, 그리고 추가 개선 사항을 정리한 내용입니다.

#### 1. 데이터 전처리 및 특성 엔지니어링 🔍

#### 문제점 ⚠️
- **결측치 및 무한대 값 처리:**  
  원본 데이터에 결측치와 무한대 값이 다수 포함되어 있어 중앙값 대체 및 `np.nan` 처리가 필요했습니다.
  
- **파생 변수 생성:**  
  "부채 비율", "신용 점수 대비 부채 비율", "연체 리스크 지표" 등 파생 변수를 생성하는 과정에서 분모가 0인 경우에 대비하지 않으면 오류가 발생할 수 있었습니다.
  
- **범주형 변수 인코딩:**  
  초기에는 LabelEncoder를 사용하여 범주형 변수를 숫자형으로 변환하였으나, 데이터의 복잡한 관계를 충분히 반영하지 못할 수 있어 이후 One-Hot 인코딩, Target Encoding, Frequency Encoding 등 다양한 기법을 실험했습니다.
  
- **이상치 처리:**  
  처음에는 1%~99% 분위수 기반 클리핑을 사용하였으나, IQR(Interquartile Range) 기반 이상치 처리를 도입해 극단치의 영향을 줄이려 시도했습니다.

#### 해결 및 개선 방안 ✅
- 결측치와 무한대 값은 `replace()`와 중앙값 대체로 안정적으로 처리하였고, 파생 변수 생성 시 분모가 0인 경우 `np.nan`으로 처리하여 오류를 방지했습니다.
- 범주형 변수는 'Unknown'으로 채운 후 LabelEncoder 또는 One-Hot 인코딩을 적용하여 모든 피처가 수치형으로 변환되도록 했습니다.
- 이상치 처리는 IQR 기반 클리핑을 도입해 극단치로 인한 왜곡을 효과적으로 줄였습니다.


#### 2. 모델 선택 및 복잡도 문제 🧩

#### 문제점 ⚠️
- **모델 복잡도와 오버피팅:**  
  복잡한 스태킹 앙상블 모델(예: LightGBM, XGBoost, CatBoost 결합)을 사용했을 때, 학습 데이터에 과도하게 맞추어져 검증 및 테스트 데이터에서 과적합이 발생하여 점수가 낮아졌습니다.
  
- **하이퍼파라미터 튜닝:**  
  GridSearchCV와 Optuna 등 다양한 도구를 사용하여 하이퍼파라미터를 튜닝하려 했지만, 전처리 파이프라인과 모델 간의 상호작용으로 인해 최적의 조합을 찾기 어려웠습니다.

#### 해결 및 개선 방안 ✅
- 모델 복잡도를 낮추기 위해 단일 결정트리나 랜덤 포레스트 모델과 같이 단순 모델로 전환하거나, 불필요한 피처를 제거 및 차원 축소(PCA, 상호작용 특성 등)를 적용하여 과적합을 줄였습니다.
- GridSearchCV를 통해 결정트리와 랜덤 포레스트의 주요 하이퍼파라미터(max_depth, min_samples_split, min_samples_leaf 등)를 조정하고, 교차검증(cv=5)을 통해 모델의 일반화 성능을 평가했습니다.
- 예측 확률 캘리브레이션(Platt Scaling, Isotonic Regression) 기법을 추가로 검토 중입니다.

#### 3. 클래스 불균형 및 SMOTE 적용 문제 ⚖️

#### 문제점 ⚠️
- **클래스 불균형:**  
  데이터의 클래스 불균형 문제를 해결하기 위해 SMOTE를 적용했으나, 생성된 합성 데이터가 실제 데이터 분포와 차이가 있어 모델의 일반화 성능에 영향을 미칠 수 있었습니다.

#### 해결 및 개선 방안 ✅
- SMOTE의 `sampling_strategy` 및 오버샘플링 비율을 조정하여, 합성 데이터가 원본 데이터 분포와 유사하게 유지되도록 노력했습니다.
- SMOTE 외에도 ADASYN 또는 언더샘플링 기법을 함께 실험하여 다양한 불균형 처리 방식을 평가했습니다.
- 교차검증을 통해 SMOTE 적용 후 모델의 성능 안정성을 지속적으로 확인했습니다.




