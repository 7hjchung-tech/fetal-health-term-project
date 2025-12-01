# fetal-health-term-project
Term Project for Computer Programming (CAS1100)

# CTG 기반 태아 건강 예측 vs cfDNA-NIPT 성능 비교 프로젝트

이 저장소는  
1) **CTG(카디오토코그래피) 기반 태아 건강 예측 모델**,  
2) **cfDNA-NIPT(비침습 산전검사) 성능 메타 요약**  

을 함께 다루는 기말 프로젝트의 분석 코드 및 결과 파일을 정리한 것입니다.

---

## 1. 파일 구조

루트 디렉토리의 주요 파일은 다음과 같습니다.

### 1.1 데이터 원본 및 가공 결과

- **`fetal_health.csv`**
  - Kaggle *Fetal Health Classification* 데이터셋 원본.
  - 2,126건의 CTG(카디오토코그램)에서 추출된 21개 특성과 `fetal_health`(정상/의심/병적, 1/2/3) 라벨이 포함됨.

- **`NIPT-cfDNA_자연어문서데이터.csv`**
  - 7편의 cfDNA-NIPT 논문에서 추출한 성능지표를 표준화해 정리한 메타데이터 테이블.
  - 주요 컬럼:
    - `study_id, condition (T21/T18/T13/SCA 등), method (cfDNA, cfDNA_Illumina, cfDNA_Proton, standard_screening 등)`
    - `subgroup (all, low_risk, 1st_trimester 등)`
    - `N_total, TP, FP, TN, FN`
    - `sens, spec, ppv, npv` 및 95% CI, `lr_pos, lr_neg`, `incidence` 등.

- **`performance_ctg_binary.csv`**
  - CTG 기반 **이분류**(고위험 vs 정상) 모델의 성능 요약.
  - 포함 모델:
    - `logistic_multinomial`
    - `xgboost_multiclass` (고위험 vs 정상으로 라벨 재구성)
  - 주요 지표:
    - `accuracy, f1, sensitivity, specificity, ppv, npv, prevalence, auc`.

- **`performance_ctg_multiclass.csv`**
  - CTG 기반 **다중분류**(정상/의심/병적, 3-class) 모델 성능 요약.
  - 포함 모델:
    - `logistic_multinomial`
    - `xgboost_multiclass`
  - 주요 지표:
    - 전체 수준: `accuracy, macro_precision, macro_recall, macro_f1, weighted_f1`
    - 각 클래스별: `precision_cls1/2/3, recall_cls1/2/3, f1_cls1/2/3`.

- **`logistic_feature_importance_long.csv`**
  - 다중분류 로지스틱 회귀 모델의 계수(피처 중요도)를 long-format으로 정리한 파일.
  - 컬럼:
    - `class` (예측 클래스: 1=정상, 2=의심, 3=병적),
    - `feature` (CTG 파생 변수 이름),
    - `coef` (로그 오즈 계수),
    - `odds_ratio` (exp(coef)),
    - `abs_coef` (절댓값 계수: 중요도 정렬용).

### 1.2 문서

- **`기말프로젝트 제안서.pdf`**
  - 프로젝트 배경, 연구 질문, 간단한 방법론을 기술한 초기 제안서.
  - README와 함께 전체 프로젝트의 맥락을 이해하는 참고 문서로 사용.

- **PNG 이미지들 (`*.png`)**
  - 각 NIPT 논문의 원본 표 캡처.
  - `NIPT-cfDNA_자연어문서데이터.csv`를 만들 때 참고한 테이블 스크린샷으로, 재확인용 원자료 역할.

---

## 2. 분석 개요

### 2.1 CTG 기반 태아 건강 예측

1. **데이터**
   - `fetal_health.csv`를 사용.
   - 타깃: `fetal_health` (1 = Normal, 2 = Suspect, 3 = Pathological).

2. **이분류(high-risk vs normal)**
   - Suspect(2) + Pathological(3)을 묶어 **고위험군**으로 정의.
   - 모델:
     - 로지스틱 회귀(`logistic_multinomial`)
     - XGBoost(`xgboost_multiclass`)를 high-risk vs normal로 재라벨링해 학습.
   - 성능 요약: `performance_ctg_binary.csv`
     - 두 모델 모두 AUC 0.98 이상, 정확도 0.93–0.97 수준에서 높은 민감도·특이도 달성.

3. **다중분류(3-class)**
   - Normal / Suspect / Pathological 3개 클래스를 직접 예측.
   - 성능 요약: `performance_ctg_multiclass.csv`
     - XGBoost 모델: 정확도 ~0.96, macro F1 > 0.92 수준.
     - 각 클래스별 F1도 함께 기록되어 있어, “의심/병적” 클래스에서의 재현율 저하 여부 평가 가능.

4. **특징 중요도**
   - 로지스틱 회귀의 계수 정보를 `logistic_feature_importance_long.csv`로 저장.
   - `abs_coef` 기준으로 정렬하면,  
     - 예: `abnormal_short_term_variability`, `prolongued_decelerations` 등  
       CTG에서 임상적으로도 중요한 지표들이 상위에 위치하는지 확인 가능.

### 2.2 cfDNA-NIPT 성능 메타 요약

1. **데이터 구성**
   - `NIPT-cfDNA_자연어문서데이터.csv`는 7편의 cfDNA-NIPT 논문에서  
     **T21, T18, T13, SCA 및 일부 기타 이상(RAA, CNV)**에 대한
     - 민감도, 특이도, PPV, NPV,
     - 95% 신뢰구간,
     - 일부 논문의 양성/음성 우도비, 유병률(incidence)
     를 공통 스키마로 통합한 표입니다.

2. **스키마**
   - 최소 컬럼:
     - `study_id, condition, method, subgroup, N_total, TP, FP, TN, FN`
     - `sens, spec, ppv, npv`, `sens_ci_low/high`, `spec_ci_low/high`, `ppv_ci_low/high`, `npv_ci_low/high`
     - 논문에 있을 경우: `lr_pos, lr_neg, incidence`.
   - 각 논문의 서브그룹(예: 저위험군, 1분기/2분기, 플랫폼별(Illumina vs Proton))을 분리해 입력.

3. **활용 방향**
   - `condition + method (+ subgroup)` 기준으로 groupby하여
     - cfDNA-NIPT의 평균/중앙 민감도·특이도·PPV/NPV 요약
     - 플랫폼별, 임신 분기별, 위험군별 성능 차이 비교
   - CTG 기반 모델 성능(`performance_ctg_*.csv`)과 정성적으로 비교하여
     - “임상 스크리닝 vs 유전 기반 스크리닝”의 상대적 강점·약점 논의에 사용.

---

## 3. 사용 방법(예시)

```python
import pandas as pd

# CTG 원본 데이터
ctg = pd.read_csv("fetal_health.csv")

# CTG 이분류 / 다중분류 성능
perf_bin = pd.read_csv("performance_ctg_binary.csv")
perf_multi = pd.read_csv("performance_ctg_multiclass.csv")

# 로지스틱 회귀 피처 중요도
feat_long = pd.read_csv("logistic_feature_importance_long.csv")

# cfDNA NIPT 메타 데이터
nipt = pd.read_csv("NIPT-cfDNA_자연어문서데이터.csv")

# 예: T21에 대한 cfDNA 민감도 분포 확인



## Reference (NIPT-cfDNA)

Norton 2015 (NEJM)
Norton ME, Jacobsson B, Swamy GK, Laurent LC, Ranzini AC, Brar H, et al. Cell-free DNA analysis for noninvasive examination of trisomy. N Engl J Med. 2015;372(17):1589-1597. doi:10.1056/NEJMoa1407349.

Zhang H 2015 (Ultrasound Obstet Gynecol)
Zhang H, Gao Y, Jiang F, Fu M, Yuan Y, Guo Y, et al. Non-invasive prenatal testing for trisomies 21, 18 and 13: clinical experience from 146,958 pregnancies. Ultrasound Obstet Gynecol. 2015;45(5):530-538. doi:10.1002/uog.14792.

Lee 2019 (JKMS)
Lee DE, Kim H, Park J, Yun T, Park DY, Kim M, et al. Clinical validation of non-invasive prenatal testing for fetal common aneuploidies in 1,055 Korean pregnant women: a single center experience. J Korean Med Sci. 2019;34(24):e172. doi:10.3346/jkms.2019.34.e172.

Xue 2019 (Mol Cytogenet)
Xue Y, Zhao G, Li H, Zhang Q, Lu J, Yu B, et al. Non-invasive prenatal testing to detect chromosome aneuploidies in 57,204 pregnancies. Mol Cytogenet. 2019;12:29. doi:10.1186/s13039-019-0441-5.

La Verde 2021 (BMC Med Genomics)
La Verde M, De Falco L, Torella A, Savarese G, Savarese P, Ruggiero R, et al. Performance of cell-free DNA sequencing-based non-invasive prenatal testing: experience on 36,456 singleton and multiple pregnancies. BMC Med Genomics. 2021;14(1):93. doi:10.1186/s12920-021-00941-y.

Zhang Y 2022 (Front Genet)
Zhang Y, Xu H, Zhang W, Liu K. Non-invasive prenatal testing for the detection of trisomy 13, 18, and 21 and sex chromosome aneuploidies in 68,763 cases. Front Genet. 2022;13:864076. doi:10.3389/fgene.2022.864076.

Zhen 2025 (PLOS ONE)
Zhen J, Zhang L, Wang H, Chen X, Wang W, Zhang Q, et al. Clinical experience of genome-wide non-invasive prenatal testing as a first-tier screening test in a cohort of 59,771 pregnancies. PLOS One. 2025;20(8):e0329463. doi:10.1371/journal.pone.0329463.
t21_cfDNA = nipt[(nipt["condition"] == "T21") & (nipt["method"].str.contains("cfDNA"))]
print(t21_cfDNA[["study_id","method","sens","spec","ppv","npv"]])
