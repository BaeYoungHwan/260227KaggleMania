import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

# 1. 설정 및 데이터 로드
DATA_PATH = "./"  # 파일이 있는 경로

def prepare_data(gender):
    print(f"--- Processing {gender} Data ---")
    
    # 파일 읽기
    reg_season = pd.read_csv(f"{DATA_PATH}{gender}RegularSeasonDetailedResults.csv")
    tourney_results = pd.read_csv(f"{DATA_PATH}{gender}NCAATourneyDetailedResults.csv")
    seeds = pd.read_csv(f"{DATA_PATH}{gender}MNCAATourneySeeds.csv" if gender == 'M' else f"{DATA_PATH}{gender}WNCAATourneySeeds.csv")
    
    # 시드 데이터 전처리 (예: W01 -> 1)
    seeds['Seed'] = seeds['Seed'].apply(lambda x: int(x[1:3]))
    
    # --- Feature Engineering: 정규 시즌 통계 ---
    # 이긴 팀 통계
    w_stats = reg_season.groupby(['Season', 'WTeamID']).agg(
        WScore_avg=('WScore', 'mean'),
        WFGM_avg=('WFGM', 'mean'),
        WStl_avg=('WStl', 'mean'),
        WTO_avg=('WTO', 'mean'),
        WCount=('WScore', 'count')
    ).reset_index().rename(columns={'WTeamID': 'TeamID'})

    # 진 팀 통계
    l_stats = reg_season.groupby(['Season', 'LTeamID']).agg(
        LScore_avg=('LScore', 'mean'),
        LFGM_avg=('LFGM', 'mean'),
        LStl_avg=('LStl', 'mean'),
        LTO_avg=('LTO', 'mean'),
        LCount=('LScore', 'count')
    ).reset_index().rename(columns={'LTeamID': 'TeamID'})

    # 두 통계 병합하여 시즌별 팀 능력치 산출
    stats = pd.merge(w_stats, l_stats, on=['Season', 'TeamID'], how='outer').fillna(0)
    stats['WinRate'] = stats['WCount'] / (stats['WCount'] + stats['LCount'])
    stats['AvgScore'] = (stats['WScore_avg'] * stats['WCount'] + stats['LScore_avg'] * stats['LCount']) / (stats['WCount'] + stats['LCount'])
    
    # 필요한 특징만 선택
    season_features = stats[['Season', 'TeamID', 'WinRate', 'AvgScore']]
    season_features = pd.merge(season_features, seeds, on=['Season', 'TeamID'], how='left').fillna(20) # 시드 없는 팀은 하위 시드 부여
    
    return season_features, tourney_results

# 데이터 준비
m_features, m_tourney = prepare_data('M')
w_features, w_tourney = prepare_data('W')

# 2. 학습 데이터셋 생성 함수
def create_train_set(tourney_results, features):
    train_data = []
    for _, row in tourney_results.iterrows():
        season = row['Season']
        t1 = row['WTeamID']
        t2 = row['LTeamID']
        
        # Team 1(Win)과 Team 2(Loss)의 특징 가져오기
        feat1 = features[(features['Season'] == season) & (features['TeamID'] == t1)].iloc[0, 2:].values
        feat2 = features[(features['Season'] == season) & (features['TeamID'] == t2)].iloc[0, 2:].values
        
        # 특징 차이 계산 및 레이블 1 (T1 승리)
        train_data.append(np.append(feat1 - feat2, 1))
        
        # 데이터 증강 (역순으로 레이블 0 생성)
        train_data.append(np.append(feat2 - feat1, 0))
        
    return pd.DataFrame(train_data)

train_df_m = create_train_set(m_tourney, m_features)
train_df_w = create_train_set(w_tourney, w_features)
train_df = pd.concat([train_df_m, train_df_w]) # 남녀 데이터 통합 학습 또는 개별 학습 선택 가능

X = train_df.iloc[:, :-1]
y = train_df.iloc[:, -1]

# 3. 모델 학습 (XGBoost)
model = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=3, eval_metric='logloss')
model.fit(X, y)

# 4. 2026 예측 및 Submission 생성
sub = pd.read_csv(f"{DATA_PATH}SampleSubmissionStage2.csv")
predictions = []

# 통합된 특징 데이터 (M + W)
all_features = pd.concat([m_features, w_features])

for _, row in sub.iterrows():
    season, t1, t2 = map(int, row['ID'].split('_'))
    
    try:
        feat1 = all_features[(all_features['Season'] == season) & (all_features['TeamID'] == t1)].iloc[0, 2:].values
        feat2 = all_features[(all_features['Season'] == season) & (all_features['TeamID'] == t2)].iloc[0, 2:].values
        diff = (feat1 - feat2).reshape(1, -1)
        
        prob = model.predict_proba(diff)[0][1] # 승리 확률
        predictions.append(prob)
    except IndexError:
        # 데이터가 없는 경우 0.5(무승부)로 처리
        predictions.append(0.5)

sub['Pred'] = predictions
sub.to_csv("submission_2026.csv", index=False)
print("Success: submission_2026.csv saved.")