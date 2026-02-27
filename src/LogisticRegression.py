import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler

# 1. 데이터 불러오기
df = pd.read_csv('data\MRegularSeasonDetailedResults.csv')


# 팀별 시즌 평균 데이터 계산
def get_season_stats(df):
    # 승리팀과 패배팀 데이터를 각각 정리 후 합치기
    w_stats = df.groupby(['Season', 'WTeamID']).agg({'WScore':'mean', 'WFGM':'sum', 'WFGA':'sum', 'WAst':'mean', 'WTO':'mean', 'WOR':'mean', 'WDR':'mean'}).reset_index()
    l_stats = df.groupby(['Season', 'LTeamID']).agg({'LScore':'mean', 'LFGM':'sum', 'LFGA':'sum', 'LAst':'mean', 'LTO':'mean', 'LOR':'mean', 'LDR':'mean'}).reset_index()
    
    # 컬럼명 통일 (W/L 제거)
    w_stats.columns = ['Season', 'TeamID', 'Score', 'FGM', 'FGA', 'Ast', 'TO', 'OR', 'DR']
    l_stats.columns = ['Season', 'TeamID', 'Score', 'FGM', 'FGA', 'Ast', 'TO', 'OR', 'DR']
    
    stats = pd.concat([w_stats, l_stats]).groupby(['Season', 'TeamID']).mean().reset_index()
    
    # 5대 핵심 지표 생성
    stats['FG_Pct'] = stats['FGM'] / stats['FGA']
    stats['Total_Reb'] = stats['OR'] + stats['DR']
    return stats[['Season', 'TeamID', 'Score', 'FG_Pct', 'Ast', 'TO', 'Total_Reb']]

# 학습용 데이터셋 구축 (차이값 계산)
def make_train_data(df, stats):
    tmp = df[['Season', 'WTeamID', 'LTeamID']].copy()
    tmp = tmp.merge(stats, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
    tmp = tmp.merge(stats, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left', suffixes=('_W', '_L'))
    
    features = ['Score', 'FG_Pct', 'Ast', 'TO', 'Total_Reb']
    for f in features:
        tmp[f+'_Diff'] = tmp[f+'_W'] - tmp[f+'_L']
    
    # Win=1, Loss=0 데이터 생성
    diff_cols = [f+'_Diff' for f in features]
    X_win = tmp[diff_cols].copy(); X_win['Result'] = 1
    X_lose = -tmp[diff_cols].copy(); X_lose['Result'] = 0
    
    return pd.concat([X_win, X_lose]).reset_index(drop=True)

season_stats = get_season_stats(df)

train_data = make_train_data(df, season_stats)
X = train_data.drop('Result', axis=1)
y = train_data['Result']

# 모델 학습
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LogisticRegression()
model.fit(X_scaled, y)

sub = pd.read_csv('data\SampleSubmissionStage1.csv')
sub_parts = sub['ID'].str.split('_', expand=True).astype(int)
sub_parts.columns = ['Season', 'TeamID1', 'TeamID2']

# 스탯 합치기 및 차이 계산
sub_parts = sub_parts.merge(season_stats, left_on=['Season', 'TeamID1'], right_on=['Season', 'TeamID'], how='left')
sub_parts = sub_parts.merge(season_stats, left_on=['Season', 'TeamID2'], right_on=['Season', 'TeamID'], how='left', suffixes=('_1', '_2'))

for f in ['Score', 'FG_Pct', 'Ast', 'TO', 'Total_Reb']:
    sub_parts[f+'_Diff'] = sub_parts[f+'_1'] - sub_parts[f+'_2']

# 예측 및 저장
X_sub = scaler.transform(sub_parts[[f+'_Diff' for f in ['Score', 'FG_Pct', 'Ast', 'TO', 'Total_Reb']]].fillna(0))
sub['Pred'] = model.predict_proba(X_sub)[:, 1]
sub.to_csv('submission_final.csv', index=False)