import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 1. 데이터 로드 (경로 에러 방지를 위해 슬래시 통일)
df_m = pd.read_csv('data/MRegularSeasonDetailedResults.csv')
df_w = pd.read_csv('data/WRegularSeasonDetailedResults.csv')
sub = pd.read_csv('data/SampleSubmissionStage1.csv')

# 2. 스탯 계산 및 학습 데이터 생성 함수 (기존과 동일)
def get_season_stats(df):
    w_stats = df.groupby(['Season', 'WTeamID']).agg({'WScore':'mean', 'WFGM':'sum', 'WFGA':'sum', 'WAst':'mean', 'WTO':'mean', 'WOR':'mean', 'WDR':'mean'}).reset_index()
    l_stats = df.groupby(['Season', 'LTeamID']).agg({'LScore':'mean', 'LFGM':'sum', 'LFGA':'sum', 'LAst':'mean', 'LTO':'mean', 'LOR':'mean', 'LDR':'mean'}).reset_index()
    w_stats.columns = ['Season', 'TeamID', 'Score', 'FGM', 'FGA', 'Ast', 'TO', 'OR', 'DR']
    l_stats.columns = ['Season', 'TeamID', 'Score', 'FGM', 'FGA', 'Ast', 'TO', 'OR', 'DR']
    stats = pd.concat([w_stats, l_stats]).groupby(['Season', 'TeamID']).mean().reset_index()
    stats['FG_Pct'] = stats['FGM'] / stats['FGA']
    stats['Total_Reb'] = stats['OR'] + stats['DR']
    return stats[['Season', 'TeamID', 'Score', 'FG_Pct', 'Ast', 'TO', 'Total_Reb']]

def make_train_data(df, stats):
    tmp = df[['Season', 'WTeamID', 'LTeamID']].copy()
    tmp = tmp.merge(stats, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
    tmp = tmp.merge(stats, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left', suffixes=('_W', '_L'))
    features = ['Score', 'FG_Pct', 'Ast', 'TO', 'Total_Reb']
    for f in features:
        tmp[f+'_Diff'] = tmp[f+'_W'] - tmp[f+'_L']
    diff_cols = [f+'_Diff' for f in features]
    X_win = tmp[diff_cols].copy(); X_win['Result'] = 1
    X_lose = -tmp[diff_cols].copy(); X_lose['Result'] = 0
    final_train = pd.concat([X_win, X_lose]).reset_index(drop=True)
    return final_train.drop('Result', axis=1), final_train['Result']

# 3. 남녀 모델 각각 학습
stats_m = get_season_stats(df_m)
X_m, y_m = make_train_data(df_m, stats_m)
scaler_m = StandardScaler(); X_m_s = scaler_m.fit_transform(X_m)
model_m = LogisticRegression().fit(X_m_s, y_m)

stats_w = get_season_stats(df_w)
X_w, y_w = make_train_data(df_w, stats_w)
scaler_w = StandardScaler(); X_w_s = scaler_w.fit_transform(X_w)
model_w = LogisticRegression().fit(X_w_s, y_w)

# 4. [핵심 수정] 제출 데이터 분리 및 일괄 예측 (병합 방식)
sub_parts = sub['ID'].str.split('_', expand=True).astype(int)
sub_parts.columns = ['Season', 'Team1', 'Team2']
sub_parts['ID'] = sub['ID']

# 남성/여성 데이터 분리 (Team1 ID 기준)
sub_m = sub_parts[sub_parts['Team1'] < 3000].copy()
sub_w = sub_parts[sub_parts['Team1'] >= 3000].copy()

def get_predictions(target_df, stats, scaler, model):
    if target_df.empty: return pd.Series()
    # 스탯 병합
    res = target_df.merge(stats, left_on=['Season', 'Team1'], right_on=['Season', 'TeamID'], how='left')
    res = res.merge(stats, left_on=['Season', 'Team2'], right_on=['Season', 'TeamID'], how='left', suffixes=('_1', '_2'))
    
    # 차이 계산
    features = ['Score', 'FG_Pct', 'Ast', 'TO', 'Total_Reb']
    for f in features:
        res[f+'_Diff'] = res[f+'_1'] - res[f+'_2']
    
    X_target = res[[f+'_Diff' for f in features]].fillna(0)
    X_target_s = scaler.transform(X_target)
    return model.predict_proba(X_target_s)[:, 1]

# 각 성별에 맞는 예측값 계산
preds_m = get_predictions(sub_m, stats_m, scaler_m, model_m)
preds_w = get_predictions(sub_w, stats_w, scaler_w, model_w)

# 결과 합치기
sub_m['Pred'] = preds_m
sub_w['Pred'] = preds_w
final_sub = pd.concat([sub_m[['ID', 'Pred']], sub_w[['ID', 'Pred']]]).sort_index()

# 최종 저장
final_sub.to_csv('submission_final_fixed.csv', index=False)
print("✅ 에러 수정 완료! 파일이 생성되었습니다.")