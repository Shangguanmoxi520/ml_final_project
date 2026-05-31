import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# 1. 讀取資料
train_df = pd.read_csv('home-credit-default-risk/application_train.csv')
test_df = pd.read_csv('home-credit-default-risk/application_test.csv')

# 儲存 TARGET 並分離
y_train = train_df['TARGET']
X_train = train_df.drop(columns=['TARGET', 'SK_ID_CURR'])
X_test = test_df.drop(columns=['SK_ID_CURR'])

# 合併處理特徵
df = pd.concat([X_train, X_test], axis=0)

# 2. 修正異常值 (DAYS_EMPLOYED)
df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

# 3. 類別特徵處理
le = LabelEncoder()
cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    if df[col].nunique() <= 2:
        # 雙值欄位用 Label Encoding
        df[col] = le.fit_transform(df[col].astype(str))
    else:
        # 多值欄位用 One-Hot Encoding
        df = pd.get_dummies(df, columns=[col], drop_first=True)

# ========================================================
# 🚀 4. 引入並處理第二張子表 bureau.csv
# ========================================================
print("正在處理 bureau.csv...")
bureau = pd.read_csv('home-credit-default-risk/bureau.csv')

# (A) 對 bureau 的類別型欄位做 One-Hot Encoding，方便後面計算次數
bureau_cat_cols = bureau.select_dtypes(include=['object']).columns
bureau = pd.get_dummies(bureau, columns=bureau_cat_cols, drop_first=True)

# (B) 定義數值型欄位要做哪些聚合計算
# DAYS_CREDIT (多久前借的), AMT_CREDIT_SUM (借了多少錢) 等都是極重要特徵
num_aggregations = {
    'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
    'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
    'DAYS_CREDIT_UPDATE': ['mean'],
    'CREDIT_DAY_OVERDUE': ['max', 'mean'],
    'AMT_CREDIT_MAX_OVERDUE': ['max'],
    'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
    'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
    'AMT_CREDIT_SUM_LIMIT': ['max', 'mean'],
    'AMT_CREDIT_SUM_OVERDUE': ['mean'],
    'CNT_CREDIT_PROLONG': ['sum']
}

# (C) 對 One-Hot 展開後的類別欄位計算平均值（代表某種貸款類型的佔比）
cat_aggregations = {}
for col in bureau.columns:
    if col not in num_aggregations and col != 'SK_ID_CURR' and col != 'SK_ID_BUREAU':
        cat_aggregations[col] = ['mean']

# 合併所有的聚合規則
bureau_agg_rules = {**num_aggregations, **cat_aggregations}

# (D) 執行 Groupby 聚合
bureau_agg = bureau.groupby('SK_ID_CURR').agg(bureau_agg_rules)

# 重新整理欄位名稱，避免出現多層級 Index (例如變成 DAYS_CREDIT_mean)
bureau_agg.columns = pd.Index(['BUREAU_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])

# (E) 新增一個特徵：該申請人總共在信用局有多少筆過往貸款紀錄
bureau_counts = bureau.groupby('SK_ID_CURR').size().to_frame('BUREAU_LOAN_COUNT')
bureau_agg = bureau_agg.join(bureau_counts, how='left')

# (F) 將聚合後的 bureau 特徵拼回主表 df
df = df.join(bureau_agg, how='left', on=df.index) # 註：因為先前 concat 後 index 就是對應原本的行，直接對齊可能會有問題。保險起見，我們用主表的 SK_ID_CURR 來對接

# 修正對接方式：將 SK_ID_CURR 加回 df 來進行 merge
df['SK_ID_CURR'] = pd.concat([train_df['SK_ID_CURR'], test_df['SK_ID_CURR']], axis=0).values
df = df.merge(bureau_agg, on='SK_ID_CURR', how='left')
df.drop(columns=['SK_ID_CURR'], inplace=True)
# ========================================================

# ========================================================
# 🚀 引入並處理第三張子表 previous_application.csv
# ========================================================
print("正在處理 previous_application.csv...")
prev = pd.read_csv('home-credit-default-risk/previous_application.csv')

# 1. 修正異常值：過去貸款資料中，天數欄位常有 365243 代表 NaN 的狀況
prev_days_cols = ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']
for col in prev_days_cols:
    prev[col].replace(365243, np.nan, inplace=True)

# 2. 類別型欄位做 One-Hot Encoding
prev_cat_cols = prev.select_dtypes(include=['object']).columns
prev = pd.get_dummies(prev, columns=prev_cat_cols, drop_first=True)

# 3. 定義數值型欄位要做哪些聚合計算
prev_num_aggregations = {
    'AMT_ANNUITY': ['max', 'mean'],      # 過去借款的年金大小
    'AMT_APPLICATION': ['max', 'mean'],  # 過去申請的金額
    'AMT_CREDIT': ['max', 'mean'],       # 實際核准的金額
    'AMT_DOWN_PAYMENT': ['max', 'mean'], # 頭期款金額
    'AMT_GOODS_PRICE': ['max', 'mean'],  # 商品價格
    'HOUR_APPR_PROCESS_START': ['mean'],
    'RATE_DOWN_PAYMENT': ['max', 'mean'],
    'DAYS_DECISION': ['min', 'max', 'mean'], # 多久以前申請的
    'CNT_PAYMENT': ['mean', 'sum'],       # 過去貸款的分期期數
}

# 4. 對 One-Hot 展開後的類別欄位計算平均值（例如：過去被拒絕的比例是多少）
prev_cat_aggregations = {}
for col in prev.columns:
    if col not in prev_num_aggregations and col != 'SK_ID_CURR' and col != 'SK_ID_PREV':
        prev_cat_aggregations[col] = ['mean']

# 合併聚合規則
prev_agg_rules = {**prev_num_aggregations, **prev_cat_aggregations}

# 5. 執行 Groupby 聚合
prev_agg = prev.groupby('SK_ID_CURR').agg(prev_agg_rules)

# 重新整理欄位名稱，加上 PREV_ 前綴避免重複
prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

# 6. 新增一個特徵：某人過去總共申請過幾次貸款
prev_counts = prev.groupby('SK_ID_CURR').size().to_frame('PREV_APPLICATION_COUNT')
prev_agg = prev_agg.join(prev_counts, how='left')

# 7. 將聚合後的 prev_agg 特徵拼回主表 df
# 註：因為先前處理 bureau 時我們已經把 SK_ID_CURR 補回 df 了，這裡直接 merge
df['SK_ID_CURR'] = pd.concat([train_df['SK_ID_CURR'], test_df['SK_ID_CURR']], axis=0).values
df = df.merge(prev_agg, on='SK_ID_CURR', how='left')
df.drop(columns=['SK_ID_CURR'], inplace=True)

print("previous_application 處理並合併完成！")


# 重新拆分回訓練集與測試集
X_train = df.iloc[:len(train_df), :]
X_test = df.iloc[len(train_df):, :]

# === 加上這兩行來清洗欄位名稱 ===
import re
X_train = X_train.rename(columns = lambda x: re.sub('[^A-Za-z0-9_]+', '_', x))
X_test = X_test.rename(columns = lambda x: re.sub('[^A-Za-z0-9_]+', '_', x))
# =================================

print(f"訓練集形狀: {X_train.shape}, 測試集形狀: {X_test.shape}")

# 5. 建立 Stratified K-Fold 驗證架構
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(X_train.shape[0])
sub_preds = np.zeros(X_test.shape[0])

# 建立一個陣列來儲存每一折的特徵重要性
feature_importance_df = pd.DataFrame()
feature_importance_df['feature'] = X_train.columns
feature_importance_df['importance'] = 0.0

# LightGBM 參數設定 (Baseline 專用)
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': -1,
    'random_state': 42
}

for fold, (train_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
    
    # 建立 LightGBM 資料集
    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va)
    
    # 訓練模型
    clf = lgb.train(
        params, 
        trn_data, 
        num_boost_round=5000, 
        valid_sets=[trn_data, val_data], 
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
    )
    
    # 預測驗證集與測試集
    oof_preds[val_idx] = clf.predict(X_va, num_iteration=clf.best_iteration)
    sub_preds += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits
    
    # 累加這一折的特徵重要性 (使用 'gain' 代表分裂帶來的訊息增益，比 'split' 更精準)
    feature_importance_df['importance'] += clf.feature_importance(importance_type='gain') / folds.n_splits

    print(f"Fold {fold+1} AUC: {roc_auc_score(y_va, oof_preds[val_idx]):.5f}")

print(f"整體 OOF AUC: {roc_auc_score(y_train, oof_preds):.5f}")
# ========================================================
# 🎯 核心：特徵篩選與降維 (Feature Selection & Reduction)
# ========================================================
print("\n=== 開始進行特徵重要性篩選 ===")

# 排序特徵重要性
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).reset_index(drop=True)

# 計算累積重要性比例
feature_importance_df['cumulative_importance'] = feature_importance_df['importance'].cumsum() / feature_importance_df['importance'].sum()

print("--- 前 10 大核心特徵 (對預測違約最關鍵) ---")
print(feature_importance_df.head(10))

# 找出完全沒貢獻 (Importance = 0) 的特徵數
zero_importance = feature_importance_df[feature_importance_df['importance'] == 0]
print(f"\n總共有 {len(zero_importance)} 個特徵的貢獻度完全為 0。")

# 設定門檻：只保留累積貢獻度達到 99% 的特徵（砍掉最後那 1% 沒貢獻又耗記憶體的特徵）
threshold = 0.99
selected_features = feature_importance_df[feature_importance_df['cumulative_importance'] <= threshold]['feature'].tolist()

# 確保至少把最不重要的部分切掉，若累積剛好壓在線上的也加進來
if len(selected_features) == 0:
    selected_features = feature_importance_df.iloc[:int(len(feature_importance_df)*threshold)]['feature'].tolist()

print(f"🎉 降維成功！特徵欄位數從 {X_train.shape[1]} 個 成功精簡至 {len(selected_features)} 個！")
# 1. 用選出來的特徵過濾訓練集與測試集
X_train_reduced = X_train[selected_features].copy()
X_test_reduced = X_test[selected_features].copy()

# 2. 【強力推薦】在下一輪順便加入「黃金特徵組合」衝分
# 既然三大外部評分和貸款金額最重要，我們直接幫模型做物理外掛：
for df_reduced in [X_train_reduced, X_test_reduced]:
    
    # 三大評分的乘積與平均
    df_reduced['EXT_SOURCES_PROD'] = df_reduced['EXT_SOURCE_1'] * df_reduced['EXT_SOURCE_2'] * df_reduced['EXT_SOURCE_3']
    df_reduced['EXT_SOURCES_MEAN'] = df_reduced[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    
    # 主表的經典金融比例
    # 註：有些欄位如果剛剛沒被選進 selected_features，這裡要確保它還在原本的 X_train 中
    df_reduced['CREDIT_TO_INCOME_RATIO'] = X_train['AMT_CREDIT'] / X_train['AMT_INCOME_TOTAL'] if 'AMT_CREDIT' in X_train else df_reduced['AMT_CREDIT'] / df_reduced['AMT_INCOME_TOTAL']
    df_reduced['ANNUITY_TO_INCOME_RATIO'] = X_train['AMT_ANNUITY'] / X_train['AMT_INCOME_TOTAL']

print(f"下一輪準備就緒！特徵形狀: {X_train_reduced.shape}")

# ========================================================
# 3. 重新跑 5-Fold 交叉驗證 (這次速度會變快很多！)
# ========================================================
oof_preds = np.zeros(X_train_reduced.shape[0])
sub_preds = np.zeros(X_test_reduced.shape[0])

# 這次我們可以把學習率調低 (0.05 -> 0.02)，讓精簡後的特徵學得更細緻
# params['learning_rate'] = 0.02 

for fold, (train_idx, val_idx) in enumerate(folds.split(X_train_reduced, y_train)):
    X_tr, y_tr = X_train_reduced.iloc[train_idx], y_train.iloc[train_idx]
    X_va, y_va = X_train_reduced.iloc[val_idx], y_train.iloc[val_idx]
    
    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va)
    
    clf = lgb.train(
        params, 
        trn_data, 
        num_boost_round=5000, # 學習率調低了，總輪數可以放寬
        valid_sets=[trn_data, val_data], 
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(100)] # early stopping 稍微拉長一點點
    )
    
    oof_preds[val_idx] = clf.predict(X_va, num_iteration=clf.best_iteration)
    sub_preds += clf.predict(X_test_reduced, num_iteration=clf.best_iteration) / folds.n_splits
    
    print(f"Fold {fold+1} AUC: {roc_auc_score(y_va, oof_preds[val_idx]):.5f}")

print(f"下一輪優化後的整體 OOF AUC: {roc_auc_score(y_train, oof_preds):.5f}")


# 儲存預測結果準備上傳 Kaggle
submission = pd.DataFrame({'SK_ID_CURR': test_df['SK_ID_CURR'], 'TARGET': sub_preds})
submission.to_csv('baseline_submission.csv', index=False)