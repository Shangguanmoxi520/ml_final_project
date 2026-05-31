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
    
    print(f"Fold {fold+1} AUC: {roc_auc_score(y_va, oof_preds[val_idx]):.5f}")

print(f"整體 OOF AUC: {roc_auc_score(y_train, oof_preds):.5f}")

# 儲存預測結果準備上傳 Kaggle
submission = pd.DataFrame({'SK_ID_CURR': test_df['SK_ID_CURR'], 'TARGET': sub_preds})
submission.to_csv('baseline_submission.csv', index=False)