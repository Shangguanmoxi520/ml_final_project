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