import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

from scipy.optimize import minimize
import joblib

print("🚀 正在直接載入黃金特徵矩陣...")
# 一秒載入，完全不需要重跑任何特徵工程！
features_data = joblib.load('reduced_features_dataset.pkl')

X_train_reduced = features_data['X_train_reduced']
X_test_reduced = features_data['X_test_reduced']
y_train = features_data['y_train']

print(f"✅ 載入成功！訓練集形狀: {X_train_reduced.shape}, 測試集形狀: {X_test_reduced.shape}")

# ========================================================
# 🎯 接下來直接無痛接你的 10-Fold 交叉驗證與三模型 GPU 訓練
# ========================================================
#print("🔥 啟動 10-Fold 終極大賽...")
#kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# ... 往下直接放你原本的 LightGBM、XGBoost、CatBoost 迴圈代碼與 Optuna 融合 ...
print(f"下一輪準備就緒！特徵形狀: {X_train_reduced.shape}")

# 1. 初始化儲存空間
# 儲存驗證集 OOF 預測
lgb_oof = np.zeros(X_train_reduced.shape[0])
xgb_oof = np.zeros(X_train_reduced.shape[0])
cat_oof = np.zeros(X_train_reduced.shape[0])

# 儲存測試集預測
lgb_preds = np.zeros(X_test_reduced.shape[0])
xgb_preds = np.zeros(X_test_reduced.shape[0])
cat_preds = np.zeros(X_test_reduced.shape[0])

# LightGBM 參數設定 (Baseline 專用)
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.07,
    'num_leaves': 31,
    'max_depth': -1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': -1,
    'random_state': 42,
}

# 2. 定義各模型的參數
lgb_params = params # 延用你原本調整好的 LightGBM 參數

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.07,
    'max_depth': 5,
    'min_child_weight': 30,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'random_state': 42,
    'device': 'cuda',
    'tree_method': 'hist' # 使用直方圖加速，速度才跟得上 LightGBM
}

# 5. 建立 Stratified K-Fold 驗證架構
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(X_train_reduced.shape[0])
sub_preds = np.zeros(X_test_reduced.shape[0])

# 建立一個陣列來儲存每一折的特徵重要性
feature_importance_df = pd.DataFrame()
feature_importance_df['feature'] = X_train_reduced.columns
feature_importance_df['importance'] = 0.0

# ========================================================
# 🚀 3. 開始 5-Fold 三模型聯合訓練
# ========================================================
for fold, (train_idx, val_idx) in enumerate(folds.split(X_train_reduced, y_train)):
    print(f"\n========== 🔥 正在訓練 FOLD {fold+1} ========== ")
    X_tr, y_tr = X_train_reduced.iloc[train_idx], y_train.iloc[train_idx]
    X_va, y_va = X_train_reduced.iloc[val_idx], y_train.iloc[val_idx]
    
    # ------------------ (Model 1) LightGBM ------------------
    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va)
    clf_lgb = lgb.train(lgb_params, trn_data, num_boost_round=5000, 
                        valid_sets=[trn_data, val_data], 
                        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)]) # 設0不印Log，畫面較乾淨
    
    lgb_oof[val_idx] = clf_lgb.predict(X_va, num_iteration=clf_lgb.best_iteration)
    lgb_preds += clf_lgb.predict(X_test_reduced, num_iteration=clf_lgb.best_iteration) / folds.n_splits
    print(f"-> LightGBM AUC: {roc_auc_score(y_va, lgb_oof[val_idx]):.5f}")
    
    # ------------------ (Model 2) XGBoost ------------------
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_va, label=y_va)
    dtest = xgb.DMatrix(X_test_reduced)
    
    clf_xgb = xgb.train(xgb_params, dtrain, num_boost_round=5000,
                        evals=[(dtrain, 'train'), (dval, 'val')],
                        early_stopping_rounds=150, verbose_eval=False)
    
    xgb_oof[val_idx] = clf_xgb.predict(dval, iteration_range=(0, clf_xgb.best_iteration + 1))
    xgb_preds += clf_xgb.predict(dtest, iteration_range=(0, clf_xgb.best_iteration + 1)) / folds.n_splits
    print(f"-> XGBoost  AUC: {roc_auc_score(y_va, xgb_oof[val_idx]):.5f}")
    
    # ------------------ (Model 3) CatBoost ------------------
    # CatBoost 的 Pool 格式
    train_pool = Pool(X_tr, y_tr)
    val_pool = Pool(X_va, y_va)
    
    clf_cat = CatBoostClassifier(
        iterations=4000, learning_rate=0.07,
        depth=5, # 🔥 關鍵：將樹深從 5 放寬到 6，完美匹配我們新做出的交叉特徵！
        eval_metric='AUC', random_seed=42, verbose=False,
        task_type='GPU', metric_period=1
    )
    
    clf_cat.fit(train_pool, eval_set=val_pool, early_stopping_rounds=150)
    
    cat_oof[val_idx] = clf_cat.predict_proba(X_va)[:, 1]
    cat_preds += clf_cat.predict_proba(X_test_reduced)[:, 1] / folds.n_splits
    print(f"-> CatBoost AUC: {roc_auc_score(y_va, cat_oof[val_idx]):.5f}")

# ========================================================
# 🎯 4. 終極合體：加權融合 (Blending)
# ========================================================
print("\n========== 🏁 訓練結束，開始計算融合結果 ==========")
print(f"LGB 單獨 OOF AUC: {roc_auc_score(y_train, lgb_oof):.5f}")
print(f"XGB 單獨 OOF AUC: {roc_auc_score(y_train, xgb_oof):.5f}")
print(f"CAT 單獨 OOF AUC: {roc_auc_score(y_train, cat_oof):.5f}")

# 1. 定義優化目標函數：尋找能讓 AUC 最大（負值最小）的權重
def objective(weights):
    # 確保權重相加為 1
    w1, w2, w3 = weights
    blend = (w1 * lgb_oof) + (w2 * cat_oof) + (w3 * xgb_oof)
    return -roc_auc_score(y_train, blend) # 加上負號因為 minimize 是找最小值

# 2. 設定初始權重與限制條件
init_weights = [0.4, 0.4, 0.2]
bounds = [(0, 1), (0, 1), (0, 1)] # 權重必須在 0~1 之間
constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)}) # 權重相加等於 1

# 3. 執行優化搜尋
res = minimize(objective, init_weights, bounds=bounds, constraints=constraints, method='SLSQP')

best_w1, best_w2, best_w3 = res.x
print(f"\n🎯 搜尋到的黃金權重比例 -> LGB: {best_w1:.3f}, CAT: {best_w2:.3f}, XGB: {best_w3:.3f}")

# 4. 用黃金權重計算最終結果
opt_oof = (best_w1 * lgb_oof) + (best_w2 * cat_oof) + (best_w3 * xgb_oof)
print(f"🚀 優化權重後的 OOF AUC: {roc_auc_score(y_train, opt_oof):.5f}")

test_df = pd.read_csv('home-credit-default-risk/application_test.csv')

# 5. 生成 Submission 檔案
opt_preds = (best_w1 * lgb_preds) + (best_w2 * cat_preds) + (best_w3 * xgb_preds)
sub = pd.DataFrame({'SK_ID_CURR': test_df['SK_ID_CURR'], 'TARGET': opt_preds})
sub.to_csv('optimized_blend_submission.csv', index=False)