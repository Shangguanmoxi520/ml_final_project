import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

from category_encoders import TargetEncoder
from sklearn.model_selection import KFold

from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

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

# 2. 指定要進行極致壓榨的高基數類別欄位
# 注意：這些欄位在前面讀入主表後，請「不要」對它們做 pd.get_dummies
te_cols = ['OCCUPATION_TYPE', 'ORGANIZATION_TYPE', 'WEEKDAY_APPR_PROCESS_START']
# 確保這些欄位目前在 X_train_full 裡還是 object 或 category 型態

for col in cat_cols:
    # 🚀 關鍵新增：如果欄位在 Target Encoding 的名單內，直接跳過不處理！
    if col in te_cols:
        continue

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

# ========================================================
# 🚀 引入並處理第四張子表 installments_payments.csv
# ========================================================
print("正在處理 installments_payments.csv...")
ins = pd.read_csv('home-credit-default-risk/installments_payments.csv')

# 1. 創造還款行為的衍生變數（在 Groupby 之前做）
# 算出每筆還款遲到幾天（正數代表遲到，負數代表提早）
ins['INS_DAYS_LATE'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']

# 抓出「真的遲到」的天數（提早還的當作 0 天）
ins['INS_DAYS_LATE_ONLY'] = ins['INS_DAYS_LATE'].apply(lambda x: x if x > 0 else 0)

# 算出這期少給了多少錢（正數代表少給，負數或0代表給足或給多）
ins['INS_AMT_LESS'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
ins['INS_AMT_LESS_ONLY'] = ins['INS_AMT_LESS'].apply(lambda x: x if x > 0 else 0)

# 2. 定義聚合規則
ins_num_aggregations = {
    'NUM_INSTALMENT_VERSION': ['nunique'], # 換過幾次還款合約（通常代表有改過期數）
    'INS_DAYS_LATE': ['max', 'mean', 'sum'], # 整體還款天數表現
    'INS_DAYS_LATE_ONLY': ['max', 'mean', 'sum'], # 純粹遲到的嚴重程度
    'INS_AMT_LESS': ['max', 'mean', 'sum'], # 整體少給錢的表現
    'INS_AMT_LESS_ONLY': ['max', 'mean', 'sum'], # 純粹少給錢的嚴重程度
    'AMT_INSTALMENT': ['max', 'mean', 'sum'], # 應還款金額的規模
    'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'], # 實際還款金額的規模
    'DAYS_ENTRY_PAYMENT': ['max', 'mean'] # 最近一次還款是什麼時候
}

# 3. 執行 Groupby 聚合
ins_agg = ins.groupby('SK_ID_CURR').agg(ins_num_aggregations)

# 重新整理欄位名稱
ins_agg.columns = pd.Index(['INS_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])

# 4. 新增一個特徵：這個人過去總共繳過幾次款
ins_counts = ins.groupby('SK_ID_CURR').size().to_frame('INS_PAYMENT_COUNT')
ins_agg = ins_agg.join(ins_counts, how='left')

# --- 🚀 關鍵新增：近 6 個月 (180天) 的還款表現 ---
print("正在計算近 6 個月還款特徵...")
# DAYS_INSTALMENT 是負數，-180 代表過去 180 天內
ins_6m = ins[ins['DAYS_INSTALMENT'] >= -180]

if not ins_6m.empty:
    ins_6m_agg = ins_6m.groupby('SK_ID_CURR').agg({
        'INS_DAYS_LATE_ONLY': ['max', 'mean', 'sum'], # 最近有沒有頻繁遲到
        'INS_AMT_LESS_ONLY': ['max', 'mean', 'sum'],  # 最近手頭是不是變緊少給錢
        'AMT_PAYMENT': ['sum']                        # 最近總共繳了多少錢
    })
    ins_6m_agg.columns = pd.Index(['INS_6M_' + e[0] + "_" + e[1].upper() for e in ins_6m_agg.columns.tolist()])
    
    # 拼回原本的 ins_agg
    ins_agg = ins_agg.join(ins_6m_agg, how='left')

    # 🔮 進階魔術：趨勢特徵（近期遲到天數 / 全局遲到天數）
    ins_agg['INS_TREND_LATE'] = ins_agg['INS_6M_INS_DAYS_LATE_ONLY_MEAN'] / (ins_agg['INS_INS_DAYS_LATE_ONLY_MEAN'] + 1e-5)

# 5. 將聚合後的特徵拼回主表 df
df['SK_ID_CURR'] = pd.concat([train_df['SK_ID_CURR'], test_df['SK_ID_CURR']], axis=0).values
df = df.merge(ins_agg, on='SK_ID_CURR', how='left')
df.drop(columns=['SK_ID_CURR'], inplace=True)

print("installments_payments 處理並合併完成！")

# ========================================================
# 🚀 引入並處理第五張子表 credit_card_balance.csv
# ========================================================
print("正在處理 credit_card_balance.csv...")
cc = pd.read_csv('home-credit-default-risk/credit_card_balance.csv')

# 1. 修正異常值（天數或次數若有異常大數，轉為 NaN）
cc['AMT_DRAWINGS_ATM_CURRENT'].replace(365243, np.nan, inplace=True)

# 2. 創造信用卡的衍生行為變數（在 Groupby 之前做）
# (A) 額度使用率：當月欠款 / 當月總額度
cc['CC_LIMIT_USE_RATIO'] = cc['AMT_BALANCE'] / (cc['AMT_CREDIT_LIMIT_ACTUAL'] + 1e-5)

# (B) 【修正版】實質提現金額 = ATM 提現 + 其他提現
cc['CC_TOTAL_CASH_DRAWINGS'] = cc['AMT_DRAWINGS_ATM_CURRENT'].fillna(0) + cc['AMT_DRAWINGS_OTHER_CURRENT'].fillna(0)

# 算出「提現金額」佔「當月總刷卡消費額」的比例 (提現比例高 = 財務週轉極度吃緊)
cc['CC_CASH_DRAW_RATIO'] = cc['CC_TOTAL_CASH_DRAWINGS'] / (cc['AMT_DRAWINGS_CURRENT'] + 1e-5)

# (C) 當月是否沒還夠錢（應付總額 - 實際還款）
cc['CC_REPAY_GAP'] = cc['AMT_TOTAL_RECEIVABLE'] - cc['AMT_PAYMENT_CURRENT']
cc['CC_REPAY_GAP_ONLY'] = cc['CC_REPAY_GAP'].apply(lambda x: x if x > 0 else 0)

# 3. 類別型欄位做 One-Hot Encoding (主要是 NAME_CONTRACT_STATUS，如 Active, Completed)
cc_cat_cols = cc.select_dtypes(include=['object']).columns
cc = pd.get_dummies(cc, columns=cc_cat_cols, drop_first=True)

# 4. 定義數值型欄位要做哪些聚合計算
cc_num_aggregations = {
    'MONTHS_BALANCE': ['min', 'max', 'mean'],      
    'AMT_BALANCE': ['max', 'mean'],                 
    'AMT_CREDIT_LIMIT_ACTUAL': ['max', 'mean'],     
    'CC_LIMIT_USE_RATIO': ['max', 'mean', 'var'],   
    'CC_TOTAL_CASH_DRAWINGS': ['max', 'sum'],       # 新增：看他過去最高一次提現多少、總共提現多少
    'CC_CASH_DRAW_RATIO': ['max', 'mean'],          
    'CC_REPAY_GAP_ONLY': ['max', 'mean', 'sum'],  
    'CNT_DRAWINGS_ATM_CURRENT': ['max', 'sum'],     
    'CNT_DRAWINGS_CURRENT': ['max', 'sum'],         
    'SK_DPD': ['max', 'mean'],                      
    'SK_DPD_DEF': ['max', 'mean']                   
}

# 5. 對 One-Hot 展開後的類別欄位計算平均值
cc_cat_aggregations = {}
for col in cc.columns:
    if col not in cc_num_aggregations and col != 'SK_ID_CURR' and col != 'SK_ID_PREV':
        cc_cat_aggregations[col] = ['mean']

# 合併聚合規則
cc_agg_rules = {**cc_num_aggregations, **cc_cat_aggregations}

# 6. 執行 Groupby 聚合
cc_agg = cc.groupby('SK_ID_CURR').agg(cc_agg_rules)

# 重新整理欄位名稱
cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])

# 7. 新增一個特徵：這個人總共有幾個月的信用卡帳單紀錄
cc_counts = cc.groupby('SK_ID_CURR').size().to_frame('CC_RECORD_COUNT')
cc_agg = cc_agg.join(cc_counts, how='left')

# --- 全局聚合 (你原本寫好的部分，保留) ---
# ... (原本的 cc_agg 邏輯) ...

# --- 🚀 關鍵新增：近 6 個月的信用卡消費與債務惡化狀況 ---
print("正在計算近 6 個月信用卡特徵...")
# MONTHS_BALANCE >= -6 代表過去 6 個月內
cc_6m = cc[cc['MONTHS_BALANCE'] >= -6]

if not cc_6m.empty:
    cc_6m_agg = cc_6m.groupby('SK_ID_CURR').agg({
        'AMT_BALANCE': ['max', 'mean'],               # 最近卡債規模是不是飆高
        'CC_LIMIT_USE_RATIO': ['max', 'mean'],        # 最近有沒有經常把卡刷爆
        'CC_REPAY_GAP_ONLY': ['max', 'mean', 'sum']   # 最近是不是開始還不出最低應繳
    })
    cc_6m_agg.columns = pd.Index(['CC_6M_' + e[0] + "_" + e[1].upper() for e in cc_6m_agg.columns.tolist()])
    
    # 拼回原本的 cc_agg
    cc_agg = cc_agg.join(cc_6m_agg, how='left')
    
    # 🔮 進階魔術：趨勢特徵（近期卡債 / 全局平均卡債）
    cc_agg['CC_TREND_BALANCE'] = cc_agg['CC_6M_AMT_BALANCE_MEAN'] / (cc_agg['CC_AMT_BALANCE_MEAN'] + 1e-5)

# 8. 將聚合後的特徵拼回主表 df
df['SK_ID_CURR'] = pd.concat([train_df['SK_ID_CURR'], test_df['SK_ID_CURR']], axis=0).values
df = df.merge(cc_agg, on='SK_ID_CURR', how='left')
df.drop(columns=['SK_ID_CURR'], inplace=True)

print("credit_card_balance 處理並合併完成！")

# ========================================================
# 🔮 魔術特徵工程 (Magic Feature Engineering)
# ========================================================
print("正在注入魔術特徵組合...")

# 1. 外部評分 (EXT_SOURCE) 的多維度組合
# 因為這三個指標最強，我們幫模型算好它們的乘積、非零平均值、幾何平均、最大與最小值
df['MAGIC_EXT_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
df['MAGIC_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
df['MAGIC_EXT_SOURCES_MAX'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].max(axis=1)
df['MAGIC_EXT_SOURCES_MIN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1)

# 2. 進階金融負擔比率 (Financial Ratios)
# 貸款額度佔年收入比例 (還款壓力)
df['MAGIC_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1e-5)
# 每期年金佔年收入比例 (每月手頭緊不緊)
df['MAGIC_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1e-5)
# 每期年金佔貸款總額比例 (相當於借款期數的倒數)
df['MAGIC_ANNUITY_TO_CREDIT_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + 1e-5)
# 實際核貸金額與商品價格的差額與比例 (代表頭期款或溢價程度)
df['MAGIC_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1e-5)
df['MAGIC_CREDIT_GOODS_DIFF'] = df['AMT_CREDIT'] - df['AMT_GOODS_PRICE']

# 3. 年齡與工作穩定度交叉
# 工作天數佔年齡的比例 (負負得正，可以看出成年後有多大比例的時間在工作)
df['MAGIC_EMPLOYED_TO_AGE_RATIO'] = df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH'] + 1e-5)

# 4. 跨表總負債比率 (結合 Bureau 資料)
# 如果外面有借錢 (BUREAU_AMT_CREDIT_SUM_SUM 來自於你先前 merge 進來的欄位名稱)
# 請檢查你先前 merge 進來後的總負債欄位名稱，通常是 'BUREAU_AMT_CREDIT_SUM_SUM_y' 或 'BUREAU_AMT_CREDIT_SUM_SUM'
bureau_debt_col = 'BUREAU_AMT_CREDIT_SUM_SUM_y' if 'BUREAU_AMT_CREDIT_SUM_SUM_y' in df.columns else 'BUREAU_AMT_CREDIT_SUM_SUM'

if bureau_debt_col in df.columns:
    df['MAGIC_TOTAL_DEBT_TO_INCOME'] = df[bureau_debt_col] / (df['AMT_INCOME_TOTAL'] + 1e-5)
    df['MAGIC_CURRENT_CREDIT_TO_OLD_DEBT'] = df['AMT_CREDIT'] / (df[bureau_debt_col] + 1e-5)

# 看他過去總共繳了多少還款，佔他這次申請收入的比例（過去的還款壓力）
if 'INS_AMT_PAYMENT_SUM' in df.columns:
    df['MAGIC_PAST_PAYMENT_TO_INCOME'] = df['INS_AMT_PAYMENT_SUM'] / (df['AMT_INCOME_TOTAL'] + 1e-5)

# 信用卡的最高欠款總額佔他這次申請年收入的比例
if 'CC_AMT_BALANCE_MAX' in df.columns:
    df['MAGIC_CC_DEBT_TO_INCOME'] = df['CC_AMT_BALANCE_MAX'] / (df['AMT_INCOME_TOTAL'] + 1e-5)

print("魔術特徵注入完畢！")
# ========================================================

print("🚀 開始執行安全版 Out-of-Fold Target Encoding...")

# 1. 準備切分訓練集與測試集（此時 df 包含 train 和 test）
# 假設你原本用來區分 train/test 的方式是看長度
train_len = len(train_df)
X_train_full = df.iloc[:train_len, :].copy()
X_test_full = df.iloc[train_len:, :].copy()
y_train_full = train_df['TARGET'].copy()

# --- 🛠️ 核心修正開始 ---
# 建立一個和 X_train_full 一樣大、欄位與 te_cols 相同的空矩陣，用來存數值結果
train_te_preds = pd.DataFrame(np.zeros((X_train_full.shape[0], len(te_cols))), 
                               columns=[c + '_TE' for c in te_cols], 
                               index=X_train_full.index)

test_te_preds = pd.DataFrame(np.zeros((X_test_full.shape[0], len(te_cols))), 
                              columns=[c + '_TE' for c in te_cols], 
                              index=X_test_full.index)

# 4. 使用 5-Fold 進行折外編碼
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in kf.split(X_train_full, y_train_full):
    X_tr, y_tr = X_train_full.iloc[train_idx], y_train_full.iloc[train_idx]
    X_va = X_train_full.iloc[val_idx]
    
    # 確保轉換時資料型態一致
    encoder = TargetEncoder(cols=te_cols, smoothing=10.0)
    encoder.fit(X_tr[te_cols].astype(str), y_tr)
    
    # 轉換驗證集，並透過 index 填入我們剛才準備好的純數值矩陣
    val_transformed = encoder.transform(X_va[te_cols].astype(str))
    train_te_preds.iloc[val_idx, :] = val_transformed.values
    
    # 同時對測試集進行轉換，並取平均
    test_transformed = encoder.transform(X_test_full[te_cols].astype(str))
    test_te_preds += test_transformed.values / 5

# 5. 把原本的文字欄位移除，並拼上我們算好的高濃度 _TE 欄位
X_train_full = X_train_full.drop(columns=te_cols).join(train_te_preds)
X_test_full = X_test_full.drop(columns=te_cols).join(test_te_preds)

# 6. 重新合併回你的完整 df
df = pd.concat([X_train_full, X_test_full], axis=0)
print("🎉 OOF Target Encoding 修正版壓榨完成！欄位已成功進化為數值型態。")

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
    'learning_rate': 0.07,
    'num_leaves': 31,
    'max_depth': -1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': -1,
    'random_state': 42,
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
    
    clf_cat = CatBoostClassifier(iterations=4000, learning_rate=0.07, depth=5,
                                 eval_metric='AUC', random_seed=42, verbose=False,
                                 task_type='GPU', metric_period=1)
    
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

# 5. 生成 Submission 檔案
opt_preds = (best_w1 * lgb_preds) + (best_w2 * cat_preds) + (best_w3 * xgb_preds)
sub = pd.DataFrame({'SK_ID_CURR': test_df['SK_ID_CURR'], 'TARGET': opt_preds})
sub.to_csv('optimized_blend_submission.csv', index=False)