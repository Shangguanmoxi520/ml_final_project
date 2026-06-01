import matplotlib.pyplot as plt
import pandas as pd

# 1. 建立原始資料集
data = {
    "feature": [
        "MAGIC_EXT_SOURCES_MEAN",
        "MAGIC_EXT_SOURCES_MIN",
        "MAGIC_EXT_SOURCES_MAX",
        "MAGIC_ANNUITY_TO_CREDIT_RATIO",
        "EXT_SOURCE_3",
        "CC_6M_CC_LIMIT_USE_RATIO_MAX",
        "DAYS_BIRTH",
        "MAGIC_CREDIT_TO_GOODS_RATIO",
        "POS_6M_CNT_INSTALMENT_FUTURE_MEAN",
        "BUREAU_DAYS_CREDIT_MAX_y",
        "others",
    ],
    "importance": [
        44577.958178,
        16465.286496,
        7913.668931,
        6087.959786,
        4385.693044,
        4226.744103,
        4204.079051,
        4030.467817,
        3492.832447,
        2504.324166,
        0,  # 這裡設為0沒關係，因為我們會用前10名的累積比例回推總分母
    ],
}

df = pd.DataFrame(data)

# 2. 精確計算真實總重要性 (Total Importance Base)
# 已知第9索引(第10個特徵)的累積比例為 0.361615
top10_importance_sum = df.iloc[0:10]["importance"].sum()
true_total_importance = top10_importance_sum / 0.361615

# 3. 計算每個特徵在「全體模型」中的真實比例
df["proportion"] = df["importance"] / true_total_importance

# 4. 過濾掉 'others' 進行繪圖準備
df_filtered = df[df["feature"] != "others"].copy()

# 5. 為了讓圖表從上到下是由大到小排列，將資料逆序
df_filtered = df_filtered.iloc[::-1]

# 6. 開始繪圖
fig, ax = plt.subplots(figsize=(12, 6))

# 繪製水平柱狀圖
bars = ax.barh(
    df_filtered["feature"],
    df_filtered["proportion"],
    color="#2b5c8f",
    edgecolor="black",
    height=0.6,
)

# 7. 在柱狀圖右側加上真實比例數值標籤
for bar in bars:
    width = bar.get_width()
    ax.text(
        width + 0.005,  # 標籤顯示在柱子右側一點點
        bar.get_y() + bar.get_height() / 2,
        f"{width:.2%}",  # 顯示到小數點後兩位更精準
        va="center",
        ha="left",
        fontsize=10,
        fontweight="bold",
    )

# 8. 設定圖表細節
ax.set_title(
    "True Feature Importance Proportion in Global Model (Excluding Others)",
    fontsize=14,
    pad=15,
    fontweight="bold",
)
ax.set_xlabel("True Proportion in Model (Total = 100%)", fontsize=12, labelpad=10)
ax.set_xlim(0, 0.20)  # 因為最大佔 16.47%，X軸切到 20% 剛剛好

# 隱藏上方和右方的框線
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 加入直向網格線
ax.grid(axis="x", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()