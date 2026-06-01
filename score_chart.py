import matplotlib.pyplot as plt
import numpy as np

# 準備資料
models = [
    "baseline", "bureau", "bu+prev", "bu+prev_magic", 
    "bu+prev+install+credit", "bu+prev+install+3model", 
    "bu+prev+install+3model+tw", "bu+prev+install+3model+tw+stack",
    "bu+prev+install+3model+tw+pos+bubal"
]

# 為了讓標籤好看，稍微縮短了長名稱
score = [0.75943, 0.76602, 0.77540, 0.78415, 0.78581, 0.78480, 0.78590, 0.78590, 0.78747]
score_reduction = [0.0, 0.76844, 0.7764, 0.78439, 0.78580, 0.79010, 0.79171, 0.79147, 0.79313]

x = np.arange(len(models))  # 標籤位置
width = 0.35  # 柱狀圖寬度

fig, ax = plt.subplots(figsize=(14, 7))

# 繪製雙柱
rects1 = ax.bar(x - width/2, score, width, label='Score', color='#1f77b4')
rects2 = ax.bar(x + width/2, score_reduction, width, label='Score Reduction', color='#ff7f0e')

# 設定圖表屬性
ax.set_ylabel('Scores', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha="right", fontsize=10)
ax.legend(fontsize=11)

# 關鍵：設定 Y 軸範圍，讓些微的差距更明顯
ax.set_ylim(0.75, 0.795)

# 加入網格線方便辨識
ax.grid(axis='y', linestyle='--', alpha=0.7)

# 自動調整版面並顯示
plt.tight_layout()
plt.show()