"""
多元校正可视化分析
生成详细的波形分析和模型性能图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.signal import savgol_filter, find_peaks
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
os.makedirs('plots', exist_ok=True)

print("=" * 80)
print("多元校正可视化分析")
print("=" * 80)

# ============================================================================
# 1. 信号预处理
# ============================================================================
def snv(X):
    """标准正态变量变换"""
    return (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)

def preprocess_waveforms(X):
    """预处理流水线"""
    X_smoothed = savgol_filter(X, window_length=11, polyorder=2, axis=1)
    X_preprocessed = snv(X_smoothed)
    return X_preprocessed

# ============================================================================
# 2. 数据加载
# ============================================================================
print("\n[1] 加载数据...")

files = {
    '铝离子': r'c:\Users\28130\Desktop\lyy-6.12\铝离子.xlsx',
    '色氨酸': r'c:\Users\28130\Desktop\lyy-6.12\色氨酸.xlsx',
    '烯酰肉碱': r'c:\Users\28130\Desktop\lyy-6.12\烯酰肉碱.xlsx',
    '混合组': r'c:\Users\28130\Desktop\lyy-6.12\混合组.xlsx'
}

all_data = {}
for name, filepath in files.items():
    df = pd.read_excel(filepath).iloc[1:]
    time = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
    
    waveforms = {}
    if name == '混合组':
        for col in df.columns[1:]:
            voltage = pd.to_numeric(df[col], errors='coerce').values
            waveforms[col] = {'time': time, 'voltage': voltage}
    else:
        concentrations = ['1mg/ml', '0.5mg/ml', '0.1mg/ml']
        for conc in concentrations:
            conc_cols = [col for col in df.columns if conc in str(col)]
            conc_waveforms = []
            for col in conc_cols:
                voltage = pd.to_numeric(df[col], errors='coerce').values
                conc_waveforms.append(voltage)
            
            if conc_waveforms:
                min_len = min(len(v) for v in conc_waveforms)
                waveforms[conc] = {
                    'time': time[:min_len],
                    'waveforms': np.array([v[:min_len] for v in conc_waveforms]),
                    'mean_waveform': np.mean([v[:min_len] for v in conc_waveforms], axis=0)
                }
    
    all_data[name] = waveforms

print(f"加载完成: {len(all_data)} 组数据")

# ============================================================================
# 3. 构建训练数据
# ============================================================================
print("\n[2] 构建训练数据...")

X = []
Y = []
labels = []

substances = ['铝离子', '色氨酸', '烯酰肉碱']
conc_values = {'1mg/ml': 1.0, '0.5mg/ml': 0.5, '0.1mg/ml': 0.1}

# 对齐所有波形
all_waves = []
for name in substances:
    for conc_label, conc_data in all_data[name].items():
        for wave in conc_data['waveforms']:
            all_waves.append(wave)
min_len = min(len(w) for w in all_waves)

for substance_name in substances:
    for conc_label, conc_data in all_data[substance_name].items():
        conc_val = conc_values[conc_label]
        for wave in conc_data['waveforms']:
            X.append(wave[:min_len])
            if substance_name == '铝离子':
                Y.append([conc_val, 0.0, 0.0])
            elif substance_name == '色氨酸':
                Y.append([0.0, conc_val, 0.0])
            else:
                Y.append([0.0, 0.0, conc_val])
            labels.append(f"{substance_name}_{conc_label}")

X = np.array(X)
Y = np.array(Y)

print(f"数据形状: X={X.shape}, Y={Y.shape}")

# 预处理
X_p = preprocess_waveforms(X)
X_train, X_test, Y_train, Y_test = train_test_split(X_p, Y, test_size=0.2, random_state=42)

# ============================================================================
# 4. 训练模型
# ============================================================================
print("\n[3] 训练模型...")

pls = PLSRegression(n_components=7)
pls.fit(X_train, Y_train)
Y_pred_pls = pls.predict(X_test)

svr_models = []
for i in range(3):
    svr = SVR(C=1, gamma='scale', kernel='rbf')
    svr.fit(X_train, Y_train[:, i])
    svr_models.append(svr)

Y_pred_svr = np.column_stack([svr.predict(X_test) for svr in svr_models])

substance_names = ['铝离子', '色氨酸', '烯酰肉碱']

# ============================================================================
# 5. 生成可视化图表
# ============================================================================
print("\n[4] 生成可视化图表...")

# 5.1 原始波形对比图
print("  生成波形对比图...")
fig, axes = plt.subplots(3, 1, figsize=(12, 15))

for idx, substance in enumerate(substances):
    ax = axes[idx]
    for conc, color in zip(['1mg/ml', '0.5mg/ml', '0.1mg/ml'], ['red', 'green', 'blue']):
        mean_wave = all_data[substance][conc]['mean_waveform']
        time = all_data[substance][conc]['time']
        ax.plot(time[:len(mean_wave)], mean_wave, color=color, label=conc, linewidth=1.5)
    
    ax.set_xlabel('时间 (s)', fontsize=12)
    ax.set_ylabel('电压 (V)', fontsize=12)
    ax.set_title(f'{substance} - 不同浓度原始波形', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/waveform_comparison_detailed.png', dpi=300, bbox_inches='tight')

# 5.2 预处理后的波形对比
print("  生成预处理波形图...")
fig, axes = plt.subplots(3, 1, figsize=(12, 15))

for idx, substance in enumerate(substances):
    ax = axes[idx]
    for conc, color in zip(['1mg/ml', '0.5mg/ml', '0.1mg/ml'], ['red', 'green', 'blue']):
        mean_wave = all_data[substance][conc]['mean_waveform'][:min_len]
        X_plot = np.array([mean_wave])
        X_plot_p = preprocess_waveforms(X_plot)
        ax.plot(range(len(X_plot_p[0])), X_plot_p[0], color=color, label=conc, linewidth=1.5)
    
    ax.set_xlabel('采样点', fontsize=12)
    ax.set_ylabel('归一化电压', fontsize=12)
    ax.set_title(f'{substance} - 预处理后波形', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/preprocessed_waveforms.png', dpi=300, bbox_inches='tight')

# 5.3 PLSR权重图
print("  生成PLSR权重图...")
fig, axes = plt.subplots(3, 2, figsize=(16, 12))

for i, name in enumerate(substance_names):
    # 查看三个主成分的权重
    for pc in range(3):
        ax = axes[i, pc//2] if pc < 2 else axes[i, 1]
        if pc < 2:
            weights = pls.coef_[i, :] if i < len(pls.coef_) else np.zeros(X_train.shape[1])
            ax.plot(weights, label=f'{name}', linewidth=1.5)
            ax.set_title(f'PLSR权重 - {name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('特征位置')
            ax.set_ylabel('权重值')
            ax.legend()
            ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/plsr_weights.png', dpi=300, bbox_inches='tight')

# 5.4 预测散点图
print("  生成预测散点图...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, name in enumerate(substance_names):
    ax = axes[i]
    ax.scatter(Y_test[:, i], Y_pred_pls[:, i], alpha=0.6, s=60, label='PLSR')
    ax.scatter(Y_test[:, i], Y_pred_svr[:, i], alpha=0.6, s=60, label='SVR')
    ax.plot([0, 1.2], [0, 1.2], 'r--', linewidth=2)
    
    r2_pls = r2_score(Y_test[:, i], Y_pred_pls[:, i])
    r2_svr = r2_score(Y_test[:, i], Y_pred_svr[:, i])
    
    ax.set_xlabel('真实浓度 (mg/ml)', fontsize=12)
    ax.set_ylabel('预测浓度 (mg/ml)', fontsize=12)
    ax.set_title(f'{name}\nPLSR R²={r2_pls:.4f}, SVR R²={r2_svr:.4f}', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.1, 1.3])
    ax.set_ylim([-0.1, 1.3])

plt.tight_layout()
plt.savefig('plots/prediction_scatter.png', dpi=300, bbox_inches='tight')

# 5.5 模型性能对比图
print("  生成模型性能对比图...")
fig = plt.figure(figsize=(12, 6))

models = ['PLSR', 'SVR']
colors = ['blue', 'green']
x = np.arange(len(substance_names))
width = 0.35

plsr_r2 = [r2_score(Y_test[:, i], Y_pred_pls[:, i]) for i in range(3)]
svr_r2 = [r2_score(Y_test[:, i], Y_pred_svr[:, i]) for i in range(3)]

plt.bar(x - width/2, plsr_r2, width, label='PLSR', color=colors[0], alpha=0.8)
plt.bar(x + width/2, svr_r2, width, label='SVR', color=colors[1], alpha=0.8)

plt.xlabel('物质', fontsize=12)
plt.ylabel('R² 分数', fontsize=12)
plt.title('PLSR vs SVR 性能对比', fontsize=14, fontweight='bold')
plt.xticks(x, substance_names)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])

plt.tight_layout()
plt.savefig('plots/model_performance.png', dpi=300, bbox_inches='tight')

# 5.6 特征峰值分析
print("  生成特征峰值分析图...")
fig, axes = plt.subplots(3, 1, figsize=(12, 15))

for idx, substance in enumerate(substances):
    ax = axes[idx]
    mean_wave = all_data[substance]['1mg/ml']['mean_waveform'][:min_len]
    X_plot = np.array([mean_wave])
    X_plot_p = preprocess_waveforms(X_plot)[0]
    
    peaks, props = find_peaks(X_plot_p, prominence=0.3)
    valleys, _ = find_peaks(-X_plot_p, prominence=0.3)
    
    ax.plot(range(len(X_plot_p)), X_plot_p, linewidth=1.5, label='预处理波形')
    ax.scatter(peaks, X_plot_p[peaks], color='red', s=50, label='峰值')
    ax.scatter(valleys, X_plot_p[valleys], color='blue', s=50, label='谷值')
    
    ax.set_xlabel('采样点', fontsize=12)
    ax.set_ylabel('归一化电压', fontsize=12)
    ax.set_title(f'{substance} - 峰值分析 (峰值数: {len(peaks)}, 谷值数: {len(valleys)})', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/peak_analysis.png', dpi=300, bbox_inches='tight')

# 5.7 误差分布直方图
print("  生成误差分布图...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, name in enumerate(substance_names):
    ax = axes[i]
    
    pls_errors = np.abs(Y_test[:, i] - Y_pred_pls[:, i])
    svr_errors = np.abs(Y_test[:, i] - Y_pred_svr[:, i])
    
    ax.hist(pls_errors, bins=10, alpha=0.6, label='PLSR', edgecolor='black')
    ax.hist(svr_errors, bins=10, alpha=0.6, label='SVR', edgecolor='black')
    
    ax.axvline(np.mean(pls_errors), color='blue', linestyle='--', label=f'PLSR均值: {np.mean(pls_errors):.4f}')
    ax.axvline(np.mean(svr_errors), color='green', linestyle='--', label=f'SVR均值: {np.mean(svr_errors):.4f}')
    
    ax.set_xlabel('预测误差 (mg/ml)', fontsize=12)
    ax.set_ylabel('频数', fontsize=12)
    ax.set_title(f'{name} - 预测误差分布', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/error_distribution.png', dpi=300, bbox_inches='tight')

# 5.8 PLSR得分图
print("  生成PLSR得分图...")
fig = plt.figure(figsize=(10, 8))

# 获取PLSR得分（训练集）
scores = pls.x_scores_

# 按浓度着色（只使用训练集对应的标签）
train_indices = []
for i, (x, y) in enumerate(zip(X_p, Y)):
    for j, (xt, yt) in enumerate(zip(X_train, Y_train)):
        if np.array_equal(x, xt) and np.array_equal(y, yt):
            if j not in train_indices:
                train_indices.append(j)
                break

conc_labels_train = []
for idx in train_indices[:len(scores)]:
    label = labels[idx]
    if '1mg/ml' in label:
        conc_labels_train.append(0)
    elif '0.5mg/ml' in label:
        conc_labels_train.append(1)
    else:
        conc_labels_train.append(2)

scatter = plt.scatter(scores[:, 0], scores[:, 1], c=conc_labels_train, cmap='viridis', s=100, alpha=0.7)
plt.xlabel('PC1', fontsize=12)
plt.ylabel('PC2', fontsize=12)
plt.title('PLSR得分图 (按浓度着色)', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ticks=[0, 1, 2], label='浓度 (0=1mg/ml, 1=0.5mg/ml, 2=0.1mg/ml)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/plsr_scores.png', dpi=300, bbox_inches='tight')

# 5.9 混合组波形对比
print("  生成混合组波形图...")
fig = plt.figure(figsize=(14, 8))

mixed_data = all_data['混合组']
colors = {'高铝': 'red', '高色': 'green', '高烯': 'blue'}

for col, color in colors.items():
    if col in mixed_data:
        voltage = mixed_data[col]['voltage']
        mask = ~np.isnan(voltage)
        clean_voltage = voltage[mask]
        X_plot = np.array([clean_voltage[:min_len]])
        X_plot_p = preprocess_waveforms(X_plot)[0]
        plt.plot(range(len(X_plot_p)), X_plot_p, color=color, label=col, linewidth=2)

plt.xlabel('采样点', fontsize=12)
plt.ylabel('归一化电压', fontsize=12)
plt.title('混合组预处理波形对比', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/mixed_group_waveforms.png', dpi=300, bbox_inches='tight')

# 5.10 预测结果条形图
print("  生成混合组预测结果图...")
fig = plt.figure(figsize=(12, 6))

mixed_results_pls = []
mixed_results_svr = []

for i, name in enumerate(substance_names):
    # 预测混合组
    if f'高{name[0]}' in mixed_data:
        voltage = mixed_data[f'高{name[0]}']['voltage']
        mask = ~np.isnan(voltage)
        clean_voltage = voltage[mask][:min_len]
        X_pred = np.array([clean_voltage])
        X_pred_p = preprocess_waveforms(X_pred)
        
        pls_pred = pls.predict(X_pred_p)[0][i]
        svr_pred = svr_models[i].predict(X_pred_p)[0]
        
        mixed_results_pls.append(pls_pred)
        mixed_results_svr.append(svr_pred)

x = np.arange(len(substance_names))
width = 0.35

plt.bar(x - width/2, mixed_results_pls, width, label='PLSR预测', color='blue', alpha=0.8)
plt.bar(x + width/2, mixed_results_svr, width, label='SVR预测', color='green', alpha=0.8)

plt.xlabel('物质', fontsize=12)
plt.ylabel('预测浓度 (mg/ml)', fontsize=12)
plt.title('混合组浓度预测结果', fontsize=14, fontweight='bold')
plt.xticks(x, substance_names)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])

for i, (pls_val, svr_val) in enumerate(zip(mixed_results_pls, mixed_results_svr)):
    plt.text(i - width/2, pls_val + 0.02, f'{pls_val:.4f}', ha='center', fontsize=10)
    plt.text(i + width/2, svr_val + 0.02, f'{svr_val:.4f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('plots/mixed_group_predictions.png', dpi=300, bbox_inches='tight')

# ============================================================================
# 总结
# ============================================================================
print("\n[5] 可视化完成！")
print("=" * 80)
print("生成的图表:")
print("-" * 40)
print("1. waveform_comparison_detailed.png - 原始波形对比")
print("2. preprocessed_waveforms.png - 预处理波形对比")
print("3. plsr_weights.png - PLSR权重图")
print("4. prediction_scatter.png - 预测散点图")
print("5. model_performance.png - 模型性能对比")
print("6. peak_analysis.png - 特征峰值分析")
print("7. error_distribution.png - 误差分布")
print("8. plsr_scores.png - PLSR得分图")
print("9. mixed_group_waveforms.png - 混合组波形")
print("10. mixed_group_predictions.png - 混合组预测")
print("=" * 80)
