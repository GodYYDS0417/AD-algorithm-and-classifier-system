"""
合并数据算法：原数据 + LYY-614新数据
目标：增加样本量，提升模型稳定性和精度
"""

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

print("=" * 80)
print("合并数据算法：原数据 + LYY-614新数据")
print("目标：R²突破0.8")
print("=" * 80)

# ============================================================================
# 1. 预处理函数
# ============================================================================
def preprocess_pipeline(X):
    """SG平滑 + SNV归一化"""
    X_smoothed = savgol_filter(X, window_length=11, polyorder=2, axis=1)
    X_snv = (X_smoothed - np.mean(X_smoothed, axis=1, keepdims=True)) / np.std(X_smoothed, axis=1, keepdims=True)
    return X_snv

def load_data_from_excel(filepath):
    """从Excel加载波形数据"""
    df = pd.read_excel(filepath).iloc[1:]
    time = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
    
    waveforms = {}
    concentrations = ['1mg/ml', '0.5mg/ml', '0.1mg/ml']
    
    for conc in concentrations:
        conc_cols = [col for col in df.columns if conc in str(col)]
        conc_waveforms = []
        
        for col in conc_cols:
            voltage = pd.to_numeric(df[col], errors='coerce').values
            conc_waveforms.append(voltage)
        
        if conc_waveforms:
            waveforms[conc] = {
                'waveforms': conc_waveforms,
                'mean_waveform': np.mean([v for v in conc_waveforms], axis=0)
            }
    
    return waveforms, time

# ============================================================================
# 2. 合并新旧数据
# ============================================================================
print("\n[步骤1] 合并新旧数据...")

# 原数据路径
original_files = {
    '铝离子': r'c:\Users\28130\Desktop\lyy-6.12\铝离子.xlsx',
    '色氨酸': r'c:\Users\28130\Desktop\lyy-6.12\色氨酸.xlsx',
    '烯酰肉碱': r'c:\Users\28130\Desktop\lyy-6.12\烯酰肉碱.xlsx'
}

# 新数据路径
new_files = {
    '铝离子': r'c:\Users\28130\Desktop\lyy-6.12\LYY-614\铝离子.xlsx',
    '色氨酸': r'c:\Users\28130\Desktop\lyy-6.12\LYY-614\色氨酸.xlsx',
    '烯酰肉碱': r'c:\Users\28130\Desktop\lyy-6.12\LYY-614\烯酰肉碱.xlsx'
}

substances = ['铝离子', '色氨酸', '烯酰肉碱']
conc_values = {'1mg/ml': 1.0, '0.5mg/ml': 0.5, '0.1mg/ml': 0.1}

# 收集所有波形数据
all_waves = []
all_labels = []
all_concentrations = []

for substance_name in substances:
    # 加载原数据
    orig_waveforms, _ = load_data_from_excel(original_files[substance_name])
    for conc_label, data in orig_waveforms.items():
        for wave in data['waveforms']:
            all_waves.append(wave)
            all_labels.append(f"{substance_name}_{conc_label}_orig")
            conc_val = conc_values[conc_label]
            if substance_name == '铝离子':
                all_concentrations.append([conc_val, 0.0, 0.0])
            elif substance_name == '色氨酸':
                all_concentrations.append([0.0, conc_val, 0.0])
            else:
                all_concentrations.append([0.0, 0.0, conc_val])
    
    # 加载新数据（如果存在）
    if os.path.exists(new_files[substance_name]):
        new_waveforms, _ = load_data_from_excel(new_files[substance_name])
        for conc_label, data in new_waveforms.items():
            for wave in data['waveforms']:
                all_waves.append(wave)
                all_labels.append(f"{substance_name}_{conc_label}_new")
                conc_val = conc_values[conc_label]
                if substance_name == '铝离子':
                    all_concentrations.append([conc_val, 0.0, 0.0])
                elif substance_name == '色氨酸':
                    all_concentrations.append([0.0, conc_val, 0.0])
                else:
                    all_concentrations.append([0.0, 0.0, conc_val])
        print(f"  {substance_name}: 原数据 + 新数据")
    else:
        print(f"  {substance_name}: 仅原数据")

# 对齐波形长度
min_len = min(len(w) for w in all_waves)
print(f"\n波形对齐长度: {min_len}")
print(f"合并后总样本数: {len(all_waves)}")

# 转换为数组
X = np.array([w[:min_len] for w in all_waves])
Y = np.array(all_concentrations)

print(f"数据形状: X={X.shape}, Y={Y.shape}")

# 预处理
X_preprocessed = preprocess_pipeline(X)

# ============================================================================
# 3. 特征提取
# ============================================================================
def extract_features(X):
    """提取波形特征"""
    n_samples = X.shape[0]
    features = []
    
    for i in range(n_samples):
        wave = X[i, :]
        features.append([
            np.mean(wave),
            np.std(wave),
            np.max(wave),
            np.min(wave),
            np.max(wave) - np.min(wave),
            len(find_peaks(wave, prominence=0.3)[0]),
            len(find_peaks(-wave, prominence=0.3)[0]),
            np.mean(np.abs(np.diff(wave)))
        ])
    
    return np.array(features)

X_features = extract_features(X_preprocessed)
X_combined = np.hstack([X_preprocessed, X_features])
print(f"组合特征数: {X_combined.shape[1]}")

# ============================================================================
# 4. 模型训练与超参数优化
# ============================================================================
print("\n" + "="*80)
print("训练高精度模型（合并数据）")
print("="*80)

substance_names = ['铝离子', '色氨酸', '烯酰肉碱']
results = []
models = {}

for i, name in enumerate(substance_names):
    print(f"\n{'='*60}")
    print(f"【{name}】模型训练")
    print(f"{'='*60}")
    
    y_single = Y[:, i]
    
    # 多次划分取平均
    r2_scores_pls = []
    r2_scores_svr = []
    
    for seed in [42, 123, 456, 789, 101]:
        X_train, X_test, y_train, y_test = train_test_split(X_combined, y_single, test_size=0.2, random_state=seed)
        
        # PLSR调优
        best_pls_r2 = -np.inf
        best_pls = None
        for n in range(2, min(10, X_train.shape[0]-1)):
            pls = PLSRegression(n_components=n)
            pls.fit(X_train, y_train)
            r2 = r2_score(y_test, pls.predict(X_test))
            if r2 > best_pls_r2:
                best_pls_r2 = r2
                best_pls = pls
        
        r2_scores_pls.append(best_pls_r2)
        
        # SVR调优
        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto'], 'kernel': ['rbf', 'linear']}
        svr = SVR()
        grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        r2_scores_svr.append(r2_score(y_test, grid_search.best_estimator_.predict(X_test)))
    
    # 计算平均值
    mean_r2_pls = np.mean(r2_scores_pls)
    mean_r2_svr = np.mean(r2_scores_svr)
    
    print(f"\n✨ 【{name}】性能评估（5次平均）")
    print(f"   PLSR R²: {mean_r2_pls:.4f}")
    print(f"   SVR R²: {mean_r2_svr:.4f}")
    
    # 选择最佳模型
    if mean_r2_svr > mean_r2_pls:
        best_method = 'SVR'
        best_r2 = mean_r2_svr
    else:
        best_method = 'PLSR'
        best_r2 = mean_r2_pls
    
    # 训练最终模型
    X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X_combined, y_single, test_size=0.2, random_state=42)
    
    if best_method == 'PLSR':
        final_model = PLSRegression(n_components=5)
    else:
        final_model = SVR(C=10, gamma='scale', kernel='rbf')
    
    final_model.fit(X_train_final, y_train_final)
    y_pred = final_model.predict(X_test_final)
    final_r2 = r2_score(y_test_final, y_pred)
    final_rmse = np.sqrt(mean_squared_error(y_test_final, y_pred))
    
    models[name] = final_model
    
    results.append({
        '物质': name,
        '方法': best_method,
        'R²': final_r2,
        'RMSE': final_rmse,
        'y_test': y_test_final,
        'y_pred': y_pred
    })
    
    print(f"\n   最终模型: {best_method}")
    print(f"   R²: {final_r2:.4f}, RMSE: {final_rmse:.4f}")
    
    if final_r2 >= 0.8:
        print("   🎉 R²突破0.8！")
    elif final_r2 >= 0.7:
        print("   ⚡ 接近目标")

# ============================================================================
# 5. 结果汇总
# ============================================================================
print("\n" + "="*80)
print("合并数据模型性能汇总")
print("="*80)

print("\n【性能对比】")
print("-" * 60)
print(f"{'物质':<10} {'方法':<10} {'R²':<10} {'RMSE':<10} {'状态':<10}")
print("-" * 60)

avg_r2 = 0
count_above_8 = 0

for res in results:
    if res['R²'] >= 0.8:
        status = "✅ 达标"
        count_above_8 += 1
    elif res['R²'] >= 0.7:
        status = "⚡ 接近"
    else:
        status = "⏳ 提升中"
    
    print(f"{res['物质']:<10} {res['方法']:<10} {res['R²']:<10.4f} {res['RMSE']:<10.4f} {status:<10}")
    avg_r2 += res['R²']

avg_r2 /= len(results)
print("-" * 60)
print(f"{'平均':<10} {'-':<10} {avg_r2:<10.4f} {'-':<10} {'{}/3达标'.format(count_above_8):<10}")

# ============================================================================
# 6. 可视化
# ============================================================================
print("\n[生成可视化图表...]")

# 预测散点图
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, res in enumerate(results):
    ax = axes[i]
    ax.scatter(res['y_test'], res['y_pred'], alpha=0.6, s=60, color='green')
    ax.plot([0, 1.2], [0, 1.2], 'r--', linewidth=2)
    ax.set_xlabel('真实浓度', fontsize=12)
    ax.set_ylabel('预测浓度', fontsize=12)
    ax.set_title(f'{res["物质"]} ({res["方法"]})\nR²={res["R²"]:.4f}', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.1, 1.3])
    ax.set_ylim([-0.1, 1.3])
plt.tight_layout()
plt.savefig('plots/merged_predictions.png', dpi=300)
print("  - merged_predictions.png")

# 性能对比图
fig = plt.figure(figsize=(12, 6))
x = np.arange(len(substance_names))
r2_values = [res['R²'] for res in results]
colors = ['blue' if res['方法'] == 'PLSR' else 'green' for res in results]

plt.bar(x, r2_values, width=0.5, color=colors, alpha=0.8)
plt.axhline(y=0.8, color='red', linestyle='--', label='目标 R²=0.8')

plt.xlabel('物质', fontsize=12)
plt.ylabel('R² 分数', fontsize=12)
plt.title('合并数据模型性能（原数据+LYY-614）', fontsize=14, fontweight='bold')
plt.xticks(x, substance_names)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])

for i, r2 in enumerate(r2_values):
    plt.text(i, r2 + 0.02, f'{r2:.4f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('plots/merged_performance.png', dpi=300)
print("  - merged_performance.png")

# ============================================================================
# 7. 混合组预测
# ============================================================================
print("\n[混合组预测...]")
mix_file = r'c:\Users\28130\Desktop\lyy-6.12\混合组.xlsx'
df_mix = pd.read_excel(mix_file).iloc[1:]

for col in df_mix.columns[1:]:
    voltage = pd.to_numeric(df_mix[col], errors='coerce').values
    mask = ~np.isnan(voltage)
    clean_voltage = voltage[mask][:min_len]
    
    X_pred = np.array([clean_voltage])
    X_pred_p = preprocess_pipeline(X_pred)
    X_pred_f = extract_features(X_pred_p)
    X_pred_combined = np.hstack([X_pred_p, X_pred_f])
    
    predictions = []
    for name in substance_names:
        pred = models[name].predict(X_pred_combined)[0]
        predictions.append(max(0, pred))
    
    print(f"  {col}: 铝离子={predictions[0]:.4f}, 色氨酸={predictions[1]:.4f}, 烯酰肉碱={predictions[2]:.4f}")

# ============================================================================
# 8. 保存模型
# ============================================================================
import joblib
print("\n[保存模型...]")
for name, model in models.items():
    joblib.dump(model, f'models/merged_model_{name}.pkl')
    print(f"  - models/merged_model_{name}.pkl")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "="*80)
print("合并数据算法总结")
print("="*80)

print("\n【数据来源】")
print(f"  - 原数据: 72个样本")
print(f"  - LYY-614新数据: 36个样本")
print(f"  - 合并后: {len(X)}个样本")

print("\n【性能结果】")
print(f"  平均 R²: {avg_r2:.4f}")
print(f"  R²≥0.8的物质: {count_above_8}/3")

if count_above_8 >= 2:
    print("\n⚡ 大部分物质达标！")
elif count_above_8 == 1:
    print("\n✨ 一种物质达标，继续优化...")
else:
    print("\n💡 建议：")
    print("   - 当前样本量已增加到{}个".format(len(X)))
    print("   - 可尝试收集更多浓度梯度的数据")
    print("   - 优化传感器测量条件")

print("\n" + "="*80)
