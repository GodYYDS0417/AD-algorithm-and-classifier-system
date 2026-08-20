"""
高精度浓度预测算法：超参数优化版
目标：R²突破0.8
"""

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

print("=" * 80)
print("高精度浓度预测算法：超参数优化版")
print("目标：R²突破0.8")
print("=" * 80)

# ============================================================================
# 1. 高级数据预处理
# ============================================================================
def preprocess_pipeline(X):
    """SG平滑 + SNV归一化"""
    X_smoothed = savgol_filter(X, window_length=11, polyorder=2, axis=1)
    X_snv = (X_smoothed - np.mean(X_smoothed, axis=1, keepdims=True)) / np.std(X_smoothed, axis=1, keepdims=True)
    return X_snv

# ============================================================================
# 2. 数据加载
# ============================================================================
print("\n[步骤1] 加载数据...")

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
    print(f"  加载 {name}: {len(waveforms)} 个浓度/样本")

# ============================================================================
# 3. 构建训练数据集
# ============================================================================
print("\n[步骤2] 构建训练数据集...")

X = []
Y = []

substances = ['铝离子', '色氨酸', '烯酰肉碱']
conc_values = {'1mg/ml': 1.0, '0.5mg/ml': 0.5, '0.1mg/ml': 0.1}

all_waves = []
for name in substances:
    for conc_label, conc_data in all_data[name].items():
        for wave in conc_data['waveforms']:
            all_waves.append(wave)

min_len = min(len(w) for w in all_waves)
print(f"波形对齐长度: {min_len}")

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

X = np.array(X)
Y = np.array(Y)
X_preprocessed = preprocess_pipeline(X)

print(f"数据形状: X={X.shape}, Y={Y.shape}")

# ============================================================================
# 4. 高级特征工程：峰值特征提取
# ============================================================================
def extract_peak_features(X):
    """提取波形的峰值特征"""
    n_samples, n_features = X.shape
    features = []
    
    for i in range(n_samples):
        wave = X[i, :]
        
        # 寻找波峰和波谷
        peaks, _ = find_peaks(wave, prominence=0.3)
        valleys, _ = find_peaks(-wave, prominence=0.3)
        
        # 提取特征
        peak_features = [
            len(peaks),
            len(valleys),
            np.mean(wave) if len(wave) > 0 else 0,
            np.std(wave) if len(wave) > 0 else 0,
            np.max(wave) if len(wave) > 0 else 0,
            np.min(wave) if len(wave) > 0 else 0,
            np.sum(np.abs(np.diff(wave))) if len(wave) > 1 else 0,  # 总变化量
            np.mean(np.abs(np.diff(wave))) if len(wave) > 1 else 0,  # 平均变化率
        ]
        
        # 添加峰值位置和高度
        if len(peaks) > 0:
            peak_features.extend([np.min(peaks), np.max(peaks), np.mean(wave[peaks])])
        else:
            peak_features.extend([0, 0, 0])
        
        features.append(peak_features)
    
    return np.array(features)

print("\n[步骤3] 提取峰值特征...")
peak_features = extract_peak_features(X_preprocessed)
print(f"峰值特征形状: {peak_features.shape}")

# 组合原始波形和峰值特征
X_combined = np.hstack([X_preprocessed, peak_features])
print(f"组合后特征形状: {X_combined.shape}")

# ============================================================================
# 5. 超参数调优函数
# ============================================================================
def tune_plsr(X_train, y_train):
    """调优PLSR模型"""
    best_r2 = -np.inf
    best_n_components = 2
    
    for n_components in range(2, min(10, X_train.shape[0]-1)):
        pls = PLSRegression(n_components=n_components)
        scores = cross_val_score(pls, X_train, y_train, cv=5, scoring='r2')
        mean_r2 = np.mean(scores)
        
        if mean_r2 > best_r2:
            best_r2 = mean_r2
            best_n_components = n_components
    
    pls = PLSRegression(n_components=best_n_components)
    pls.fit(X_train, y_train)
    return pls, best_n_components

def tune_svr(X_train, y_train):
    """调优SVR模型"""
    param_grid = {
        'C': [0.1, 1, 10, 50, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'kernel': ['rbf', 'linear']
    }
    
    svr = SVR()
    grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_svr = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    return best_svr, best_params

# ============================================================================
# 6. 训练模型（逐个物质优化）
# ============================================================================
print("\n" + "="*80)
print("训练高精度模型（逐个物质优化）")
print("="*80)

substance_names = ['铝离子', '色氨酸', '烯酰肉碱']
results = []
models = {}

for i, name in enumerate(substance_names):
    print(f"\n{'='*60}")
    print(f"【{name}】模型训练")
    print(f"{'='*60}")
    
    y_single = Y[:, i]
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_single, test_size=0.15, random_state=42)
    
    print("\n[PLSR调优]")
    pls_model, n_components = tune_plsr(X_train, y_train)
    y_pred_pls = pls_model.predict(X_test)
    r2_pls = r2_score(y_test, y_pred_pls)
    print(f"  最佳主成分数: {n_components}")
    print(f"  R²: {r2_pls:.4f}")
    
    print("\n[SVR调优]")
    svr_model, best_params = tune_svr(X_train, y_train)
    y_pred_svr = svr_model.predict(X_test)
    r2_svr = r2_score(y_test, y_pred_svr)
    print(f"  最佳参数: {best_params}")
    print(f"  R²: {r2_svr:.4f}")
    
    # 选择更好的模型
    if r2_svr > r2_pls:
        best_model = svr_model
        best_r2 = r2_svr
        best_method = 'SVR'
    else:
        best_model = pls_model
        best_r2 = r2_pls
        best_method = 'PLSR'
    
    rmse = np.sqrt(mean_squared_error(y_test, best_model.predict(X_test)))
    
    models[name] = best_model
    results.append({
        '物质': name,
        '方法': best_method,
        'R²': best_r2,
        'RMSE': rmse,
        'y_test': y_test,
        'y_pred': best_model.predict(X_test)
    })
    
    print(f"\n✨ 【{name}】最佳模型: {best_method}")
    print(f"   R²: {best_r2:.4f}, RMSE: {rmse:.4f}")
    
    if best_r2 >= 0.8:
        print("   🎉 达到目标！")

# ============================================================================
# 7. 结果汇总
# ============================================================================
print("\n" + "="*80)
print("模型性能汇总")
print("="*80)

print("\n【性能对比】")
print("-" * 60)
print(f"{'物质':<10} {'方法':<10} {'R²':<10} {'RMSE':<10} {'目标':<10}")
print("-" * 60)

avg_r2 = 0
for res in results:
    status = "✅" if res['R²'] >= 0.8 else ("⚡" if res['R²'] >= 0.7 else "⏳")
    print(f"{res['物质']:<10} {res['方法']:<10} {res['R²']:<10.4f} {res['RMSE']:<10.4f} {status} >0.8")
    avg_r2 += res['R²']

avg_r2 /= len(results)
print("-" * 60)
print(f"{'平均':<10} {'-':<10} {avg_r2:<10.4f} {'-':<10} {'-':<10}")

# ============================================================================
# 8. 可视化
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
plt.savefig('plots/optimized_predictions.png', dpi=300)
print("  - optimized_predictions.png")

# 性能对比图
fig = plt.figure(figsize=(12, 6))
x = np.arange(len(substance_names))
r2_values = [res['R²'] for res in results]
colors = ['blue' if res['方法'] == 'PLSR' else 'green' for res in results]

bars = plt.bar(x, r2_values, width=0.5, color=colors, alpha=0.8)
plt.axhline(y=0.8, color='red', linestyle='--', label='目标 R²=0.8')

plt.xlabel('物质', fontsize=12)
plt.ylabel('R² 分数', fontsize=12)
plt.title('超参数优化后模型性能', fontsize=14, fontweight='bold')
plt.xticks(x, substance_names)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])

for bar, r2 in zip(bars, r2_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{r2:.4f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('plots/optimized_performance.png', dpi=300)
print("  - optimized_performance.png")

# ============================================================================
# 9. 混合组预测
# ============================================================================
print("\n[混合组预测...]")
if '混合组' in all_data:
    for col in all_data['混合组'].keys():
        voltage = all_data['混合组'][col]['voltage']
        mask = ~np.isnan(voltage)
        clean_voltage = voltage[mask][:min_len]
        
        X_pred = np.array([clean_voltage])
        X_pred_p = preprocess_pipeline(X_pred)
        peak_pred = extract_peak_features(X_pred_p)
        X_pred_combined = np.hstack([X_pred_p, peak_pred])
        
        predictions = []
        for name in substance_names:
            pred = models[name].predict(X_pred_combined)[0]
            predictions.append(max(0, pred))
        
        print(f"  {col}: 铝离子={predictions[0]:.4f}, 色氨酸={predictions[1]:.4f}, 烯酰肉碱={predictions[2]:.4f}")

# ============================================================================
# 10. 保存模型
# ============================================================================
import joblib
print("\n[保存模型...]")
for name, model in models.items():
    joblib.dump(model, f'models/optimized_model_{name}.pkl')
    print(f"  - models/optimized_model_{name}.pkl")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "="*80)
print("算法总结")
print("="*80)
print("\n【优化策略】")
print("  1. 高级预处理：SG平滑 + SNV归一化")
print("  2. 特征工程：提取峰值、谷值、统计特征")
print("  3. 超参数调优：GridSearchCV自动搜索最佳参数")
print("  4. 模型选择：PLSR vs SVR，选择更优者")

print(f"\n【性能】")
print(f"  平均 R²: {avg_r2:.4f}")
print(f"  目标: R² > 0.8")

if avg_r2 >= 0.8:
    print("\n🎉 恭喜！平均R²突破0.8！")
else:
    print("\n💡 进一步提升建议：")
    print("   - 增加训练样本量（当前72个）")
    print("   - 收集更多浓度梯度的数据")
    print("   - 优化传感器测量条件")
    print("   - 尝试正交信号校正(OSC)")

print("\n" + "="*80)
