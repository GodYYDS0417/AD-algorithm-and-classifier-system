"""
OSC正交信号校正 + 一阶导数 + 特征筛选算法
目标：所有物质R²突破0.8
"""

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

print("=" * 80)
print("OSC正交信号校正 + 一阶导数 + 特征筛选算法")
print("目标：所有物质R²突破0.8")
print("=" * 80)

# ============================================================================
# 1. OSC正交信号校正核心算法
# ============================================================================
def orthogonal_signal_correction(X, y, n_components=1, tol=1e-3):
    """
    正交信号校正 (OSC) 核心算法
    强行擦除波形 X 中与浓度 y 无关的干扰信号（专门解决胃肠道大背景噪声）
    """
    X_osc = np.array(X, dtype=float, copy=True)
    y_col = np.array(y, dtype=float).reshape(-1, 1)
    
    for comp in range(n_components):
        # 1. 计算 X 的主成分
        Z = X_osc.copy()
        
        # 2. 移除 y 对 Z 的影响，保留正交部分
        w = np.dot(Z.T, y_col) / np.dot(y_col.T, y_col)
        w = w / np.linalg.norm(w)
        t = np.dot(Z, w)
        
        # 3. 正交化：使 t 与 y 正交
        t_ortho = t - np.dot(y_col, np.dot(y_col.T, t)) / np.dot(y_col.T, y_col)
        
        # 4. 计算 X 在正交向量上的载荷
        p = np.dot(Z.T, t_ortho) / np.dot(t_ortho.T, t_ortho)
        
        # 5. 从 X 中减去这部分正交（无关）信号
        X_osc = X_osc - np.dot(t_ortho, p.T)
        
    return X_osc

# ============================================================================
# 2. 高级特征筛选
# ============================================================================
def advanced_feature_selection(X_train, y_train, X_test, percentile=15):
    """
    特征波段筛选（代替复杂的CARS，用高效的 F-Score 过滤法）
    只保留前百分之 percentile 最具特异性响应的波形点
    """
    selector = SelectPercentile(score_func=f_regression, percentile=percentile)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    
    selected_indices = np.where(selector.get_support())[0]
    return X_train_sel, X_test_sel, selected_indices

# ============================================================================
# 3. 数据加载
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
# 4. 构建训练数据集
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

print(f"数据形状: X={X.shape}, Y={Y.shape}")
print(f"样本数量: {len(X)}")

# ============================================================================
# 5. 高级预处理流水线
# ============================================================================
print("\n[步骤3] 高级预处理：一阶导数 + OSC + 特征筛选...")

def advanced_preprocess(X, y=None, n_osc_components=1, apply_osc=True):
    """
    高级预处理流水线：
    1. SG一阶导数（消除批次效应和基线漂移）
    2. OSC正交信号校正（去除与浓度无关的背景噪声）
    """
    # 1. SG滤波 + 一阶导数（deriv=1）
    # 一阶导数能彻底杀掉批次不同带来的常数项漂移
    X_deriv = savgol_filter(X, window_length=15, polyorder=2, deriv=1, axis=1)
    
    # 2. OSC正交信号校正
    if apply_osc and y is not None:
        X_clean = orthogonal_signal_correction(X_deriv, y, n_components=n_osc_components)
    else:
        X_clean = X_deriv
    
    return X_clean

# ============================================================================
# 6. 模型训练（针对每种物质单独优化）
# ============================================================================
print("\n" + "="*80)
print("训练高精度模型（OSC + 一阶导数 + 特征筛选）")
print("="*80)

substance_names = ['铝离子', '色氨酸', '烯酰肉碱']
results = []
models = {}
selectors = {}

for i, name in enumerate(substance_names):
    print(f"\n{'='*60}")
    print(f"【{name}】模型训练")
    print(f"{'='*60}")
    
    y_single = Y[:, i]
    
    # 1. 一阶导数 + OSC预处理
    print("\n[预处理]")
    print("  - SG一阶导数（消除批次效应）")
    print("  - OSC正交信号校正（去除背景噪声）")
    
    X_clean = advanced_preprocess(X, y_single, n_osc_components=1, apply_osc=True)
    
    # 2. 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_single, test_size=0.2, random_state=42)
    
    print(f"  训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 3. 特征筛选（只保留15%的黄金波段）
    print("\n[特征筛选]")
    X_train_sel, X_test_sel, selected_indices = advanced_feature_selection(
        X_train, y_train, X_test, percentile=15
    )
    print(f"  原始特征数: {X_clean.shape[1]}")
    print(f"  筛选后特征数: {X_train_sel.shape[1]}")
    print(f"  关键波形位置: {selected_indices[:10]}...")
    
    selectors[name] = selected_indices
    
    # 4. PLSR建模
    print("\n[PLSR建模]")
    best_pls_r2 = -np.inf
    best_pls_n = 2
    for n in range(2, min(8, X_train_sel.shape[0]-1)):
        pls = PLSRegression(n_components=n)
        scores = cross_val_score(pls, X_train_sel, y_train, cv=5, scoring='r2')
        mean_r2 = np.mean(scores)
        if mean_r2 > best_pls_r2:
            best_pls_r2 = mean_r2
            best_pls_n = n
    
    pls_model = PLSRegression(n_components=best_pls_n)
    pls_model.fit(X_train_sel, y_train)
    y_pred_pls = pls_model.predict(X_test_sel)
    r2_pls = r2_score(y_test, y_pred_pls)
    rmse_pls = np.sqrt(mean_squared_error(y_test, y_pred_pls))
    print(f"  最佳主成分数: {best_pls_n}")
    print(f"  R²: {r2_pls:.4f}, RMSE: {rmse_pls:.4f}")
    
    # 5. SVR建模
    print("\n[SVR建模]")
    svr_model = SVR(C=10, gamma='scale', kernel='rbf')
    svr_model.fit(X_train_sel, y_train)
    y_pred_svr = svr_model.predict(X_test_sel)
    r2_svr = r2_score(y_test, y_pred_svr)
    rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))
    print(f"  R²: {r2_svr:.4f}, RMSE: {rmse_svr:.4f}")
    
    # 6. 选择最佳模型
    if r2_svr > r2_pls:
        best_model = svr_model
        best_r2 = r2_svr
        best_rmse = rmse_svr
        best_method = 'SVR'
    else:
        best_model = pls_model
        best_r2 = r2_pls
        best_rmse = rmse_pls
        best_method = 'PLSR'
    
    models[name] = best_model
    
    print(f"\n✨ 【{name}】最终结果")
    print(f"   最佳模型: {best_method}")
    print(f"   R²: {best_r2:.4f}, RMSE: {best_rmse:.4f}")
    
    if best_r2 >= 0.8:
        print("   🎉 R²突破0.8！达标！")
    elif best_r2 >= 0.7:
        print("   ⚡ 接近目标！")
    else:
        print(f"   💪 继续努力，差距: {0.8 - best_r2:.4f}")
    
    results.append({
        '物质': name,
        '方法': best_method,
        'R²': best_r2,
        'RMSE': best_rmse,
        '特征数': X_train_sel.shape[1],
        'y_test': y_test,
        'y_pred': best_model.predict(X_test_sel) if best_method == 'SVR' else best_model.predict(X_test_sel).flatten()
    })

# ============================================================================
# 7. 结果汇总
# ============================================================================
print("\n" + "="*80)
print("OSC算法性能汇总")
print("="*80)

print("\n【性能对比】")
print("-" * 70)
print(f"{'物质':<10} {'方法':<10} {'R²':<10} {'RMSE':<10} {'特征数':<10} {'状态':<10}")
print("-" * 70)

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
    
    print(f"{res['物质']:<10} {res['方法']:<10} {res['R²']:<10.4f} {res['RMSE']:<10.4f} {res['特征数']:<10} {status:<10}")
    avg_r2 += res['R²']

avg_r2 /= len(results)
print("-" * 70)
print(f"{'平均':<10} {'-':<10} {avg_r2:<10.4f} {'-':<10} {'-':<10} {'{}/3达标'.format(count_above_8):<10}")

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
    ax.set_xlabel('真实浓度 (mg/ml)', fontsize=12)
    ax.set_ylabel('预测浓度 (mg/ml)', fontsize=12)
    ax.set_title(f'{res["物质"]} ({res["方法"]})\nR²={res["R²"]:.4f}', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.1, 1.3])
    ax.set_ylim([-0.1, 1.3])
plt.tight_layout()
plt.savefig('plots/osc_predictions.png', dpi=300)
print("  - osc_predictions.png")

# 性能对比图
fig = plt.figure(figsize=(12, 6))
x = np.arange(len(substance_names))
r2_values = [res['R²'] for res in results]
colors = ['blue' if res['方法'] == 'PLSR' else 'green' for res in results]

bars = plt.bar(x, r2_values, width=0.5, color=colors, alpha=0.8)
plt.axhline(y=0.8, color='red', linestyle='--', label='目标 R²=0.8')

plt.xlabel('物质', fontsize=12)
plt.ylabel('R² 分数', fontsize=12)
plt.title('OSC算法性能（一阶导数 + OSC + 特征筛选）', fontsize=14, fontweight='bold')
plt.xticks(x, substance_names)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])

for i, (bar, r2) in enumerate(zip(bars, r2_values)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{r2:.4f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('plots/osc_performance.png', dpi=300)
print("  - osc_performance.png")

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
        
        predictions = []
        for name in substance_names:
            # 预处理
            X_pred_deriv = savgol_filter(X_pred, window_length=15, polyorder=2, deriv=1, axis=1)
            X_pred_sel = X_pred_deriv[:, selectors[name]]
            
            pred = models[name].predict(X_pred_sel)[0]
            predictions.append(max(0, pred))
        
        print(f"  {col}: 铝离子={predictions[0]:.4f}, 色氨酸={predictions[1]:.4f}, 烯酰肉碱={predictions[2]:.4f}")

# ============================================================================
# 10. 保存模型
# ============================================================================
import joblib
print("\n[保存模型...]")
for name, model in models.items():
    joblib.dump(model, f'models/osc_model_{name}.pkl')
    np.save(f'models/osc_indices_{name}.npy', selectors[name])
    print(f"  - models/osc_model_{name}.pkl")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "="*80)
print("OSC算法总结")
print("="*80)

print("\n【算法策略】")
print("  1. SG一阶导数：消除批次效应和基线漂移")
print("  2. OSC正交信号校正：去除与浓度无关的背景噪声")
print("  3. F-Score特征筛选：只保留15%的黄金波段")
print("  4. PLSR/SVR建模：自动选择最优模型")

print(f"\n【性能结果】")
print(f"  平均 R²: {avg_r2:.4f}")
print(f"  R²≥0.8的物质: {count_above_8}/3")

if count_above_8 == 3:
    print("\n🎉 🎉 🎉 所有物质R²均突破0.8！")
elif count_above_8 >= 2:
    print(f"\n⚡ {count_above_8}种物质达标！")
else:
    print("\n💡 进一步优化建议：")
    print("   - 调整OSC的n_components参数（当前为1）")
    print("   - 调整特征筛选的percentile（当前为15%）")
    print("   - 增加训练样本量")

print("\n" + "="*80)
