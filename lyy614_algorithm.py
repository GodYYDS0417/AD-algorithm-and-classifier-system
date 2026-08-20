"""
使用LYY-614新数据训练高精度模型
目标：所有物质R²突破0.8
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
print("使用LYY-614新数据训练高精度模型")
print("目标：所有物质R²突破0.8")
print("=" * 80)

# ============================================================================
# 1. 高级预处理
# ============================================================================
def preprocess_pipeline(X):
    """SG平滑 + SNV归一化 + 一阶导数"""
    X_smoothed = savgol_filter(X, window_length=11, polyorder=2, axis=1)
    X_snv = (X_smoothed - np.mean(X_smoothed, axis=1, keepdims=True)) / np.std(X_smoothed, axis=1, keepdims=True)
    # 添加一阶导数特征
    X_deriv = np.diff(X_snv, axis=1)
    X_deriv = np.column_stack([X_deriv, np.zeros((X_deriv.shape[0], 1))])
    X_final = np.hstack([X_snv, X_deriv])
    return X_final

# ============================================================================
# 2. 加载LYY-614新数据
# ============================================================================
print("\n[步骤1] 加载LYY-614新数据...")

data_dir = r'c:\Users\28130\Desktop\lyy-6.12\LYY-614'

files_new = {
    '铝离子': os.path.join(data_dir, '铝离子.xlsx'),
    '色氨酸': os.path.join(data_dir, '色氨酸.xlsx'),
    '烯酰肉碱': os.path.join(data_dir, '烯酰肉碱.xlsx'),
}

files_old = {
    '铝离子': r'c:\Users\28130\Desktop\lyy-6.12\铝离子.xlsx',
    '色氨酸': r'c:\Users\28130\Desktop\lyy-6.12\色氨酸.xlsx',
    '烯酰肉碱': r'c:\Users\28130\Desktop\lyy-6.12\烯酰肉碱.xlsx',
    '混合组': r'c:\Users\28130\Desktop\lyy-6.12\混合组.xlsx'
}

# 优先使用新数据，如果不存在则使用旧数据
all_data = {}
for name in ['铝离子', '色氨酸', '烯酰肉碱']:
    filepath = files_new[name] if os.path.exists(files_new[name]) else files_old[name]
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
            min_len = min(len(v) for v in conc_waveforms)
            waveforms[conc] = {
                'time': time[:min_len],
                'waveforms': np.array([v[:min_len] for v in conc_waveforms]),
                'mean_waveform': np.mean([v[:min_len] for v in conc_waveforms], axis=0)
            }
    
    all_data[name] = waveforms
    source = "新数据" if os.path.exists(files_new[name]) else "旧数据"
    print(f"  加载 {name} ({source}): {len(waveforms)} 个浓度")

# 加载混合组
df_mix = pd.read_excel(files_old['混合组']).iloc[1:]
time_mix = pd.to_numeric(df_mix.iloc[:, 0], errors='coerce').values
mix_waveforms = {}
for col in df_mix.columns[1:]:
    voltage = pd.to_numeric(df_mix[col], errors='coerce').values
    mix_waveforms[col] = {'time': time_mix, 'voltage': voltage}
all_data['混合组'] = mix_waveforms

# ============================================================================
# 3. 构建训练数据集（合并新旧数据）
# ============================================================================
print("\n[步骤2] 构建训练数据集...")

X = []
Y = []
sources = []

substances = ['铝离子', '色氨酸', '烯酰肉碱']
conc_values = {'1mg/ml': 1.0, '0.5mg/ml': 0.5, '0.1mg/ml': 0.1}

# 收集所有波形
all_waves = []
for name in substances:
    for conc_label, conc_data in all_data[name].items():
        for wave in conc_data['waveforms']:
            all_waves.append(wave)

min_len = min(len(w) for w in all_waves)
print(f"波形对齐长度: {min_len}")

# 构建训练数据
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

print(f"训练集形状: X={X.shape}, Y={Y.shape}")
print(f"样本数量: {len(X)}")

# 预处理
X_preprocessed = preprocess_pipeline(X)
print(f"预处理后特征数: {X_preprocessed.shape[1]}")

# ============================================================================
# 4. 高级特征选择（基于互信息）
# ============================================================================
def select_features_by_importance(X, y, n_features=60):
    """基于随机森林重要性选择特征"""
    from sklearn.ensemble import RandomForestRegressor
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:n_features]
    
    return indices

# ============================================================================
# 5. 训练模型（超参数优化）
# ============================================================================
print("\n" + "="*80)
print("训练高精度模型（LYY-614数据）")
print("="*80)

substance_names = ['铝离子', '色氨酸', '烯酰肉碱']
results = []
models = {}
selected_indices_dict = {}

for i, name in enumerate(substance_names):
    print(f"\n{'='*60}")
    print(f"【{name}】模型训练")
    print(f"{'='*60}")
    
    y_single = Y[:, i]
    
    # 特征选择
    selected_indices = select_features_by_importance(X_preprocessed, y_single, n_features=60)
    selected_indices_dict[name] = selected_indices
    X_selected = X_preprocessed[:, selected_indices]
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y_single, test_size=0.2, random_state=42)
    
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # PLSR调优
    print("\n[PLSR调优]")
    best_pls_r2 = -np.inf
    best_pls_n = 2
    for n in range(2, min(8, X_train.shape[0]-1)):
        pls = PLSRegression(n_components=n)
        scores = cross_val_score(pls, X_train, y_train, cv=5, scoring='r2')
        mean_r2 = np.mean(scores)
        if mean_r2 > best_pls_r2:
            best_pls_r2 = mean_r2
            best_pls_n = n
    
    pls_model = PLSRegression(n_components=best_pls_n)
    pls_model.fit(X_train, y_train)
    y_pred_pls = pls_model.predict(X_test)
    r2_pls = r2_score(y_test, y_pred_pls)
    print(f"  最佳主成分数: {best_pls_n}, R²: {r2_pls:.4f}")
    
    # SVR调优
    print("\n[SVR调优]")
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'kernel': ['rbf', 'linear']
    }
    
    svr = SVR()
    grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    svr_model = grid_search.best_estimator_
    y_pred_svr = svr_model.predict(X_test)
    r2_svr = r2_score(y_test, y_pred_svr)
    print(f"  最佳参数: {grid_search.best_params_}, R²: {r2_svr:.4f}")
    
    # 选择最佳模型
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
        print("   🎉 达到目标 R² > 0.8！")
    elif best_r2 >= 0.7:
        print("   ⚡ 接近目标！")

# ============================================================================
# 6. 结果汇总
# ============================================================================
print("\n" + "="*80)
print("模型性能汇总（LYY-614数据）")
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
# 7. 可视化
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
plt.savefig('plots/lyy614_predictions.png', dpi=300)
print("  - lyy614_predictions.png")

# 性能对比图
fig = plt.figure(figsize=(12, 6))
x = np.arange(len(substance_names))
r2_values = [res['R²'] for res in results]
colors = ['blue' if res['方法'] == 'PLSR' else 'green' for res in results]

bars = plt.bar(x, r2_values, width=0.5, color=colors, alpha=0.8)
plt.axhline(y=0.8, color='red', linestyle='--', label='目标 R²=0.8')

plt.xlabel('物质', fontsize=12)
plt.ylabel('R² 分数', fontsize=12)
plt.title('LYY-614新数据模型性能', fontsize=14, fontweight='bold')
plt.xticks(x, substance_names)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])

for bar, r2 in zip(bars, r2_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{r2:.4f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('plots/lyy614_performance.png', dpi=300)
print("  - lyy614_performance.png")

# ============================================================================
# 8. 混合组预测
# ============================================================================
print("\n[混合组预测...]")
if '混合组' in all_data:
    for col in all_data['混合组'].keys():
        voltage = all_data['混合组'][col]['voltage']
        mask = ~np.isnan(voltage)
        clean_voltage = voltage[mask][:min_len]
        
        X_pred = np.array([clean_voltage])
        X_pred_p = preprocess_pipeline(X_pred)
        
        predictions = []
        for name in substance_names:
            X_pred_selected = X_pred_p[:, selected_indices_dict[name]]
            pred = models[name].predict(X_pred_selected)[0]
            predictions.append(max(0, pred))
        
        print(f"  {col}: 铝离子={predictions[0]:.4f}, 色氨酸={predictions[1]:.4f}, 烯酰肉碱={predictions[2]:.4f}")

# ============================================================================
# 9. 保存模型
# ============================================================================
import joblib
print("\n[保存模型...]")
for name, model in models.items():
    joblib.dump(model, f'models/lyy614_model_{name}.pkl')
    joblib.dump(selected_indices_dict[name], f'models/lyy614_indices_{name}.npy')
    print(f"  - models/lyy614_model_{name}.pkl")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "="*80)
print("LYY-614新数据训练总结")
print("="*80)

print("\n【数据来源】")
print("  - 使用LYY-614目录下的新数据")
print("  - 样本数量: {}".format(len(X)))

print("\n【优化策略】")
print("  1. 高级预处理：SG平滑 + SNV + 一阶导数")
print("  2. 特征选择：基于随机森林重要性选择60个关键特征")
print("  3. 超参数调优：PLSR主成分数搜索 + SVR GridSearch")
print("  4. 模型选择：自动选择PLSR或SVR中性能更优者")

print(f"\n【性能结果】")
print(f"  平均 R²: {avg_r2:.4f}")
print(f"  R²≥0.8的物质: {count_above_8}/3")

if count_above_8 == 3:
    print("\n🎉 🎉 🎉 所有物质R²均突破0.8！")
elif count_above_8 >= 1:
    print(f"\n⚡ {count_above_8}种物质达标，继续优化剩余{3-count_above_8}种...")
else:
    print("\n💡 建议：")
    print("   - 增加训练样本量")
    print("   - 尝试正交信号校正(OSC)")
    print("   - 收集更多浓度梯度的数据")

print("\n" + "="*80)
