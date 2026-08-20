"""
多元校正与信号解混算法
使用PLSR（偏最小二乘回归）和SVR（支持向量回归）进行浓度预测
"""

import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("多元校正与信号解混算法")
print("=" * 80)

# ============================================================================
# 1. 信号预处理模块（对抗胃肠道复杂环境的关键）
# ============================================================================
print("\n[步骤1] 信号预处理模块...")

def snv(X):
    """标准正态变量变换 (Standard Normal Variate)"""
    return (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)

def msc(X):
    """多元散射校正 (Multiplicative Scatter Correction)"""
    n_samples, n_features = X.shape
    # 计算参考光谱（所有样本的均值）
    reference = np.mean(X, axis=0)
    
    corrected = np.zeros_like(X)
    for i in range(n_samples):
        # 线性回归拟合
        coeffs = np.polyfit(reference, X[i, :], 1)
        corrected[i, :] = (X[i, :] - coeffs[1]) / coeffs[0]
    
    return corrected

def preprocess_waveforms(X):
    """预处理流水线：平滑滤波 + 归一化"""
    # 1. Savitzky-Golay 平滑 (窗口大小11, 多项式次数2)
    X_smoothed = savgol_filter(X, window_length=11, polyorder=2, axis=1)
    # 2. SNV 归一化
    X_preprocessed = snv(X_smoothed)
    return X_preprocessed

# ============================================================================
# 2. 数据加载模块
# ============================================================================
print("\n[步骤2] 数据加载与对齐...")

def load_waveform_data():
    """加载波形数据并转换为标准格式"""
    files = {
        '铝离子': r'c:\Users\28130\Desktop\lyy-6.12\铝离子.xlsx',
        '色氨酸': r'c:\Users\28130\Desktop\lyy-6.12\色氨酸.xlsx',
        '烯酰肉碱': r'c:\Users\28130\Desktop\lyy-6.12\烯酰肉碱.xlsx',
        '混合组': r'c:\Users\28130\Desktop\lyy-6.12\混合组.xlsx'
    }
    
    # 存储所有物质的数据
    all_data = {}
    
    for name, filepath in files.items():
        df = pd.read_excel(filepath)
        # 跳过第一行（标题）
        df = df.iloc[1:].reset_index(drop=True)
        
        # 第一列是时间
        time = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
        
        # 其他列是波形数据
        waveforms = {}
        
        if name == '混合组':
            # 混合组的列名格式：高烯, 高色, 高铝等
            waveform_cols = df.columns[1:]
            for col in waveform_cols:
                voltage = pd.to_numeric(df[col], errors='coerce').values
                waveforms[col] = {'time': time, 'voltage': voltage}
        else:
            # 单物质的列名格式：1mg/ml, 0.5mg/ml, 0.1mg/ml
            concentrations = ['1mg/ml', '0.5mg/ml', '0.1mg/ml']
            for conc in concentrations:
                conc_cols = [col for col in df.columns if conc in str(col)]
                conc_waveforms = []
                for col in conc_cols:
                    voltage = pd.to_numeric(df[col], errors='coerce').values
                    conc_waveforms.append(voltage)
                
                # 对齐长度
                if conc_waveforms:
                    min_len = min(len(v) for v in conc_waveforms)
                    waveforms[conc] = {
                        'time': time[:min_len],
                        'waveforms': np.array([v[:min_len] for v in conc_waveforms]),
                        'mean_waveform': np.mean([v[:min_len] for v in conc_waveforms], axis=0)
                    }
        
        all_data[name] = waveforms
    
    return all_data

# 加载数据
data = load_waveform_data()
print(f"成功加载 {len(data)} 组数据")

# ============================================================================
# 3. 构建训练数据集
# ============================================================================
print("\n[步骤3] 构建训练数据集...")

def build_training_dataset(data):
    """构建用于监督学习的训练数据集"""
    X = []  # 波形特征矩阵
    y_al = []  # 铝离子浓度
    y_trp = []  # 色氨酸浓度
    y_car = []  # 烯酰肉碱浓度
    labels = []  # 标签
    
    # 从单物质数据构建训练样本
    substances = ['铝离子', '色氨酸', '烯酰肉碱']
    conc_values = {'1mg/ml': 1.0, '0.5mg/ml': 0.5, '0.1mg/ml': 0.1}
    
    # 首先找到所有波形的最小长度
    all_waveforms = []
    for substance_name in substances:
        substance_data = data[substance_name]
        for conc_label, conc_data in substance_data.items():
            for waveform in conc_data['waveforms']:
                all_waveforms.append(waveform)
    
    min_len = min(len(w) for w in all_waveforms)
    print(f"所有波形对齐到最小长度: {min_len}")
    
    # 构建训练数据
    for substance_name in substances:
        substance_data = data[substance_name]
        
        for conc_label, conc_data in substance_data.items():
            conc_val = conc_values[conc_label]
            
            # 每个重复测量作为一个样本
            for waveform in conc_data['waveforms']:
                # 对齐长度
                X.append(waveform[:min_len])
                
                # 设置浓度标签（独热编码）
                if substance_name == '铝离子':
                    y_al.append(conc_val)
                    y_trp.append(0.0)
                    y_car.append(0.0)
                elif substance_name == '色氨酸':
                    y_al.append(0.0)
                    y_trp.append(conc_val)
                    y_car.append(0.0)
                else:
                    y_al.append(0.0)
                    y_trp.append(0.0)
                    y_car.append(conc_val)
                
                labels.append(f"{substance_name}_{conc_label}")
    
    X = np.array(X)
    Y = np.column_stack([y_al, y_trp, y_car])
    
    print(f"训练数据集形状: X={X.shape}, Y={Y.shape}")
    print(f"样本数: {len(labels)}")
    
    return X, Y, labels

X, Y, labels = build_training_dataset(data)

# 预处理
X_preprocessed = preprocess_waveforms(X)

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(
    X_preprocessed, Y, test_size=0.2, random_state=42
)

print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

# ============================================================================
# 4. 模型训练与评估
# ============================================================================
print("\n[步骤4] 模型训练与评估...")

substance_names = ['铝离子 (Al)', '色氨酸', '烯酰肉碱']
results = {}

# 4.1 PLSR（偏最小二乘回归）
print("\n" + "=" * 60)
print("方法A：偏最小二乘回归 (PLSR)")
print("=" * 60)

# 交叉验证选择最佳成分数
best_n_components = 1
best_r2 = -float('inf')

for n in range(2, min(10, X_train.shape[1])):
    pls = PLSRegression(n_components=n)
    scores = cross_val_score(pls, X_train, Y_train, cv=5, scoring='r2')
    avg_r2 = np.mean(scores)
    
    if avg_r2 > best_r2:
        best_r2 = avg_r2
        best_n_components = n

print(f"选择最佳主成分数: {best_n_components}")

pls = PLSRegression(n_components=best_n_components)
pls.fit(X_train, Y_train)
Y_pred_pls = pls.predict(X_test)

pls_results = {}
for i, name in enumerate(substance_names):
    rmse = np.sqrt(mean_squared_error(Y_test[:, i], Y_pred_pls[:, i]))
    r2 = r2_score(Y_test[:, i], Y_pred_pls[:, i])
    pls_results[name] = {'rmse': rmse, 'r2': r2}
    print(f"[{name}] RMSE: {rmse:.4f}, R²: {r2:.4f}")

results['PLSR'] = pls_results

# 4.2 SVR（支持向量回归）
print("\n" + "=" * 60)
print("方法B：支持向量回归 (SVR)")
print("=" * 60)

svr_results = {}

for i, name in enumerate(substance_names):
    # 网格搜索调参
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'kernel': ['rbf', 'linear']
    }
    
    svr = SVR()
    grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, Y_train[:, i])
    
    best_svr = grid_search.best_estimator_
    Y_pred_svr = best_svr.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(Y_test[:, i], Y_pred_svr))
    r2 = r2_score(Y_test[:, i], Y_pred_svr)
    svr_results[name] = {'rmse': rmse, 'r2': r2, 'model': best_svr}
    
    print(f"[{name}] RMSE: {rmse:.4f}, R²: {r2:.4f}")
    print(f"    最佳参数: {grid_search.best_params_}")

results['SVR'] = svr_results

# 4.3 随机森林（对比）
print("\n" + "=" * 60)
print("方法C：随机森林回归 (Random Forest)")
print("=" * 60)

rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, Y_train)
Y_pred_rf = rf.predict(X_test)

rf_results = {}
for i, name in enumerate(substance_names):
    rmse = np.sqrt(mean_squared_error(Y_test[:, i], Y_pred_rf[:, i]))
    r2 = r2_score(Y_test[:, i], Y_pred_rf[:, i])
    rf_results[name] = {'rmse': rmse, 'r2': r2}
    print(f"[{name}] RMSE: {rmse:.4f}, R²: {r2:.4f}")

results['RandomForest'] = rf_results

# ============================================================================
# 5. NNLS线性解混
# ============================================================================
print("\n[步骤5] NNLS线性解混...")

def build_standard_matrix(data):
    """构建标准物质波形矩阵"""
    standards = []
    names = []
    all_waves = []
    
    for name in ['铝离子', '色氨酸', '烯酰肉碱']:
        # 使用1mg/ml浓度的平均波形作为标准
        if '1mg/ml' in data[name]:
            mean_wave = data[name]['1mg/ml']['mean_waveform']
            all_waves.append(mean_wave)
            names.append(name)
    
    # 对齐长度
    min_len = min(len(w) for w in all_waves)
    standards = [w[:min_len] for w in all_waves]
    
    return np.array(standards), names

# 构建标准矩阵
X_standards, std_names = build_standard_matrix(data)

# 预处理标准波形
X_standards_p = preprocess_waveforms(X_standards)

# 测试NNLS解混
from scipy.optimize import nnls

print("\nNNLS线性解混测试（使用训练集中的混合样本）:")
print("-" * 40)

# 随机选择一个测试样本
test_idx = np.random.randint(0, len(X_test))
test_wave = X_test[test_idx]
true_conc = Y_test[test_idx]

# 使用NNLS解混
coeffs, _ = nnls(X_standards_p.T, test_wave)

print(f"真实浓度: {true_conc}")
print(f"NNLS解混系数: {coeffs}")
print(f"归一化解混浓度: {coeffs / np.sum(coeffs) if np.sum(coeffs) > 0 else coeffs}")

# ============================================================================
# 6. 混合组预测
# ============================================================================
print("\n[步骤6] 混合组预测...")

def predict_mixed_group(data, pls_model, svr_models):
    """预测混合组中各物质的浓度"""
    mixed_data = data['混合组']
    results = {}
    
    # 混合组列名映射
    col_map = {
        '铝离子': ['高铝', '高铝.1'],
        '色氨酸': ['高色', '高色.1'],
        '烯酰肉碱': ['高烯', '高烯.1']
    }
    
    for substance_name, cols in col_map.items():
        predictions_pls = []
        predictions_svr = []
        
        for col in cols:
            if col in mixed_data:
                voltage = mixed_data[col]['voltage']
                mask = ~np.isnan(voltage)
                clean_voltage = voltage[mask]
                
                if len(clean_voltage) > 10:
                    # 预处理
                    X_pred = np.array([clean_voltage])
                    X_pred_p = preprocess_waveforms(X_pred)
                    
                    # PLSR预测
                    pls_pred = pls_model.predict(X_pred_p)
                    predictions_pls.append(pls_pred[0])
                    
                    # SVR预测
                    svr_preds = []
                    for i, name in enumerate(substance_names):
                        svr_model = svr_models[name]['model']
                        svr_preds.append(svr_model.predict(X_pred_p)[0])
                    predictions_svr.append(svr_preds)
        
        if predictions_pls:
            avg_pls = np.mean(predictions_pls, axis=0)
            avg_svr = np.mean(predictions_svr, axis=0)
            
            results[substance_name] = {
                'PLSR预测': avg_pls,
                'SVR预测': avg_svr
            }
    
    return results

# 提取SVR模型
svr_models = {name: results['SVR'][name] for name in substance_names}

# 预测混合组
mixed_results = predict_mixed_group(data, pls, svr_models)

print("\n混合组预测结果:")
print("-" * 60)

for substance_name, preds in mixed_results.items():
    print(f"\n{substance_name}:")
    print(f"  PLSR预测: {preds['PLSR预测']}")
    print(f"  SVR预测: {preds['SVR预测']}")
    
    # 找到该物质对应的预测值
    idx = substance_names.index(f"{substance_name} (Al)" if substance_name == '铝离子' else substance_name)
    print(f"  最终预测浓度:")
    print(f"    PLSR: {preds['PLSR预测'][idx]:.4f} mg/ml")
    print(f"    SVR: {preds['SVR预测'][idx]:.4f} mg/ml")

# ============================================================================
# 7. 可视化结果
# ============================================================================
print("\n[步骤7] 生成可视化图表...")

fig = plt.figure(figsize=(16, 12))

# 7.1 模型性能对比
ax1 = plt.subplot(2, 3, 1)
models = ['PLSR', 'SVR', 'RandomForest']
r2_means = []

for model in models:
    r2s = [results[model][name]['r2'] for name in substance_names]
    r2_means.append(np.mean(r2s))

ax1.bar(models, r2_means, color=['blue', 'green', 'orange'])
ax1.set_ylabel('平均R²')
ax1.set_title('模型性能对比')
ax1.grid(True, alpha=0.3)

# 7.2 每种物质的模型对比
ax2 = plt.subplot(2, 3, 2)
x = np.arange(len(substance_names))
width = 0.25

plsr_r2 = [results['PLSR'][name]['r2'] for name in substance_names]
svr_r2 = [results['SVR'][name]['r2'] for name in substance_names]
rf_r2 = [results['RandomForest'][name]['r2'] for name in substance_names]

ax2.bar(x - width, plsr_r2, width, label='PLSR', alpha=0.8)
ax2.bar(x, svr_r2, width, label='SVR', alpha=0.8)
ax2.bar(x + width, rf_r2, width, label='RandomForest', alpha=0.8)

ax2.set_xlabel('物质')
ax2.set_ylabel('R²')
ax2.set_title('各物质模型性能')
ax2.set_xticks(x)
ax2.set_xticklabels([n.split(' ')[0] for n in substance_names], rotation=45)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 7.3 预处理后的波形对比
ax3 = plt.subplot(2, 3, 3)
for name in ['铝离子', '色氨酸', '烯酰肉碱']:
    mean_wave = data[name]['1mg/ml']['mean_waveform']
    time = data[name]['1mg/ml']['time']
    
    # 预处理
    X_plot = np.array([mean_wave])
    X_plot_p = preprocess_waveforms(X_plot)
    
    ax3.plot(time[:len(X_plot_p[0])], X_plot_p[0], label=name, linewidth=1.5)

ax3.set_xlabel('时间 (s)')
ax3.set_ylabel('归一化电压')
ax3.set_title('预处理后的波形对比')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 7.4 PLSR预测结果
ax4 = plt.subplot(2, 3, 4)
for i, name in enumerate(substance_names):
    ax4.scatter(Y_test[:, i], Y_pred_pls[:, i], alpha=0.6, label=name.split(' ')[0])

ax4.plot([0, 1.2], [0, 1.2], 'r--', linewidth=2)
ax4.set_xlabel('真实浓度')
ax4.set_ylabel('预测浓度')
ax4.set_title('PLSR预测结果')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 7.5 SVR预测结果
ax5 = plt.subplot(2, 3, 5)
for i, name in enumerate(substance_names):
    Y_pred = svr_results[name]['model'].predict(X_test)
    ax5.scatter(Y_test[:, i], Y_pred, alpha=0.6, label=name.split(' ')[0])

ax5.plot([0, 1.2], [0, 1.2], 'r--', linewidth=2)
ax5.set_xlabel('真实浓度')
ax5.set_ylabel('预测浓度')
ax5.set_title('SVR预测结果')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 7.6 特征重要性（PLSR权重）
ax6 = plt.subplot(2, 3, 6)
# 取第一个成分的权重作为特征重要性
weights = np.abs(pls.x_weights_[:, 0])
ax6.plot(weights, linewidth=1.5)
ax6.set_xlabel('特征位置')
ax6.set_ylabel('权重绝对值')
ax6.set_title('PLSR第一主成分权重')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/multivariate_calibration.png', dpi=300, bbox_inches='tight')
print("  多元校正分析图已保存: plots/multivariate_calibration.png")

# ============================================================================
# 8. 保存模型
# ============================================================================
print("\n[步骤8] 保存模型...")

import joblib

# 保存PLSR模型
joblib.dump(pls, 'models/plsr_model.pkl')

# 保存SVR模型
for name, svr_result in svr_results.items():
    joblib.dump(svr_result['model'], f'models/svr_model_{name.split(" ")[0]}.pkl')

print("  PLSR模型: models/plsr_model.pkl")
print("  SVR模型已保存到 models/ 目录")

# ============================================================================
# 9. 总结
# ============================================================================
print("\n" + "=" * 80)
print("多元校正算法总结")
print("=" * 80)

print("\n【模型性能对比】")
print("-" * 60)
print(f"{'模型':<20} {'铝离子R²':<15} {'色氨酸R²':<15} {'烯酰肉碱R²':<15} {'平均R²':<10}")
print("-" * 60)

for model in ['PLSR', 'SVR', 'RandomForest']:
    r2s = [results[model][name]['r2'] for name in substance_names]
    avg_r2 = np.mean(r2s)
    print(f"{model:<20} {r2s[0]:<15.4f} {r2s[1]:<15.4f} {r2s[2]:<15.4f} {avg_r2:<10.4f}")

print("\n【混合组预测结果】")
for substance_name, preds in mixed_results.items():
    idx = substance_names.index(f"{substance_name} (Al)" if substance_name == '铝离子' else substance_name)
    print(f"  {substance_name}: PLSR={preds['PLSR预测'][idx]:.4f} mg/ml, SVR={preds['SVR预测'][idx]:.4f} mg/ml")

print("\n【结论】")
avg_plsr_r2 = np.mean([results['PLSR'][name]['r2'] for name in substance_names])
avg_svr_r2 = np.mean([results['SVR'][name]['r2'] for name in substance_names])

if avg_plsr_r2 > avg_svr_r2:
    print("  PLSR表现更好，说明数据中线性关系占主导")
else:
    print("  SVR表现更好，说明数据中存在非线性关系")

print("\n" + "=" * 80)
