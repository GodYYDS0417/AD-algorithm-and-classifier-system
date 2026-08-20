"""
进阶提升算法：特征波段筛选 + Stacking融合
目标：将R²从0.7提升到0.8以上
"""

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

print("=" * 80)
print("进阶提升算法：特征波段筛选 + Stacking融合")
print("目标：R²突破0.8")
print("=" * 80)

# ============================================================================
# 1. 高级数据预处理（SG平滑 + SNV归一化）
# ============================================================================
def preprocess_pipeline(X):
    """SG平滑 + 标准正态变量变换(SNV)"""
    X_smoothed = savgol_filter(X, window_length=7, polyorder=2, axis=1)
    X_snv = (X_smoothed - np.mean(X_smoothed, axis=1, keepdims=True)) / np.std(X_smoothed, axis=1, keepdims=True)
    return X_snv

# ============================================================================
# 2. 数据加载（支持新数据目录）
# ============================================================================
print("\n[步骤1] 加载新数据...")

data_dir = r'c:\Users\28130\Desktop\lyy-6.12\LYY-614'

# 使用原始数据（样本量更大，更稳定）
files = {
    '铝离子': r'c:\Users\28130\Desktop\lyy-6.12\铝离子.xlsx',
    '色氨酸': r'c:\Users\28130\Desktop\lyy-6.12\色氨酸.xlsx',
    '烯酰肉碱': r'c:\Users\28130\Desktop\lyy-6.12\烯酰肉碱.xlsx',
    '混合组': r'c:\Users\28130\Desktop\lyy-6.12\混合组.xlsx'
}

all_data = {}
for name, filepath in files.items():
    if os.path.exists(filepath):
        df = pd.read_excel(filepath).iloc[1:]  # 跳过第一行表头
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
    else:
        print(f"  ⚠️ 文件不存在: {filepath}")

print(f"\n数据加载完成！共 {len(all_data)} 组数据")

# ============================================================================
# 3. 构建训练数据集
# ============================================================================
print("\n[步骤2] 构建训练数据集...")

X = []
Y = []
labels = []

substances = ['铝离子', '色氨酸', '烯酰肉碱']
conc_values = {'1mg/ml': 1.0, '0.5mg/ml': 0.5, '0.1mg/ml': 0.1}

# 收集所有波形以确定最小长度
all_waves = []
for name in substances:
    if name in all_data:
        for conc_label, conc_data in all_data[name].items():
            for wave in conc_data['waveforms']:
                all_waves.append(wave)

if not all_waves:
    print("❌ 没有找到波形数据！")
    exit()

min_len = min(len(w) for w in all_waves)
print(f"所有波形对齐到最小长度: {min_len}")

# 构建训练数据
for substance_name in substances:
    if substance_name not in all_data:
        continue
    
    for conc_label, conc_data in all_data[substance_name].items():
        conc_val = conc_values.get(conc_label, 0.0)
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

print(f"训练数据形状: X={X.shape}, Y={Y.shape}")
print(f"样本数量: {len(X)}")

# 预处理
X_preprocessed = preprocess_pipeline(X)
print("数据预处理完成！")

# ============================================================================
# 4. 核心提升架构：特征筛选 + Stacking融合
# ============================================================================
def train_high_precision_model(X, Y_single, substance_name, feature_count=50):
    """
    针对单一物质进行高精度定制化建模
    X: 预处理后的波形矩阵 (n_samples, n_features)
    Y_single: 该物质的真实浓度向量 (n_samples,)
    """
    print(f"\n{'='*60}")
    print(f"正在为【{substance_name}】训练高精度升级版模型")
    print(f"{'='*60}")
    
    # 划分训练集与测试集（使用更小的测试集以增加训练数据）
    X_train, X_test, y_train, y_test = train_test_split(X, Y_single, test_size=0.15, random_state=42)
    
    # 【突破点一】：基于随机森林的特征重要性波段筛选
    print(f"\n[特征筛选]")
    print(f"原始波形点数: {X.shape[1]}")
    
    # 使用更稳健的特征选择
    rf_estimator = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    feature_selector = SelectFromModel(
        rf_estimator,
        max_features=feature_count,
        threshold=-np.inf
    )
    
    X_train_selected = feature_selector.fit_transform(X_train, y_train)
    X_test_selected = feature_selector.transform(X_test)
    
    selected_indices = feature_selector.get_support(indices=True)
    print(f"筛选后核心特征点数: {X_train_selected.shape[1]}")
    print(f"筛选出的关键波形位置索引: {selected_indices[:10]}...")
    
    # 【突破点二】：构建 Stacking 融合模型（强强联手）
    print("\n[Stacking融合模型]")
    print("基模型组合: PLSR + SVR + Random Forest")
    
    # 基模型1：PLSR（捕捉波形的多重共线性和线性重叠）
    base_pls = PLSRegression(n_components=5)
    
    # 基模型2：SVR（使用更保守的参数）
    base_svr = SVR(C=10, gamma='auto', kernel='rbf')
    
    # 基模型3：Random Forest（使用更保守的参数）
    base_rf = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
    
    # 元模型：使用带交叉验证的岭回归（RidgeCV）
    meta_learner = RidgeCV(alphas=[0.1, 1.0, 10.0])
    
    # 组装 Stacking 模型（使用更简单的配置）
    stacking_regressor = StackingRegressor(
        estimators=[
            ('pls', base_pls),
            ('svr', base_svr),
            ('rf', base_rf)
        ],
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1,
        passthrough=False
    )
    
    # 训练融合模型
    print("训练融合模型中...")
    stacking_regressor.fit(X_train_selected, y_train)
    
    # 交叉验证评估
    cv_scores = cross_val_score(stacking_regressor, X_train_selected, y_train, cv=5, scoring='r2')
    print(f"5折交叉验证 R²: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # 测试集评估
    y_pred = stacking_regressor.predict(X_test_selected)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n✨ 升级版融合模型评估结果：")
    print(f"   - 决定系数 (R²): {r2:.4f}")
    print(f"   - 预测均方根误差 (RMSE): {rmse:.4f}")
    
    if r2 >= 0.8:
        print("   🎉 恭喜！达到目标 R² > 0.8！")
    elif r2 >= 0.75:
        print("   ⚡ 接近目标！继续优化...")
    else:
        print("   💪 继续优化中，当前 R² = {:.4f}".format(r2))
    
    return stacking_regressor, feature_selector, r2, rmse, y_test, y_pred

# ============================================================================
# 5. 训练三种物质的高精度模型
# ============================================================================
print("\n" + "="*80)
print("开始训练三种物质的高精度模型")
print("="*80)

substance_names = ['铝离子', '色氨酸', '烯酰肉碱']
results = []
models = {}
selectors = {}

for i, name in enumerate(substance_names):
    y_single = Y[:, i]
    model, selector, r2, rmse, y_test, y_pred = train_high_precision_model(
        X_preprocessed, y_single, name, feature_count=30
    )
    models[name] = model
    selectors[name] = selector
    results.append({
        '物质': name,
        'R²': r2,
        'RMSE': rmse,
        'y_test': y_test,
        'y_pred': y_pred
    })

# ============================================================================
# 6. 结果汇总与可视化
# ============================================================================
print("\n" + "="*80)
print("进阶算法结果汇总")
print("="*80)

# 打印性能对比表
print("\n【模型性能对比】")
print("-" * 50)
print(f"{'物质':<10} {'R²':<10} {'RMSE':<10} {'目标':<10}")
print("-" * 50)

avg_r2 = 0
for res in results:
    status = "✅" if res['R²'] >= 0.8 else "⏳"
    print(f"{res['物质']:<10} {res['R²']:<10.4f} {res['RMSE']:<10.4f} {status} >0.8")
    avg_r2 += res['R²']

avg_r2 /= len(results)
print("-" * 50)
print(f"{'平均':<10} {avg_r2:<10.4f} {'-':<10} {'-':<10}")

# 生成可视化图表
print("\n[生成可视化图表...]")

# 6.1 预测结果对比图
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, res in enumerate(results):
    ax = axes[i]
    ax.scatter(res['y_test'], res['y_pred'], alpha=0.6, s=60, color='green')
    ax.plot([0, 1.2], [0, 1.2], 'r--', linewidth=2)
    ax.set_xlabel('真实浓度 (mg/ml)', fontsize=12)
    ax.set_ylabel('预测浓度 (mg/ml)', fontsize=12)
    ax.set_title(f'{res["物质"]}\nR²={res["R²"]:.4f}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.1, 1.3])
    ax.set_ylim([-0.1, 1.3])
plt.tight_layout()
plt.savefig('plots/high_precision_predictions.png', dpi=300, bbox_inches='tight')
print("  - high_precision_predictions.png")

# 6.2 性能对比条形图
fig = plt.figure(figsize=(12, 6))
x = np.arange(len(substance_names))
r2_values = [res['R²'] for res in results]

bars = plt.bar(x, r2_values, width=0.5, color=['blue', 'green', 'orange'], alpha=0.8)
plt.axhline(y=0.8, color='red', linestyle='--', label='目标 R²=0.8')

plt.xlabel('物质', fontsize=12)
plt.ylabel('R² 分数', fontsize=12)
plt.title('高级模型性能对比 (特征筛选 + Stacking融合)', fontsize=14, fontweight='bold')
plt.xticks(x, substance_names)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])

for bar, r2 in zip(bars, r2_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{r2:.4f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('plots/high_precision_performance.png', dpi=300, bbox_inches='tight')
print("  - high_precision_performance.png")

# 6.3 特征重要性可视化
fig, axes = plt.subplots(3, 1, figsize=(12, 15))
for i, name in enumerate(substance_names):
    ax = axes[i]
    selector = selectors[name]
    importances = selector.estimator_.feature_importances_
    ax.plot(importances, linewidth=1.5, label='特征重要性')
    
    # 标记筛选出的特征
    selected_mask = selector.get_support()
    ax.scatter(np.where(selected_mask)[0], importances[selected_mask], 
               color='red', s=30, label='筛选出的特征')
    
    ax.set_xlabel('波形位置', fontsize=12)
    ax.set_ylabel('重要性', fontsize=12)
    ax.set_title(f'{name} - 特征重要性与筛选结果', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/feature_importance_selection.png', dpi=300, bbox_inches='tight')
print("  - feature_importance_selection.png")

# ============================================================================
# 7. 混合组预测
# ============================================================================
print("\n[混合组预测...]")
if '混合组' in all_data:
    mixed_data = all_data['混合组']
    mixed_results = []
    
    for col in mixed_data.keys():
        voltage = mixed_data[col]['voltage']
        mask = ~np.isnan(voltage)
        clean_voltage = voltage[mask][:min_len]
        X_pred = np.array([clean_voltage])
        X_pred_p = preprocess_pipeline(X_pred)
        
        pred_concentrations = []
        for i, name in enumerate(substance_names):
            X_pred_selected = selectors[name].transform(X_pred_p)
            pred = models[name].predict(X_pred_selected)[0]
            pred_concentrations.append(max(0, pred))
        
        mixed_results.append({
            '样本': col,
            '预测浓度': pred_concentrations
        })
        print(f"  {col}: 铝离子={pred_concentrations[0]:.4f}, 色氨酸={pred_concentrations[1]:.4f}, 烯酰肉碱={pred_concentrations[2]:.4f}")

# ============================================================================
# 8. 保存模型
# ============================================================================
import joblib
print("\n[保存模型...]")
for name, model in models.items():
    joblib.dump(model, f'models/stacking_model_{name}.pkl')
    joblib.dump(selectors[name], f'models/selector_{name}.pkl')
    print(f"  - models/stacking_model_{name}.pkl")
    print(f"  - models/selector_{name}.pkl")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "="*80)
print("进阶算法总结")
print("="*80)
print("\n【提升策略】")
print("  1. 特征波段筛选：基于随机森林重要性，只保留最核心的特征点")
print("  2. Stacking融合：PLSR + SVR + Random Forest 三模型融合")
print("  3. 元模型：RidgeCV进行最优权重分配")

print("\n【性能提升】")
print("  目标: R² > 0.8")
print(f"  当前平均 R²: {avg_r2:.4f}")

if avg_r2 >= 0.8:
    print("\n🎉 🎉 🎉 恭喜！平均R²突破0.8！")
else:
    print("\n💡 提示：可尝试调整 feature_count 参数或增加训练样本量")
    print("   当前feature_count=30，可尝试范围：20-60")

print("\n" + "="*80)
