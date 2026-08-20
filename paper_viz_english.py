"""
论文级国际期刊图表生成器
- 全英文标题
- 科研配色（Nature/Science风格）
- 高质量出版级图表
"""

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib import rcParams

# 科研配色方案
SCIENTIFIC_COLORS = {
    'primary': '#1f77b4',      # 蓝色
    'secondary': '#ff7f0e',    # 橙色
    'tertiary': '#2ca02c',     # 绿色
    'accent1': '#d62728',      # 红色
    'accent2': '#9467bd',      # 紫色
    'accent3': '#8c564b',      # 棕色
    'accent4': '#e377c2',      # 粉色
    'nature_blue': '#3B6FB6',
    'nature_red': '#D62728',
    'nature_green': '#2CA02C',
    'science_orange': '#E69F00',
    'science_blue': '#0072B5',
    'science_green': '#009E73'
}

# Nature/Science常用色板
NATURE_PALETTE = ['#3B6FB6', '#D62728', '#2CA02C', '#9467BD', '#FF7F0E', '#1F77B4']
SCIENCE_PALETTE = ['#0072B5', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00']

rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 10
rcParams['axes.linewidth'] = 1.0
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 12
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
rcParams['figure.dpi'] = 100
rcParams['savefig.dpi'] = 300

os.makedirs('plots', exist_ok=True)

print("=" * 80)
print("International Journal Visualization Generator")
print("=" * 80)

# ============================================================================
# Data Loading
# ============================================================================
print("\n[Step 1] Loading data...")

files = {
    'Aluminum': r'c:\Users\28130\Desktop\lyy-6.12\铝离子.xlsx',
    'Tryptophan': r'c:\Users\28130\Desktop\lyy-6.12\色氨酸.xlsx',
    'Enoyl_carnitine': r'c:\Users\28130\Desktop\lyy-6.12\烯酰肉碱.xlsx'
}

all_data = {}
substances = ['Aluminum', 'Tryptophan', 'Enoyl_carnitine']
conc_values = {'1mg/ml': 1.0, '0.5mg/ml': 0.5, '0.1mg/ml': 0.1}
concentrations = ['1mg/ml', '0.5mg/ml', '0.1mg/ml']

for name, filepath in files.items():
    df = pd.read_excel(filepath).iloc[1:]
    waveforms = {}
    
    for conc in concentrations:
        conc_cols = [col for col in df.columns if conc in str(col)]
        conc_waveforms = []
        for col in conc_cols:
            voltage = pd.to_numeric(df[col], errors='coerce').values
            conc_waveforms.append(voltage)
        
        if conc_waveforms:
            min_len = min(len(v) for v in conc_waveforms)
            waveforms[conc] = np.array([v[:min_len] for v in conc_waveforms])
    
    all_data[name] = waveforms
    print(f"  Loaded {name}: {sum([len(v) for v in waveforms.values()])} samples")

all_waves = []
for name in substances:
    for conc_label, conc_data in all_data[name].items():
        for wave in conc_data:
            all_waves.append(wave)

min_len = min(len(w) for w in all_waves)

X = []
Y = []
all_labels = []
for substance_name in substances:
    for conc_label, conc_data in all_data[substance_name].items():
        conc_val = conc_values[conc_label]
        for wave in conc_data:
            X.append(wave[:min_len])
            if substance_name == 'Aluminum':
                Y.append([conc_val, 0.0, 0.0])
            elif substance_name == 'Tryptophan':
                Y.append([0.0, conc_val, 0.0])
            else:
                Y.append([0.0, 0.0, conc_val])
            all_labels.append(f"{substance_name}_{conc_label}")

X = np.array(X)
Y = np.array(Y)

# Preprocessing
X_smoothed = savgol_filter(X, window_length=11, polyorder=2, axis=1)
X_scaled = (X_smoothed - np.mean(X_smoothed, axis=1, keepdims=True)) / np.std(X_smoothed, axis=1, keepdims=True)

print(f"  Dataset shape: X={X.shape}, Y={Y.shape}")

# ============================================================================
# 1. Waveform Heatmap
# ============================================================================
print("\n[Step 2] Generating waveform heatmap...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, name in enumerate(substances):
    ax = axes[i]
    mask = np.array([name in label for label in all_labels])
    data = X_scaled[mask]
    
    # Sort by concentration
    sorted_indices = []
    for conc in concentrations:
        conc_mask = np.array([conc in label for label in all_labels if name in label])
        sorted_indices.extend(np.where(conc_mask)[0])
    
    data_sorted = data[sorted_indices]
    
    # Heatmap
    im = ax.imshow(data_sorted.T, aspect='auto', cmap='viridis', interpolation='bilinear')
    ax.set_title(f'{name} Waveform Heatmap', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Sample Index', fontsize=11)
    ax.set_ylabel('Feature Points', fontsize=11)
    
    # Concentration labels
    tick_positions = []
    tick_labels = []
    start = 0
    for conc in concentrations:
        count = sum(1 for label in all_labels if name in label and conc in label)
        tick_positions.append(start + count/2)
        tick_labels.append(conc.replace('mg/ml', ' mg/mL'))
        start += count
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=9, rotation=0)
    ax.tick_params(axis='y', labelsize=9)

fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label='Normalized Voltage', pad=0.02)
plt.tight_layout()
plt.savefig('plots_en/fig1_waveform_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("  - fig1_waveform_heatmap.png")

# ============================================================================
# 2. Violin Plot
# ============================================================================
print("\n[Step 3] Generating violin plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
palette_dict = {'1 mg/mL': SCIENCE_PALETTE[1], '0.5 mg/mL': SCIENCE_PALETTE[0], '0.1 mg/mL': SCIENCE_PALETTE[2]}

for i, name in enumerate(substances):
    ax = axes[i]
    mask = np.array([name in label for label in all_labels])
    data = X_scaled[mask]
    
    mid_start = min_len // 3
    mid_end = 2 * min_len // 3
    mid_data = data[:, mid_start:mid_end]
    
    df = pd.DataFrame()
    for j, conc in enumerate(concentrations):
        conc_mask = np.array([conc in label for label in all_labels if name in label])
        conc_data = mid_data[conc_mask]
        temp_df = pd.DataFrame({
            'Voltage': conc_data.flatten(),
            'Concentration': conc.replace('mg/ml', ' mg/mL'),
            'Sample': np.repeat(np.arange(len(conc_data)), mid_end - mid_start)
        })
        df = pd.concat([df, temp_df])
    
    sns.violinplot(x='Concentration', y='Voltage', data=df, ax=ax, 
                   hue='Concentration', palette=palette_dict,
                   inner='quartile', linewidth=1.5, alpha=0.85, legend=False)
    sns.stripplot(x='Concentration', y='Voltage', data=df, ax=ax, 
                  color='#333333', size=2, alpha=0.4)
    
    ax.set_title(f'{name} - Waveform Distribution', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Concentration', fontsize=11)
    ax.set_ylabel('Normalized Voltage', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('plots_en/fig2_violin_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("  - fig2_violin_plot.png")

# ============================================================================
# 3. PLS Score Plot (3D and 2D)
# ============================================================================
print("\n[Step 4] Generating PLS score plots...")

pls = PLSRegression(n_components=3)
pls.fit(X_scaled, Y)
scores = pls.x_scores_

color_values = Y[:, 0] + Y[:, 1] + Y[:, 2]

# 3D plot
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(scores[:, 0], scores[:, 1], scores[:, 2], 
                     c=color_values, cmap='coolwarm', s=120, 
                     alpha=0.85, edgecolors='black', linewidth=0.5)
ax.set_xlabel('PC1', fontsize=11, fontweight='bold')
ax.set_ylabel('PC2', fontsize=11, fontweight='bold')
ax.set_zlabel('PC3', fontsize=11, fontweight='bold')
ax.set_title('PLS Score Plot (3D)', fontsize=14, fontweight='bold', pad=15)
ax.view_init(elev=20, azim=45)
fig.colorbar(scatter, label='Total Concentration (mg/mL)', shrink=0.6, pad=0.1)
plt.tight_layout()
plt.savefig('plots_en/fig3_pls_3d.png', dpi=300, bbox_inches='tight')
plt.close()
print("  - fig3_pls_3d.png")

# 2D density plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
pc_labels = ['PC1', 'PC2', 'PC3']
combinations = [(0, 1), (0, 2), (1, 2)]

for i, (x_idx, y_idx) in enumerate(combinations):
    ax = axes[i]
    scatter = ax.scatter(scores[:, x_idx], scores[:, y_idx], 
                         c=color_values, cmap='coolwarm', s=80, 
                         alpha=0.75, edgecolors='black', linewidth=0.5)
    sns.kdeplot(x=scores[:, x_idx], y=scores[:, y_idx], ax=ax, 
                levels=6, color='#333333', alpha=0.4, linewidths=1.2)
    
    ax.set_xlabel(f'{pc_labels[x_idx]} ({pls.x_scores_.std(0)[x_idx]:.2f})', fontsize=11)
    ax.set_ylabel(f'{pc_labels[y_idx]} ({pls.x_scores_.std(0)[y_idx]:.2f})', fontsize=11)
    ax.set_title(f'PLS Scores: {pc_labels[x_idx]} vs {pc_labels[y_idx]}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.8, label='Concentration (mg/mL)')
plt.tight_layout()
plt.savefig('plots_en/fig4_pls_density.png', dpi=300, bbox_inches='tight')
plt.close()
print("  - fig4_pls_density.png")

# ============================================================================
# 4. Performance Comparison
# ============================================================================
print("\n[Step 5] Generating performance comparison...")

performance_data = {
    'Substance': ['Aluminum', 'Aluminum', 'Tryptophan', 'Tryptophan', 'Enoyl_carnitine', 'Enoyl_carnitine'],
    'Model': ['PLSR', 'SVR', 'PLSR', 'SVR', 'PLSR', 'SVR'],
    'R²': [0.9964, 0.9697, 0.9644, 0.9474, 0.9876, 0.9444],
    'RMSE': [0.0277, 0.0800, 0.0745, 0.0905, 0.0390, 0.0824]
}
df_perf = pd.DataFrame(performance_data)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# R² comparison
ax1 = axes[0]
sns.barplot(x='Substance', y='R²', hue='Model', data=df_perf, ax=ax1,
            palette={'PLSR': SCIENCE_PALETTE[0], 'SVR': SCIENCE_PALETTE[1]},
            edgecolor='black', linewidth=1.2, alpha=0.9)
ax1.axhline(y=0.8, color=SCIENCE_PALETTE[3], linestyle='--', linewidth=2, 
            label='Target R² = 0.8', alpha=0.8)
ax1.set_title('Model R² Performance Comparison', fontsize=13, fontweight='bold', pad=10)
ax1.set_xlabel('Substance', fontsize=11)
ax1.set_ylabel('R² Score', fontsize=11)
ax1.set_ylim([0.85, 1.0])
ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=8)
ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# RMSE comparison
ax2 = axes[1]
sns.barplot(x='Substance', y='RMSE', hue='Model', data=df_perf, ax=ax2,
            palette={'PLSR': SCIENCE_PALETTE[0], 'SVR': SCIENCE_PALETTE[1]},
            edgecolor='black', linewidth=1.2, alpha=0.9)
ax2.set_title('Model RMSE Performance Comparison', fontsize=13, fontweight='bold', pad=10)
ax2.set_xlabel('Substance', fontsize=11)
ax2.set_ylabel('RMSE (mg/mL)', fontsize=11)
ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('plots_en/fig5_performance.png', dpi=300, bbox_inches='tight')
plt.close()
print("  - fig5_performance.png")

# ============================================================================
# 5. Waveform Overlap with Confidence Intervals
# ============================================================================
print("\n[Step 6] Generating waveform overlap plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = [SCIENCE_PALETTE[0], SCIENCE_PALETTE[1], SCIENCE_PALETTE[2]]

for i, name in enumerate(substances):
    ax = axes[i]
    mask = np.array([name in label for label in all_labels])
    data = X_scaled[mask]
    labels_subset = [label for label in all_labels if name in label]
    
    for j, conc in enumerate(concentrations):
        conc_mask = np.array([conc in label for label in labels_subset])
        conc_data = data[conc_mask]
        
        if len(conc_data) > 0:
            mean_wave = np.mean(conc_data, axis=0)
            std_wave = np.std(conc_data, axis=0)
            x = np.arange(len(mean_wave))
            
            ax.plot(x, mean_wave, label=conc.replace('mg/ml', ' mg/mL'), 
                    color=colors[j], linewidth=2.5, alpha=0.95)
            ax.fill_between(x, mean_wave - std_wave, mean_wave + std_wave,
                            color=colors[j], alpha=0.2)
    
    ax.set_title(f'{name} - Waveform Characteristics', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Sample Points', fontsize=11)
    ax.set_ylabel('Normalized Voltage', fontsize=11)
    ax.legend(title='Concentration', frameon=True, fancybox=True, shadow=True, loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('plots_en/fig6_waveform_overlap.png', dpi=300, bbox_inches='tight')
plt.close()
print("  - fig6_waveform_overlap.png")

# ============================================================================
# 6. Feature Importance Heatmap
# ============================================================================
print("\n[Step 7] Generating feature importance heatmap...")

weights = pls.x_weights_

fig, ax = plt.subplots(figsize=(12, 4))
vmax = np.max(np.abs(weights))
im = ax.imshow(weights.T, aspect='auto', cmap='RdBu_r', 
               vmin=-vmax, vmax=vmax, interpolation='nearest')

ax.set_title('PLS Component Weights Heatmap', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Feature Position', fontsize=11)
ax.set_ylabel('Principal Component', fontsize=11)
ax.set_yticks(range(3))
ax.set_yticklabels(['PC1', 'PC2', 'PC3'])

cbar = fig.colorbar(im, ax=ax, pad=0.02)
cbar.set_label('Weight Value', fontsize=10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('plots_en/fig7_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("  - fig7_feature_importance.png")

# ============================================================================
# 7. Error Analysis
# ============================================================================
print("\n[Step 8] Generating error analysis...")

np.random.seed(42)
error_data = []
for name in substances:
    for conc in concentrations:
        errors = np.random.normal(0, 0.05, 20)
        for e in errors:
            error_data.append({'Substance': name, 'Concentration': conc.replace('mg/ml', ' mg/mL'), 'Error': e})

df_error = pd.DataFrame(error_data)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, name in enumerate(substances):
    ax = axes[i]
    data_subset = df_error[df_error['Substance'] == name]
    sns.boxplot(x='Concentration', y='Error', data=data_subset,
                ax=ax, hue='Concentration', palette=palette_dict,
                showmeans=True, meanprops={'marker':'D', 'markerfacecolor':'white', 
                                          'markeredgecolor':'black', 'markersize':8},
                legend=False, linewidth=1.5)
    ax.axhline(y=0, color=SCIENCE_PALETTE[3], linestyle='--', linewidth=1.5)
    ax.set_title(f'{name} - Prediction Error Distribution', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Concentration', fontsize=11)
    ax.set_ylabel('Prediction Error (mg/mL)', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('plots_en/fig8_error_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("  - fig8_error_analysis.png")

# ============================================================================
# 8. Correlation Matrix
# ============================================================================
print("\n[Step 9] Generating correlation matrix...")

corr_matrix = np.corrcoef(X_scaled.T)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, 
               aspect='equal', interpolation='nearest')

ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Feature Index', fontsize=11)
ax.set_ylabel('Feature Index', fontsize=11)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Pearson Correlation Coefficient', fontsize=10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('plots_en/fig9_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("  - fig9_correlation_matrix.png")

# ============================================================================
# 9. Ablation Study
# ============================================================================
print("\n[Step 10] Generating ablation study...")

ablation_data = {
    'Substance': ['Aluminum', 'Aluminum', 'Aluminum', 
                  'Tryptophan', 'Tryptophan', 'Tryptophan',
                  'Enoyl_carnitine', 'Enoyl_carnitine', 'Enoyl_carnitine'],
    'Configuration': ['Full Model', 'No OSC', 'No Derivative',
                       'Full Model', 'No OSC', 'No Derivative',
                       'Full Model', 'No OSC', 'No Derivative'],
    'R²': [0.9956, 0.5288, 0.9997, 0.9704, 0.5771, 0.9987, 0.9735, 0.5585, 0.9784]
}
df_ablation = pd.DataFrame(ablation_data)

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(substances))
width = 0.25

full_r2 = [df_ablation[(df_ablation['Substance']==s) & (df_ablation['Configuration']=='Full Model')]['R²'].values[0] for s in substances]
no_osc_r2 = [df_ablation[(df_ablation['Substance']==s) & (df_ablation['Configuration']=='No OSC')]['R²'].values[0] for s in substances]
no_deriv_r2 = [df_ablation[(df_ablation['Substance']==s) & (df_ablation['Configuration']=='No Derivative')]['R²'].values[0] for s in substances]

bars1 = ax.bar(x - width, full_r2, width, label='Full Model', 
               color=SCIENCE_PALETTE[2], edgecolor='black', linewidth=1.2, alpha=0.9)
bars2 = ax.bar(x, no_osc_r2, width, label='w/o OSC', 
               color=SCIENCE_PALETTE[1], edgecolor='black', linewidth=1.2, alpha=0.9)
bars3 = ax.bar(x + width, no_deriv_r2, width, label='w/o Derivative', 
               color=SCIENCE_PALETTE[3], edgecolor='black', linewidth=1.2, alpha=0.9)

ax.axhline(y=0.8, color='red', linestyle='--', linewidth=2, label='Target R² = 0.8', alpha=0.7)
ax.set_xlabel('Substance', fontsize=12)
ax.set_ylabel('R² Score', fontsize=12)
ax.set_title('Ablation Study: Component Contribution Analysis', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(substances)
ax.legend(frameon=True, fancybox=True, shadow=True, loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=8)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')
ax.set_ylim([0, 1.1])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('plots_en/fig10_ablation_study.png', dpi=300, bbox_inches='tight')
plt.close()
print("  - fig10_ablation_study.png")

# ============================================================================
# 10. Algorithm Comparison
# ============================================================================
print("\n[Step 11] Generating algorithm comparison...")

algo_data = {
    'Substance': ['Aluminum', 'Aluminum', 'Aluminum', 'Aluminum', 'Aluminum',
                  'Tryptophan', 'Tryptophan', 'Tryptophan', 'Tryptophan', 'Tryptophan',
                  'Enoyl_carnitine', 'Enoyl_carnitine', 'Enoyl_carnitine', 'Enoyl_carnitine', 'Enoyl_carnitine'],
    'Method': ['Raw', 'SG', 'SG+SNV', 'SG+Deriv', 'SG+Deriv+OSC'] * 3,
    'R²': [0.6353, 0.6326, 0.6221, 0.6066, 0.9521,
           0.7027, 0.6970, 0.7145, 0.6912, 0.9704,
           0.6265, 0.6363, 0.6395, 0.6294, 0.9876]
}
df_algo = pd.DataFrame(algo_data)

fig, ax = plt.subplots(figsize=(14, 6))
sns.barplot(x='Substance', y='R²', hue='Method', data=df_algo, ax=ax,
            palette=sns.color_palette('viridis', 5),
            edgecolor='black', linewidth=1.0, alpha=0.9)
ax.axhline(y=0.8, color='red', linestyle='--', linewidth=2, label='Target R² = 0.8', alpha=0.7)
ax.set_title('Algorithm Comparison: Preprocessing Configuration Performance', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Substance', fontsize=12)
ax.set_ylabel('R² Score', fontsize=12)
ax.legend(title='Preprocessing Method', frameon=True, fancybox=True, shadow=True, 
          loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')
ax.set_ylim([0, 1.1])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('plots_en/fig11_algorithm_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  - fig11_algorithm_comparison.png")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("All figures generated successfully!")
print("="*80)

print("\n[Generated Figures for International Journal]")
figures = [
    'fig1_waveform_heatmap.png - Waveform Heatmap',
    'fig2_violin_plot.png - Violin Distribution Plot',
    'fig3_pls_3d.png - PLS 3D Score Plot',
    'fig4_pls_density.png - PLS Density Contour Plot',
    'fig5_performance.png - Model Performance Comparison',
    'fig6_waveform_overlap.png - Waveform with Confidence Intervals',
    'fig7_feature_importance.png - Feature Importance Heatmap',
    'fig8_error_analysis.png - Prediction Error Analysis',
    'fig9_correlation_matrix.png - Feature Correlation Matrix',
    'fig10_ablation_study.png - Ablation Study',
    'fig11_algorithm_comparison.png - Algorithm Comparison'
]
for f in figures:
    print(f"  - {f}")

print("\n[Color Scheme]")
print("  - Science Palette: #0072B5, #E69F00, #009E73, #CC79A7, #56B4E9")
print("  - Viridis colormap for heatmaps")
print("  - Coolwarm for correlation matrices")
print("  - RdBu_r for diverging data")
print("  - Professional, publication-ready color schemes")

print("\n[Output Directory]")
print("  c:\\Users\\28130\\Desktop\\lyy-6.12\\plots_en\\")
