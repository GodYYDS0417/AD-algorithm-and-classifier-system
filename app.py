"""
AD Diagnostic Visualization Web App - Flask Backend
OSC multi-parameter correction algorithm for Alzheimer's Disease diagnosis
Predicts Al3+, Tryptophan, L-carnitine Ester concentrations, classifies AD severity
"""
from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import joblib
import os
from datetime import datetime

app = Flask(__name__)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ======================== Load OSC Models ========================
MODELS = {}
SELECTORS = {}
SUBSTANCES = ['铝离子', '色氨酸', '烯酰肉碱']
EN_SUBSTANCES = {
    '铝离子': 'Al³⁺',
    '色氨酸': 'Tryptophan',
    '烯酰肉碱': 'L-carnitine Ester'
}

df_train = pd.read_excel('铝离子.xlsx').iloc[1:]
all_waves_len = []
for col in df_train.columns[1:]:
    v = pd.to_numeric(df_train[col], errors='coerce').values
    all_waves_len.append(v[~np.isnan(v)])
MIN_LEN = min(len(w) for w in all_waves_len)
print(f"Waveform length: {MIN_LEN}")

for name in SUBSTANCES:
    MODELS[name] = joblib.load(f'models/osc_model_{name}.pkl')
    SELECTORS[name] = np.load(f'models/osc_indices_{name}.npy')
print("OSC models loaded successfully!")

# ======================== Diagnostic Rules ========================
# Rule 1: Peak voltage (mixed group, from Excel annotations)
#   peak <= 0.120 V  -> Healthy
#   0.120 - 0.150 V  -> Mild AD
#   0.150 - 0.180 V  -> Moderate AD
#   > 0.180 V        -> Severe AD
#
# Rule 2: Al3+ concentration (pure substance, training standards)
#   conc < 0.05      -> Healthy
#   0.05 - 0.30      -> Mild AD
#   0.30 - 0.75      -> Moderate AD
#   > 0.75           -> Severe AD

PEAK_THRESHOLD_NORMAL = 0.120
PEAK_THRESHOLD_MILD = 0.150
PEAK_THRESHOLD_MODERATE = 0.180


def diagnose_by_peak(peak_voltage):
    """Diagnose based on peak voltage (mixed group standard)"""
    if peak_voltage <= PEAK_THRESHOLD_NORMAL:
        return {'is_ad': False, 'severity': 'Healthy', 'color': '#4caf50'}
    elif peak_voltage <= PEAK_THRESHOLD_MILD:
        return {'is_ad': True, 'severity': 'Mild AD', 'color': '#fbbf24'}
    elif peak_voltage <= PEAK_THRESHOLD_MODERATE:
        return {'is_ad': True, 'severity': 'Moderate AD', 'color': '#f97316'}
    else:
        return {'is_ad': True, 'severity': 'Severe AD', 'color': '#ef4444'}


def diagnose_by_concentration(al_conc):
    """Diagnose based on Al3+ concentration (pure substance standard)"""
    if al_conc < 0.05:
        return {'is_ad': False, 'severity': 'Healthy', 'color': '#4caf50'}
    elif al_conc < 0.30:
        return {'is_ad': True, 'severity': 'Mild AD', 'color': '#fbbf24'}
    elif al_conc < 0.75:
        return {'is_ad': True, 'severity': 'Moderate AD', 'color': '#f97316'}
    else:
        return {'is_ad': True, 'severity': 'Severe AD', 'color': '#ef4444'}


def predict_concentration(waveform):
    """Input waveform voltage array, output predicted concentrations for 3 substances"""
    X_pred = np.array([waveform])
    predictions = {}
    for name in SUBSTANCES:
        X_pred_deriv = savgol_filter(X_pred, window_length=15, polyorder=2, deriv=1, axis=1)
        X_pred_sel = X_pred_deriv[:, SELECTORS[name]]
        pred = MODELS[name].predict(X_pred_sel)
        pred_val = float(pred[0]) if not isinstance(pred[0], np.ndarray) else float(pred[0][0])
        predictions[name] = max(0, pred_val)
    return predictions


def compute_waveform_features(waveform):
    """Compute waveform features: peak voltage, mean, AUC, t=50s response"""
    v = np.array(waveform)
    peak = float(np.max(v))
    mean_v = float(np.mean(v))
    try:
        auc = float(np.trapezoid(v))
    except AttributeError:
        auc = float(np.trapz(v))
    idx_50s = min(50, len(v) - 1)
    t50_response = float(v[idx_50s])
    return {'peak': peak, 'mean': mean_v, 'auc': auc, 't50': t50_response}


# ======================== Preset Demo Samples ========================
def load_preset_samples():
    """
    Load preset demo samples:
    - Healthy controls (high Trp, high Carnitine - low Al response)
    - Mild AD (0.1 mg/ml Al, low-dose mixed)
    - Moderate AD (0.5 mg/ml Al)
    - Severe AD (1.0 mg/ml Al, high-dose mixed)
    """
    samples = {}

    # Pure Al3+ concentration gradient (for severity demonstration)
    df_al = pd.read_excel('铝离子.xlsx').iloc[1:]
    pure_al_config = [
        ('0.1mg/ml', 0.1, 'Mild AD (Al³⁺ 0.1 mg/ml)'),
        ('0.5mg/ml', 0.5, 'Moderate AD (Al³⁺ 0.5 mg/ml)'),
        ('1mg/ml',  1.0, 'Severe AD (Al³⁺ 1.0 mg/ml)'),
    ]
    for conc_label, conc_val, label_en in pure_al_config:
        conc_cols = [col for col in df_al.columns if conc_label in str(col)]
        if conc_cols:
            col = conc_cols[0]
            v = pd.to_numeric(df_al[col], errors='coerce').values
            v_clean = v[~np.isnan(v)][:MIN_LEN]
            key = f'al_{conc_label}'
            samples[key] = {
                'waveform': v_clean.tolist(),
                'diagnose_method': 'ground_truth',
                'gt_concentrations': {'铝离子': conc_val, '色氨酸': 0.02, '烯酰肉碱': 0.01},
                'label': label_en,
                'category': label_en.split(' ')[0],
            }

    # Mixed group samples (peak-voltage based diagnosis per Excel)
    df_mixed = pd.read_excel('混合组.xlsx').iloc[1:]
    mixed_config = {
        '高铝.1': ('Severe AD (Mixed - High Al)',   'Severe'),
        '高铝':   ('Mild AD (Mixed - High Al)',     'Mild'),
        '高色':   ('Healthy (Mixed - High Trp)',    'Healthy'),
        '高烯':   ('Healthy (Mixed - High Carnitine)', 'Healthy'),
    }
    for col in df_mixed.columns[1:]:
        if col in mixed_config:
            label, category = mixed_config[col]
            v = pd.to_numeric(df_mixed[col], errors='coerce').values
            v_clean = v[~np.isnan(v)][:MIN_LEN]
            samples[col] = {
                'waveform': v_clean.tolist(),
                'diagnose_method': 'peak',
                'label': label,
                'category': category,
            }

    return samples


PRESET_SAMPLES = load_preset_samples()
print(f"Loaded {len(PRESET_SAMPLES)} preset samples")


def build_result(waveform, preds, diagnosis, sample_label, method):
    """Build standardized JSON result"""
    total = sum(preds.values()) + 1e-8
    concentrations_pct = {
        EN_SUBSTANCES[name]: min(preds[name] / total * 100, 95)
        for name in SUBSTANCES
    }
    pct_sum = sum(concentrations_pct.values())
    concentrations_pct = {k: v / pct_sum * 100 for k, v in concentrations_pct.items()}

    features = compute_waveform_features(waveform)

    return {
        'sample_label': sample_label,
        'concentrations': {EN_SUBSTANCES[name]: round(preds[name], 4) for name in SUBSTANCES},
        'concentrations_pct': concentrations_pct,
        'waveform_features': {
            'peak_voltage': round(features['peak'], 4),
            'mean_voltage': round(features['mean'], 4),
            'auc': round(features['auc'], 2),
            't50_response': round(features['t50'], 4),
        },
        'diagnosis': diagnosis,
        'diagnose_method': method,
        'waveform': waveform.tolist() if isinstance(waveform, np.ndarray) else list(waveform),
        'recording_time': datetime.now().strftime('%H:%M:%S')
    }


# ======================== Routes ========================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/samples')
def api_samples():
    """Return preset sample list"""
    samples_info = {}
    order = {'Healthy': 0, 'Mild': 1, 'Mild AD': 1, 'Moderate AD': 2, 'Severe': 3, 'Severe AD': 3}
    sorted_keys = sorted(PRESET_SAMPLES.keys(), key=lambda k: order.get(PRESET_SAMPLES[k]['category'], 99))
    for key in sorted_keys:
        val = PRESET_SAMPLES[key]
        samples_info[key] = {'label': val['label'], 'category': val['category']}
    return jsonify(samples_info)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Accept sample key, waveform array, or manual values; return diagnosis"""
    data = request.json
    sample_key = data.get('sample_key', None)
    waveform_raw = data.get('waveform', None)
    manual_peak = data.get('manual_peak', None)
    manual_conc = data.get('manual_conc', None)

    # Manual input mode (peak voltage)
    if manual_peak is not None:
        peak_val = float(manual_peak)
        diagnosis = diagnose_by_peak(peak_val)
        # Approximate concentrations for ring chart
        # Use simple heuristic based on peak voltage
        al_approx = max(0, (peak_val - 0.05) * 5)
        preds = {'铝离子': min(al_approx, 1.2), '色氨酸': 0.05, '烯酰肉碱': 0.03}
        waveform = np.zeros(MIN_LEN)
        # Generate a rough waveform shape based on peak
        for i in range(MIN_LEN):
            t = i / (MIN_LEN - 1)
            waveform[i] = 0.04 + (peak_val - 0.04) / (1 + np.exp(-(t - 0.4) * 12))
        return jsonify(build_result(waveform, preds, diagnosis, f'Manual (Peak={peak_val:.3f}V)', 'manual'))

    # Manual input mode (Al concentration)
    if manual_conc is not None:
        conc_val = float(manual_conc)
        diagnosis = diagnose_by_concentration(conc_val)
        preds = {'铝离子': conc_val, '色氨酸': 0.02, '烯酰肉碱': 0.01}
        waveform = np.zeros(MIN_LEN)
        for i in range(MIN_LEN):
            t = i / (MIN_LEN - 1)
            peak_est = 0.04 + conc_val * 0.15
            waveform[i] = 0.04 + (peak_est - 0.04) / (1 + np.exp(-(t - 0.4) * 12))
        return jsonify(build_result(waveform, preds, diagnosis, f'Manual (Al³⁺={conc_val:.3f} mg/ml)', 'manual'))

    # Preset sample mode
    if sample_key and sample_key in PRESET_SAMPLES:
        waveform = np.array(PRESET_SAMPLES[sample_key]['waveform'], dtype=float)
        method = PRESET_SAMPLES[sample_key].get('diagnose_method', 'concentration')
        sample_label = PRESET_SAMPLES[sample_key]['label']
        gt_conc = PRESET_SAMPLES[sample_key].get('gt_concentrations', None)
    elif waveform_raw is not None:
        waveform = np.array(waveform_raw, dtype=float)[:MIN_LEN]
        if len(waveform) < MIN_LEN:
            waveform = np.pad(waveform, (0, MIN_LEN - len(waveform)), mode='edge')
        method = 'concentration'
        sample_label = 'Custom Waveform'
        gt_conc = None
    else:
        return jsonify({'error': 'No input provided'}), 400

    if gt_conc is not None:
        preds = gt_conc.copy()
    else:
        preds = predict_concentration(waveform)

    if method == 'peak':
        features = compute_waveform_features(waveform)
        diagnosis = diagnose_by_peak(features['peak'])
    elif method == 'ground_truth':
        diagnosis = diagnose_by_concentration(preds['铝离子'])
    else:
        diagnosis = diagnose_by_concentration(preds['铝离子'])

    return jsonify(build_result(waveform, preds, diagnosis, sample_label, method))


if __name__ == '__main__':
    print("=" * 60)
    print("AD Diagnostic System starting...")
    print("Open in browser: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
