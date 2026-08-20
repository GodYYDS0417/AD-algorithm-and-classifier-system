#!/usr/bin/env python3
"""Append algorithm methods to the existing experimental methods docx."""
from docx import Document
from docx.shared import Pt, Cm
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# Open the original document
docx_path = r"C:\Users\28130\Desktop\lyy-6.12\实验方法部分.docx"
doc = Document(docx_path)


def add_heading_style(doc, text):
    """Add a section heading matching the original document style."""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(12)
    pf.space_after = Pt(6)
    run = p.add_run(text)
    run.font.bold = True
    run.font.size = Pt(11)
    # Set font
    rPr = run._element.find(qn('w:rPr'))
    if rPr is None:
        rPr = OxmlElement('w:rPr')
        run._element.insert(0, rPr)
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.insert(0, rFonts)
    rFonts.set(qn('w:eastAsia'), 'Times New Roman')
    rFonts.set(qn('w:ascii'), 'Times New Roman')
    rFonts.set(qn('w:hAnsi'), 'Times New Roman')
    return p


def add_body_paragraph(doc, text):
    """Add a body paragraph matching the original document style."""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = 1.15
    pf.space_after = Pt(6)
    pf.first_line_indent = Cm(0.74)
    run = p.add_run(text)
    run.font.size = Pt(10.5)
    # Set font
    rPr = run._element.find(qn('w:rPr'))
    if rPr is None:
        rPr = OxmlElement('w:rPr')
        run._element.insert(0, rPr)
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.insert(0, rFonts)
    rFonts.set(qn('w:eastAsia'), 'Times New Roman')
    rFonts.set(qn('w:ascii'), 'Times New Roman')
    rFonts.set(qn('w:hAnsi'), 'Times New Roman')
    return p


# Add a page break before algorithm section
doc.add_page_break()

# ========== Algorithm Methods Section ==========

# Section 1: Sample preparation and sensor data acquisition
add_heading_style(doc, "Sample preparation and sensor waveform acquisition")
add_body_paragraph(doc,
    "Three target analytes including aluminum ion (Al³⁺), tryptophan, and enoyl-carnitine "
    "were prepared at three concentration gradients: 1 mg/mL, 0.5 mg/mL, and 0.1 mg/mL to "
    "simulate the gastrointestinal matrix samples. The Pt-L-MOF sensor was incubated with "
    "each analyte solution under identical neutral pH conditions, and the normalized voltage "
    "waveform signals were continuously recorded by the sensor device for subsequent multivariate "
    "quantitative analysis. Raw waveform data of all concentration groups were collected and "
    "stored for preprocessing and modeling."
)

# Section 2: Signal preprocessing pipeline
add_heading_style(doc, "Signal preprocessing pipeline")
add_body_paragraph(doc,
    "Five preprocessing strategies were systematically compared to evaluate their effects on "
    "prediction performance: (1) Raw: original voltage waveform without any correction; "
    "(2) SG: Savitzky-Golay smoothing only for eliminating high-frequency random noise; "
    "(3) SG+SNV: SG smoothing combined with standard normal variate transformation for reducing "
    "light scattering effects; (4) SG+Deriv: SG smoothing plus first-order derivative for "
    "removing slow baseline drift caused by sensor drift and gastrointestinal matrix interference; "
    "(5) SG+Deriv+OSC: full pipeline integrating SG smoothing, first-order derivative, and "
    "orthogonal signal correction (OSC) for eliminating orthogonal irrelevant signals uncorrelated "
    "with analyte concentration."
)
add_body_paragraph(doc,
    "Ablation experiments were further implemented by sequentially removing derivative or OSC "
    "modules from the complete preprocessing pipeline to quantify the independent contribution "
    "of each processing unit to the final prediction accuracy."
)

# Section 3: Multivariate regression models
add_heading_style(doc, "Multivariate regression models")
add_body_paragraph(doc,
    "Two classical multivariate calibration algorithms were established for concentration "
    "prediction: Partial Least Squares Regression (PLSR) and Support Vector Regression (SVR). "
    "PLSR extracts latent variables that maximize the covariance between the waveform features "
    "and the concentration values, which is particularly effective for handling multicollinearity "
    "in high-dimensional spectral data. SVR maps the input features into a high-dimensional "
    "feature space via kernel functions and constructs an optimal hyperplane for regression, "
    "which exhibits strong capability for capturing nonlinear relationships."
)
add_body_paragraph(doc,
    "The full preprocessed waveform features were input into both models. The dataset was "
    "randomly divided into training set and test set, and cross-validation was performed to "
    "optimize model parameters and avoid overfitting."
)

# Section 4: Performance evaluation metrics
add_heading_style(doc, "Performance evaluation metrics")
add_body_paragraph(doc,
    "Coefficient of determination (R²) and root mean square error (RMSE) were adopted as core "
    "evaluation metrics to compare fitting accuracy and prediction error of different preprocessing "
    "configurations and regression models across all three target substances. R² measures the "
    "proportion of variance in the dependent variable that is predictable from the independent "
    "variables, with values closer to 1 indicating better model performance. RMSE represents the "
    "square root of the average of squared differences between predicted and actual concentration "
    "values, providing a measure of prediction error in the original concentration units."
)

# Section 5: Prediction error analysis
add_heading_style(doc, "Prediction error distribution analysis")
add_body_paragraph(doc,
    "To further assess the reliability and robustness of the established models, prediction error "
    "distributions across different concentration levels were analyzed for each substance. The "
    "errors were calculated as the difference between predicted concentrations and actual concentrations, "
    "and the distribution patterns were examined to detect potential systematic biases or "
    "concentration-dependent prediction performance variations."
)

# Save the new document
output_path = r"C:\Users\28130\Desktop\lyy-6.12\doubao_final.docx"
doc.save(output_path)
print(f"Document saved to: {output_path}")
print(f"Total paragraphs: {len(doc.paragraphs)}")
