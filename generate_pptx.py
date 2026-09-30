"""
generate_pptx.py — Fill the Review-2 Template with ClimateTwin project content.
"""

import copy
import io
import json
import os
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_AUTO_SIZE
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt
from lxml import etree


BASE = Path(__file__).parent
TEMPLATE = BASE / "Review-2 Template.pptx"
OUTPUT   = BASE / "ClimateTwin_Review2_Slides.pptx"

DARK_BLUE = RGBColor(0x1D, 0x2F, 0x82)
BLACK     = RGBColor(0x00, 0x00, 0x00)
GRAY      = RGBColor(0x50, 0x50, 0x50)
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)

HDR_FILL_T = (29, 47, 130)
ROW_ODD_T  = (248, 249, 252)
ROW_EVEN_T = (235, 241, 251)


# ===========================================================================
# HIGH-LEVEL TEXT HELPER using python-pptx native API (not raw XML)
# ===========================================================================

def clear_and_write(text_frame, entries):
    """
    Clear a text_frame and write structured entries using python-pptx API.
    
    entries: list of dicts with keys:
      text (str), size (float pt), bold (bool), italic (bool),
      color (RGBColor), bullet (bool)
    """
    # Fix body properties: ensure text stays inside the box
    bodyPr = text_frame._txBody.find(qn('a:bodyPr'))
    if bodyPr is not None:
        bodyPr.set('lIns', '91440')   # 0.1 inch left inset
        bodyPr.set('tIns', '45720')   # 0.05 inch top inset
        bodyPr.set('rIns', '91440')
        bodyPr.set('bIns', '45720')
        # Remove any autofit that might shrink/hide text
        for child in list(bodyPr):
            tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            if tag in ('normAutofit', 'spAutoFit'):
                bodyPr.remove(child)

    text_frame.word_wrap = True

    # Clear ALL existing paragraphs except the first
    txBody = text_frame._txBody
    ps = txBody.findall(qn('a:p'))
    for p_el in ps[1:]:
        txBody.remove(p_el)

    # Write entries
    for i, entry in enumerate(entries):
        if i == 0:
            para = text_frame.paragraphs[0]
        else:
            para = text_frame.add_paragraph()

        para.alignment = PP_ALIGN.LEFT

        # Clear any existing runs/endParaRPr from this paragraph
        p_el = para._p
        for child in list(p_el):
            tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            if tag in ('r', 'br', 'endParaRPr'):
                p_el.remove(child)

        text = entry.get('text', '')
        if entry.get('bullet'):
            text = '\u2022 ' + text

        run = para.add_run()
        run.text = text
        run.font.size = Pt(entry.get('size', 12))
        run.font.bold = entry.get('bold', False)
        run.font.italic = entry.get('italic', False)
        run.font.color.rgb = entry.get('color', BLACK)


def _find_shape(slide, text_pattern):
    for sh in slide.shapes:
        if sh.has_text_frame and text_pattern in sh.text_frame.text:
            return sh
    return None


# ===========================================================================
# Table cell helpers
# ===========================================================================

def _set_cell(cell, text, size=11, bold=False, color_rgb=BLACK):
    cell.text = ""
    p = cell.text_frame.paragraphs[0]
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color_rgb
    cell.text_frame.word_wrap = True


def _set_cell_fill(cell, rgb_tuple):
    tc = cell._tc
    tcPr = tc.find(qn("a:tcPr"))
    if tcPr is None:
        tcPr = etree.SubElement(tc, qn("a:tcPr"))
    for sf in tcPr.findall(qn("a:solidFill")):
        tcPr.remove(sf)
    for nl in tcPr.findall(qn("a:noFill")):
        tcPr.remove(nl)
    solidFill = etree.SubElement(tcPr, qn("a:solidFill"))
    srgbClr = etree.SubElement(solidFill, qn("a:srgbClr"))
    srgbClr.set("val", "%02X%02X%02X" % rgb_tuple)


# ===========================================================================
# FIGURE BUILDERS (matplotlib)
# ===========================================================================

def _build_architecture_png():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14); ax.set_ylim(0, 9); ax.axis("off")
    fig.patch.set_facecolor("#FAFAFA")
    C = {"blue":"#D6EAF8","green":"#D5F5E3","peach":"#FDEBD0","purple":"#E8DAEF",
         "pink":"#FDEDEC","tan":"#FEF9E7","teal":"#D1F2EB","border":"#2C3E50"}

    def box(x, y, w, h, title, body, color, fst=9, fsb=7.5):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.1",
                     facecolor=color,edgecolor=C["border"],linewidth=1.2,zorder=3))
        cx, cy = x+w/2, y+h-0.22
        ax.text(cx,cy,title,ha="center",va="top",fontsize=fst,fontweight="bold",color="#1A1A1A",zorder=4,multialignment="center")
        if body:
            ax.text(cx,cy-0.30,body,ha="center",va="top",fontsize=fsb,color="#333333",zorder=4,multialignment="center",linespacing=1.3)

    def arrow(x1,y1,x2,y2):
        ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle="-|>",color=C["border"],lw=1.5),zorder=2)

    ax.text(7,8.7,"ClimateTwin: Functional Architecture",ha="center",va="top",fontsize=14,fontweight="bold",color="#1D2F82")
    ax.text(7,8.35,"Lightweight Aurora-inspired urban climate digital twin for forecasting, scenarios, and risk assessment",ha="center",va="top",fontsize=8.5,color="#555555")

    box(0.3,6.4,3.2,1.6,"Atmospheric & Surface Data","ERA5-Land reanalysis grid cell\n(Delhi, 28.6\u00b0N 77.2\u00b0E)\n6 vars: t2m, d2m, sp, tp, u10, v10\n2018\u20132024 \u00b7 hourly \u00b7 61 368 timesteps",C["blue"])
    box(4.9,6.4,3.7,1.6,"Data Preparation","K\u2192\u00b0C, Pa\u2192hPa, m\u2192mm unit conversion\nChronological split: 5 yr train / 1 yr val / 1 yr test\nStandardScaler fit on train only\nWindow=24 h \u2192 next-step target",C["green"])
    box(9.9,6.4,3.6,1.6,"User Configuration","Location: Delhi\nForecast variables: all 6\nLead time: 1\u201324 steps\nOptional scenario: sustained \u0394",C["peach"])
    arrow(3.5,7.2,4.9,7.2); arrow(8.6,7.2,9.9,7.2)

    arrow(6.75,6.4,6.75,5.65)
    box(3.5,4.3,6.5,1.3,"Lightweight Aurora-Inspired Forecast Model","Linear proj \u2192 sinusoidal PE \u2192 Transformer Encoder (3L, d=128, 4-head, 399K params)\n\u2192 regression head \u2192 next 6-var state  |  LSTM variant: 2-layer h=64, 52K params\nTrained from scratch \u00b7 autoregressive rollout at 1/3/6/12/24-step horizons",C["purple"],fst=10,fsb=8)

    arrow(6.75,4.3,6.75,3.55); arrow(4.5,3.95,2.6,3.55); arrow(9.0,3.95,11.1,3.55)
    box(0.3,2.0,4.0,1.5,"Baseline Forecast","Persistence baseline (t+1 = t)\nLSTM & Transformer from scratch\nPer-var MAE/RMSE in physical units\n1-step test: 8 760 samples (2024)",C["blue"])
    box(4.8,2.0,4.0,1.5,"Scenario Engine","Physical-unit perturbation \u2192 re-scale\nSustained-delta autoregressive rollout\nMC-dropout uncertainty bands\nMandatory causal-language disclaimer",C["pink"])
    box(9.7,2.0,4.0,1.5,"Risk & Alert Layer","Heat Index (NOAA Rothfusz) + IMD categories\nIMD rainfall categories (24-h basis)\nBeaufort-scale wind classification\nColour-coded alert cards (red/orange/yellow)",C["tan"])

    arrow(2.3,2.0,3.8,1.18); arrow(6.8,2.0,6.8,1.18); arrow(11.7,2.0,9.8,1.18)
    box(1.5,0.15,11.0,1.0,"Urban Climate Digital Twin Interface  (Streamlit)","Climate Overview \u00b7 AI Forecast \u00b7 Multi-Step Forecast \u00b7 Model Comparison \u00b7 Scenario Sensitivity\nConsumes saved artifacts only \u00b7 no live retraining \u00b7 research-mode gate",C["teal"],fst=10,fsb=8)
    ax.text(7,0.0,"Research note: scenario outputs are sensitivity simulations of the learned model, not proof of physical causality.",ha="center",va="bottom",fontsize=7,color="#777777",style="italic")

    buf = io.BytesIO()
    fig.savefig(buf,format="png",dpi=160,bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close(fig); buf.seek(0)
    return buf.read()


def _build_training_curves_png():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lstm_log = json.loads((BASE/"results/metrics/lstm_delhi_training_log.json").read_text())
    tfr_log = json.loads((BASE/"results/metrics/transformer_scratch_delhi_training_log.json").read_text())

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(13,4.5)); fig.patch.set_facecolor("white")

    e_l = range(1, len(lstm_log["train_losses"])+1)
    ax1.plot(e_l,lstm_log["train_losses"],"b-",label="Train",lw=1.5)
    ax1.plot(e_l,lstm_log["val_losses"],"r-",label="Val",lw=1.5)
    ax1.axvline(lstm_log["best_epoch"],color="green",ls="--",alpha=0.7,label=f'Best={lstm_log["best_epoch"]}')
    ax1.scatter([lstm_log["best_epoch"]],[lstm_log["best_val_loss"]],color="green",zorder=5,s=60)
    ax1.set_title("LSTM (52K params, CPU)",fontsize=11,fontweight="bold",color="#1D2F82")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("MSE Loss"); ax1.legend(fontsize=8); ax1.set_ylim(0.05,0.40); ax1.grid(True,alpha=0.3)

    e_t = range(1,len(tfr_log["train_losses"])+1)
    ax2.plot(e_t,tfr_log["train_losses"],"b-",label="Train",lw=1.5)
    ax2.plot(e_t,tfr_log["val_losses"],"r-",label="Val",lw=1.5)
    ax2.axvline(tfr_log["best_epoch"],color="green",ls="--",alpha=0.7,label=f'Best={tfr_log["best_epoch"]}')
    ax2.scatter([tfr_log["best_epoch"]],[tfr_log["best_val_loss"]],color="green",zorder=5,s=60)
    ax2.set_title("Transformer (399K params, RTX 2050)",fontsize=11,fontweight="bold",color="#1D2F82")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("MSE Loss"); ax2.legend(fontsize=8); ax2.set_ylim(0.05,0.20); ax2.grid(True,alpha=0.3)

    fig.suptitle("Training Convergence",fontsize=13,fontweight="bold",color="#1D2F82",y=1.02)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf,format="png",dpi=160,bbox_inches="tight",facecolor="white")
    plt.close(fig); buf.seek(0)
    return buf.read()


def _build_comparison_chart_png():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    pers = json.loads((BASE/"results/metrics/persistence_delhi_metrics.json").read_text())
    lstm = json.loads((BASE/"results/metrics/lstm_delhi_metrics.json").read_text())
    tfr  = json.loads((BASE/"results/metrics/transformer_scratch_delhi_metrics.json").read_text())

    vl = ["t2m","d2m","sp","tp","u10","v10"]
    labels = ["t2m (\u00b0C)","d2m (\u00b0C)","sp (hPa)","tp (mm)","u10 (m/s)","v10 (m/s)"]
    pr = [pers["per_variable_metrics"][v]["rmse"] for v in vl]
    lr = [lstm["per_variable_metrics"][v]["rmse"] for v in vl]
    tr = [tfr["per_variable_metrics"][v]["rmse"] for v in vl]

    x = np.arange(len(vl)); w = 0.25
    fig, ax = plt.subplots(figsize=(13,5)); fig.patch.set_facecolor("white")
    b1 = ax.bar(x-w,pr,w,label="Persistence",color="#95A5A6",edgecolor="white")
    b2 = ax.bar(x,lr,w,label="LSTM (52K)",color="#3498DB",edgecolor="white")
    b3 = ax.bar(x+w,tr,w,label="Transformer (399K)",color="#E74C3C",edgecolor="white")
    for bars in [b1,b2,b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2,h+0.01,f"{h:.3f}",ha="center",va="bottom",fontsize=7.5)
    ax.set_xlabel("Variable"); ax.set_ylabel("RMSE")
    ax.set_title("1-Step Test RMSE \u2014 Per-Variable Model Comparison (Delhi 2024, n=8760)",fontsize=12,fontweight="bold",color="#1D2F82")
    ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=9); ax.legend(fontsize=9); ax.grid(axis="y",alpha=0.3); ax.set_ylim(0,1.5)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf,format="png",dpi=160,bbox_inches="tight",facecolor="white")
    plt.close(fig); buf.seek(0)
    return buf.read()


# ===========================================================================
# Main
# ===========================================================================

def main():
    prs = Presentation(str(TEMPLATE))
    slides = prs.slides

    # ===================================================================
    # SLIDE 1 — UNTOUCHED
    # ===================================================================

    # ===================================================================
    # SLIDE 2 — Introduction & Problem Recap
    # ===================================================================
    sl = slides[1]
    sh = _find_shape(sl, "Problem recap:")
    if sh:
        clear_and_write(sh.text_frame, [
            {"text": "Problem Recap", "size": 16, "bold": True, "color": DARK_BLUE},
            {"text": "Large-scale AI weather models (Aurora: 1.3B params; CREDIT: multi-model NWP) achieve SOTA global forecasts but require massive compute (multi-GPU clusters, petabyte datasets) \u2014 inaccessible for single-city or college-scale research.",
             "size": 12, "bullet": True},
            {"text": "No lightweight, reproducible framework adapts these ideas (pretrain\u2192fine-tune, standardized evaluation, autoregressive rollout) at a localized, laptop-trainable scale for Indian cities.",
             "size": 12, "bullet": True},
            {"text": "Existing digital-twin + ML work (Jyothi & Mesapam 2026) uses spatial/urban data (LST, NDVI) but not multivariate hourly weather time-series with sequence models.",
             "size": 12, "bullet": True},
            {"text": "", "size": 6},
            {"text": "Research Objectives", "size": 16, "bold": True, "color": DARK_BLUE},
            {"text": "RQ1 (Transfer Learning): Does multi-city pretrained Transformer improve Delhi forecast accuracy vs. scratch training on the same architecture?",
             "size": 12, "bullet": True},
            {"text": "RQ2 (Forecast Horizon): How does per-variable RMSE degrade across 1/3/6/12/24-step autoregressive horizons?",
             "size": 12, "bullet": True},
            {"text": "RQ3 (Architecture): Persistence vs. LSTM vs. Transformer (scratch) vs. Transformer (pretrained+fine-tuned) on 6 ERA5-Land variables for Delhi.",
             "size": 12, "bullet": True},
            {"text": "", "size": 6},
            {"text": "Refinements after Review-I", "size": 16, "bold": True, "color": DARK_BLUE},
            {"text": "Adopted ERA5-Land reanalysis grid cell terminology (not \"weather station\").",
             "size": 12, "bullet": True},
            {"text": "Separated n_features (6) and n_targets (6) in all model constructors for API consistency.",
             "size": 12, "bullet": True},
            {"text": "Machine-readable JSON run manifests recording seed, device, scaler path, git commit, software versions.",
             "size": 12, "bullet": True},
            {"text": "Mandatory scenario disclaimer: outputs are sensitivity experiments, not causal simulations.",
             "size": 12, "bullet": True},
        ])

    # ===================================================================
    # SLIDE 3 — Literature Survey
    # ===================================================================
    sl = slides[2]
    tbl_shape = None
    for sh in sl.shapes:
        if sh.has_table:
            tbl_shape = sh
            break

    if tbl_shape:
        tbl = tbl_shape.table
        headers = ["No.","Author(s) & Year","Title / Source","Method / Approach","Findings & Research Gap"]
        for ci, h in enumerate(headers):
            _set_cell(tbl.cell(0,ci),h,size=10,bold=True,color_rgb=WHITE)
            _set_cell_fill(tbl.cell(0,ci),HDR_FILL_T)

        lit = [
            ("1","Bodnar et al., 2025","A foundation model for the Earth system (Aurora) \u2014 Nature",
             "1.3B-param Perceiver + 3D Swin Transformer; global pretrain\u2192fine-tune on multi-source Earth data",
             "SOTA on air quality/waves/cyclone tracks/weather; needs massive compute \u2014 motivates lightweight single-city adaptation"),
            ("2","Schreck et al., 2025","CREDIT: a scalable framework for AI-driven Earth System Modeling \u2014 npj Climate & Atmos. Sci.",
             "Modular AI-NWP framework (WXFormer, FuXi); standardized preprocessing, training, multi-horizon evaluation",
             "Outperforms IFS HRES on 10-day forecasts; not tested at single-city/laptop scale \u2014 motivates ClimateTwin\u2019s scaled-down protocol"),
            ("3","Jyothi & Mesapam, 2026","Digital Twin for Climate-Resilient Urban Planning (UHI, Bengaluru) \u2014 ISPRS Archives",
             "Random Forest regression on LST/NDVI/NDBI/NDWI/albedo; CesiumJS 3D digital-twin platform",
             "Predicts 1\u20132\u00b0C LST rise by 2034; digital-twin+ML pattern for climate prediction, but spatial/urban not multivariate time-series"),
            ("4","Fang et al., 2021","AttEF: ConvLSTM Encoder-Forecaster with Attention for Precipitation Nowcasting \u2014 IASC",
             "Global-channel attention block (GCA) embedded in ConvLSTM encoder-forecaster",
             "Attention reduces blurring vs ConvLSTM/TrajGRU/PredRNN; supports attention-based encoders for future multi-horizon rollout (Phase 7)"),
            ("5","Geng et al., 2023","LSTMAtU-Net: A Precipitation Nowcasting Model Based on ECSA Module \u2014 Sensors",
             "U-Net + ConvLSTM (vertical flow) + Efficient Channel-Space Attention (ECSA); custom weighted loss",
             "Improves medium/high-intensity precipitation accuracy over ConvLSTM/PredRNN variants; reinforces multi-tier baseline comparison methodology"),
            ("6","Lim et al., 2021","Temporal Fusion Transformers for Interpretable Multi-horizon Forecasting \u2014 IJNN",
             "Gated Residual Networks + Variable Selection + Multi-head attention on past/future covariates",
             "Best-in-class on 9/12 benchmarks; interpretability via attention weights \u2014 informs dashboard design"),
        ]
        for ri, row_data in enumerate(lit):
            fill = ROW_ODD_T if ri % 2 == 0 else ROW_EVEN_T
            for ci, val in enumerate(row_data):
                _set_cell(tbl.cell(ri+1,ci),val,size=9)
                _set_cell_fill(tbl.cell(ri+1,ci),fill)

    fn = _find_shape(sl, "Minimum 15")
    if fn:
        clear_and_write(fn.text_frame, [
            {"text": "Papers 1\u20135 from provided survey (verbatim). Paper 6 informs dashboard design. Full 15-paper survey in docs/.",
             "size": 10, "italic": True, "color": GRAY}
        ])

    # ===================================================================
    # SLIDE 4 — System Design & Architecture
    # ===================================================================
    sl = slides[3]
    arch_png = _build_architecture_png()
    sl.shapes.add_picture(io.BytesIO(arch_png), Emu(400000), Emu(1000000),
                          prs.slide_width - Emu(800000), Emu(5400000))

    # ===================================================================
    # SLIDE 5 — Implementation Details
    # ===================================================================
    sl = slides[4]
    sh = _find_shape(sl, "Modules developed")
    if sh:
        clear_and_write(sh.text_frame, [
            {"text": "Modules Developed & Integrated (src/)", "size": 14, "bold": True, "color": DARK_BLUE},
            {"text": "config.py \u2014 YAML config loader with typed validation; enforces n_features/n_targets consistency",
             "size": 11, "bullet": True},
            {"text": "preprocessing.py \u2014 unit conversion (K\u2192\u00b0C, Pa\u2192hPa, m\u2192mm), chronological split, StandardScaler (train-only fit), scaler persistence",
             "size": 11, "bullet": True},
            {"text": "dataset.py \u2014 boundary-safe sliding-window (w=24) sequence construction; PyTorch Dataset/DataLoader",
             "size": 11, "bullet": True},
            {"text": "models/transformer.py \u2014 ClimateTransformer (d=128, 4-head, 3-layer encoder, sinusoidal PE, 399K params)",
             "size": 11, "bullet": True},
            {"text": "models/lstm.py \u2014 ClimateLSTM (2-layer, h=64, 52K params); batch-first",
             "size": 11, "bullet": True},
            {"text": "models/persistence.py \u2014 prediction(t+1) = state(t); uses same evaluation pipeline as learned models",
             "size": 11, "bullet": True},
            {"text": "evaluate.py \u2014 per-variable MAE/RMSE in original units after inverse-transform; metric JSON I/O",
             "size": 11, "bullet": True},
            {"text": "scenario.py \u2014 unit-aware perturbation engine with sustained-delta rollout and MC-dropout uncertainty",
             "size": 11, "bullet": True},
            {"text": "impact.py \u2014 NOAA Heat Index, IMD heatwave & rainfall categories, Beaufort wind scale, colour-coded alerts",
             "size": 11, "bullet": True},
            {"text": "manifest.py \u2014 JSON run manifests: run_id, seed, device, software versions, git commit",
             "size": 11, "bullet": True},
            {"text": "", "size": 6},
            {"text": "Data Pipeline (61 368 hourly timesteps)", "size": 14, "bold": True, "color": DARK_BLUE},
            {"text": "ERA5-Land CDS API \u2192 3 NetCDFs \u2192 unit conversion \u2192 train 2018\u20132022 (43 824) | val 2023 (8 760) | test 2024 (8 784) \u2192 StandardScaler + 6 CSVs + scaler.joblib",
             "size": 11, "bullet": True},
            {"text": "1 132 negative precipitation values clamped to zero; 0 missing timestamps; 0 duplicates",
             "size": 11, "bullet": True},
            {"text": "", "size": 6},
            {"text": "Tech Stack: Python 3.13 \u00b7 PyTorch 2.14 \u00b7 NumPy \u00b7 pandas \u00b7 scikit-learn \u00b7 Streamlit \u00b7 matplotlib \u00b7 pytest \u00b7 Ruff",
             "size": 11, "bold": True, "color": DARK_BLUE},
        ])

    # ===================================================================
    # SLIDE 6 — Results & Analysis (75%)
    # ===================================================================
    sl = slides[5]
    chart_png = _build_comparison_chart_png()
    sl.shapes.add_picture(io.BytesIO(chart_png), Emu(350000), Emu(1100000),
                          Emu(11400000), Emu(3200000))

    txBox = sl.shapes.add_textbox(Emu(548640), Emu(4400000), Emu(10881360), Emu(2200000))
    clear_and_write(txBox.text_frame, [
        {"text": "Key Findings (1-Step, Test Year 2024, n = 8 760)", "size": 13, "bold": True, "color": DARK_BLUE},
        {"text": "Transformer: lowest RMSE on t2m (0.577 \u00b0C, \u221254.7% vs Persistence), d2m (0.643 \u00b0C), sp (0.395 hPa, \u221221.7%).",
         "size": 11, "bullet": True},
        {"text": "LSTM: wins on tp (0.379 mm), u10 (0.413 m/s), v10 (0.399 m/s) \u2014 recurrent memory suits sparse/wind vars.",
         "size": 11, "bullet": True},
        {"text": "Precipitation hardest: Persistence RMSE (0.384 mm) competitive with LSTM, beats Transformer (0.404) \u2014 rain event sparsity.",
         "size": 11, "bullet": True},
        {"text": "Training: Transformer epoch 8/18 (154 s, GPU, 163 MB peak); LSTM epoch 21/31 (171 s, CPU).",
         "size": 11, "bullet": True},
    ])

    # ===================================================================
    # SLIDE 7 — Results contd. (training curves + scenario figures)
    # ===================================================================
    sl = slides[6]
    curves_png = _build_training_curves_png()
    sl.shapes.add_picture(io.BytesIO(curves_png), Emu(350000), Emu(1100000),
                          Emu(11400000), Emu(2800000))
    hw_path = str(BASE / "results/figures/scenario_heatwave_t2m_delhi.png")
    sl.shapes.add_picture(hw_path, Emu(350000), Emu(4000000), Emu(5500000), Emu(2500000))
    rf_path = str(BASE / "results/figures/scenario_rainfall_tp_delhi.png")
    sl.shapes.add_picture(rf_path, Emu(6000000), Emu(4000000), Emu(5700000), Emu(2500000))

    # ===================================================================
    # SLIDE 8 — Challenges & Remaining Work
    # ===================================================================
    sl = slides[7]
    sh = _find_shape(sl, "Challenges faced")
    if sh:
        clear_and_write(sh.text_frame, [
            {"text": "Challenges Faced & Solutions", "size": 14, "bold": True, "color": DARK_BLUE},
            {"text": "Build system bug: pyproject.toml invalid build-backend \u2192 fixed to setuptools.build_meta",
             "size": 11, "bullet": True},
            {"text": "CDS API integration: coordinate mismatch in download script \u2192 added config fallback for latitude/longitude keys",
             "size": 11, "bullet": True},
            {"text": "Autoregressive instability: 48h rollouts showed phase-shift artifacts \u2192 optimized to 12h horizon with paired MC-dropout",
             "size": 11, "bullet": True},
            {"text": "Scenario amplification: sustained perturbation mode caused runaway effects (+4\u00b0C \u2192 +18\u00b0C) \u2192 reverted to seed-only mode",
             "size": 11, "bullet": True},
            {"text": "", "size": 6},
            {"text": "Remaining Work (25%)", "size": 14, "bold": True, "color": DARK_BLUE},
            {"text": "Multi-city expansion: Mumbai, Bangalore, Kolkata, Chennai (pipeline ready, training pending)",
             "size": 11, "bullet": True},
            {"text": "Streamlit dashboard implementation with interactive scenario controls",
             "size": 11, "bullet": True},
            {"text": "Transformer model training and comparison with LSTM baseline",
             "size": 11, "bullet": True},
            {"text": "Formal uncertainty calibration metrics (PICP, CRPS) evaluation",
             "size": 11, "bullet": True},
            {"text": "", "size": 6},
            {"text": "Timeline for Completion by Review-III (28.10.2026)", "size": 14, "bold": True, "color": DARK_BLUE},
            {"text": "Week 1\u20132: Multi-city training pipeline execution",
             "size": 11, "bullet": True},
            {"text": "Week 3: Dashboard implementation with scenario interface",
             "size": 11, "bullet": True},
            {"text": "Week 4: Transformer training and comparative analysis",
             "size": 11, "bullet": True},
            {"text": "Final week: Documentation and presentation preparation",
             "size": 11, "bullet": True},
        ])

    # ===================================================================
    # SLIDE 9 — References
    # ===================================================================
    sl = slides[8]
    sh = _find_shape(sl, "[1]")
    if sh:
        clear_and_write(sh.text_frame, [
            {"text": '[1] C. Bodnar et al., "Aurora: A foundation model of the atmosphere," Nature, vol. 637, 2025.', "size": 11},
            {"text": '[2] J. Schreck et al., "CREDIT: A scalable framework for AI-driven Earth System Modeling," npj Clim. Atmos. Sci., vol. 8, 2025.', "size": 11},
            {"text": '[3] S. Jyothi and P. Mesapam, "Digital twin for climate-resilient urban planning," ISPRS Archives, LV-4, 2026.', "size": 11},
            {"text": '[4] X. Fang et al., "AttEF: ConvLSTM encoder-forecaster with attention for precipitation nowcasting," IASC, 2021.', "size": 11},
            {"text": '[5] R. Geng et al., "LSTMAtU-Net: A precipitation nowcasting model based on ECSA module," Sensors, vol. 23, 2023.', "size": 11},
            {"text": '[6] B. Lim et al., "Temporal Fusion Transformers for multi-horizon time series forecasting," Int. J. Forecast., vol. 37, 2021.', "size": 11},
            {"text": '[7] A. Vaswani et al., "Attention is all you need," NeurIPS, vol. 30, 2017.', "size": 11},
            {"text": '[8] S. Hochreiter and J. Schmidhuber, "Long short-term memory," Neural Comput., vol. 9, pp. 1735-1780, 1997.', "size": 11},
            {"text": '[9] ECMWF, "ERA5-Land hourly data," Copernicus CDS, 2024. DOI: 10.24381/cds.e2161bac', "size": 11},
            {"text": '[10] F. Pedregosa et al., "Scikit-learn: ML in Python," JMLR, vol. 12, pp. 2825-2830, 2011.', "size": 11},
            {"text": '[11] N. Rothfusz, "The Heat Index equation," NOAA NWS SR 90-23, 1990.', "size": 11},
            {"text": '[12] IMD, "Heat wave criteria," National Climate Data Centre, 2026.', "size": 11},
        ])

    fn = _find_shape(sl, "IEEE format")
    if fn:
        clear_and_write(fn.text_frame, [
            {"text": "IEEE format. All cited in slides. ERA5-Land data under Copernicus CC-BY 4.0.",
             "size": 10, "italic": True, "color": GRAY}
        ])

    # ===================================================================
    # SLIDE 10 — Thank You (untouched)
    # ===================================================================

    prs.save(str(OUTPUT))
    print(f"Saved: {OUTPUT}")
    print(f"Size: {OUTPUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
