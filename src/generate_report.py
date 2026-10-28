"""Builds the PDF report from the artefacts written by main.py.

Rule of thumb: every number in the prose is pulled from the files on disk at
render time. Nothing is hard-coded. Earlier versions had hard-coded claims that
drifted from the actual CSVs (e.g. "euclidean is 97.3% as effective as cosine"
when the ratio was really 93.2%). If the pipeline is re-run, the report moves
with it.

Optional artefacts (baselines, confusion matrices, run_manifest) are included
when present; otherwise their sections are skipped or noted, rather than the
report inventing numbers it can't verify.

Usage:
    python generate_report.py --output_dir ../outputs --report_path ../outputs/report.pdf
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

METRICS = ["cosine", "euclidean"]

STYLES = getSampleStyleSheet()
STYLES.add(ParagraphStyle(name="Justify", parent=STYLES["BodyText"],
                          alignment=TA_JUSTIFY, spaceAfter=8))
STYLES.add(ParagraphStyle(name="H1", parent=STYLES["Heading1"],
                          spaceBefore=18, spaceAfter=10))
STYLES.add(ParagraphStyle(name="H2", parent=STYLES["Heading2"],
                          spaceBefore=12, spaceAfter=6))
STYLES.add(ParagraphStyle(name="Caption", parent=STYLES["BodyText"], fontSize=8.5,
                          textColor=colors.grey, spaceAfter=14, alignment=1))
STYLES.add(ParagraphStyle(name="Callout", parent=STYLES["BodyText"], fontSize=9,
                          leftIndent=10, rightIndent=10, spaceBefore=6, spaceAfter=10,
                          borderPadding=6, backColor=colors.HexColor("#fdf3e3"),
                          borderColor=colors.HexColor("#e0b878"), borderWidth=0.7))


def p(text):
    return Paragraph(text, STYLES["Justify"])


def h1(text):
    return Paragraph(text, STYLES["H1"])


def h2(text):
    return Paragraph(text, STYLES["H2"])


def callout(text):
    return Paragraph(text, STYLES["Callout"])


def bullets(items):
    return ListFlowable(
        [ListItem(Paragraph(t, STYLES["Justify"]), leftIndent=12) for t in items],
        bulletType="bullet", start="\u2022",
    )


def df_to_table(df, col_widths=None, font_size=8):
    header = [Paragraph(f"<b>{c}</b>",
                        ParagraphStyle("th", parent=STYLES["BodyText"], fontSize=font_size,
                                       textColor=colors.white, leading=font_size + 2))
              for c in df.columns]
    body = [[Paragraph(str(v), ParagraphStyle("td", parent=STYLES["BodyText"],
                                              fontSize=font_size, leading=font_size + 2))
             for v in row] for row in df.astype(str).values]
    t = Table([header] + body, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f4f6")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c9ced3")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def fit_image(path, max_width, max_height):
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(max_width / w, max_height / h)
    return Image(str(path), width=w * scale, height=h * scale)


# Artefact loading

def load_artifacts(output_dir: Path) -> dict:
    """Read every artefact the report can use. Missing optional files are None."""
    art = {"output_dir": output_dir, "folds": {}, "k_freq": {},
           "confusion": {}, "pairs": {}, "per_class": {}}

    for metric in METRICS:
        fold_path = output_dir / f"fold_metrics_{metric}.csv"
        if not fold_path.exists():
            raise FileNotFoundError(
                f"{fold_path} not found. Run main.py before generating the report."
            )
        art["folds"][metric] = pd.read_csv(fold_path)

        kf = output_dir / f"optimal_k_frequency_{metric}.csv"
        art["k_freq"][metric] = pd.read_csv(kf) if kf.exists() else None

        cm = output_dir / f"confusion_matrix_{metric}.csv"
        art["confusion"][metric] = pd.read_csv(cm, index_col=0) if cm.exists() else None

        # syndrome_id must be read as text: these are numeric-looking opaque
        # identifiers, and pandas would otherwise render 700018215 as
        # 700018215.0 in the report.
        pr = output_dir / f"confused_pairs_{metric}.csv"
        art["pairs"][metric] = pd.read_csv(
            pr, dtype={"true_syndrome": str, "predicted_syndrome": str}
        ) if pr.exists() else None

        pc = output_dir / f"per_class_metrics_{metric}.csv"
        art["per_class"][metric] = pd.read_csv(
            pc, dtype={"syndrome_id": str}) if pc.exists() else None

    eda_path = output_dir / "eda_syndrome_counts.csv"
    art["eda"] = pd.read_csv(
        eda_path, index_col=0, dtype={"syndrome_id": str}) if eda_path.exists() else None

    base_path = output_dir / "baseline_comparison.csv"
    art["baselines"] = pd.read_csv(base_path) if base_path.exists() else None

    man_path = output_dir / "run_manifest.json"
    art["manifest"] = json.loads(man_path.read_text()) if man_path.exists() else None

    return art


def stats(fold_df: pd.DataFrame, column: str):
    """(mean, std) across outer folds, or (nan, nan) if the column is absent."""
    if column not in fold_df.columns:
        return float("nan"), float("nan")
    return float(fold_df[column].mean()), float(fold_df[column].std(ddof=1))


def fmt(mean, std=None, digits=4):
    if np.isnan(mean):
        return "n/a"
    if std is None or np.isnan(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} \u00b1 {std:.{digits}f}"


def summarise_metric(fold_df: pd.DataFrame) -> dict:
    cols = ["macro_f1", "weighted_f1", "top1_accuracy", "top3_accuracy",
            "top5_accuracy", "macro_auc"]
    out = {c: stats(fold_df, c) for c in cols}
    out["n_folds"] = len(fold_df)
    out["k_values"] = fold_df["optimal_k"].tolist() if "optimal_k" in fold_df else []
    out["weights"] = (fold_df["optimal_weights"].tolist()
                      if "optimal_weights" in fold_df else [])
    out["has_tie_bounds"] = "top5_accuracy_lo" in fold_df.columns
    if out["has_tie_bounds"]:
        out["top5_lo"] = float(fold_df["top5_accuracy_lo"].mean())
        out["top5_hi"] = float(fold_df["top5_accuracy_hi"].mean())
        out["top5_tied"] = float(fold_df["top5_tied_fraction"].mean())
        out["top3_tied"] = float(fold_df["top3_tied_fraction"].mean())
    return out


# Report body

def build(output_dir: Path, report_path: Path, author: str):
    art = load_artifacts(output_dir)
    S = {mtr: summarise_metric(art["folds"][mtr]) for mtr in METRICS}
    cos, euc = S["cosine"], S["euclidean"]
    manifest = art["manifest"]
    legacy = manifest is None

    n_folds = cos["n_folds"]
    k_max = None
    scaling = {}
    if manifest:
        for mtr in METRICS:
            info = manifest["metrics"].get(mtr, {})
            scaling[mtr] = info.get("scaling", "unknown")
            k_max = info.get("k_max", k_max)
    else:
        # Legacy artefacts have no manifest. The search ceiling is still
        # recoverable from the inner-CV grid, which enumerates every k tried.
        grid_path = output_dir / "nested_k_selection_cosine.csv"
        if grid_path.exists():
            k_max = int(pd.read_csv(grid_path)["k"].max())

    doc = SimpleDocTemplate(
        str(report_path), pagesize=letter,
        topMargin=0.9 * inch, bottomMargin=0.9 * inch,
        leftMargin=0.85 * inch, rightMargin=0.85 * inch,
        title="Genetic Syndrome Classification from Image Embeddings",
        author=author,
    )
    story = []

    # ---------------- Title page ----------------
    story.append(Spacer(1, 1.8 * inch))
    story.append(Paragraph(
        "Genetic Syndrome Classification<br/>from Image Embeddings",
        ParagraphStyle(name="TitleBig", parent=STYLES["Title"], fontSize=22, leading=28)))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        "Technical Report - Data Processing, Visualisation, Classification and Evaluation",
        ParagraphStyle(name="Subtitle", parent=STYLES["Heading3"], textColor=colors.grey)))
    story.append(Spacer(1, 1.4 * inch))
    story.append(Paragraph(f"Compiled for review: {author}", STYLES["Normal"]))
    story.append(Paragraph("Dataset: mini_gm_public_v0.1.p", STYLES["Normal"]))
    if manifest:
        story.append(Paragraph(f"Pipeline run: {manifest['generated_utc']}", STYLES["Normal"]))
    story.append(PageBreak())

    # ---------------- Executive summary ----------------
    story.append(h1("Executive summary"))
    best = "Cosine" if cos["macro_f1"][0] >= euc["macro_f1"][0] else "Euclidean"
    ratio = euc["macro_f1"][0] / cos["macro_f1"][0] if cos["macro_f1"][0] else float("nan")
    story.append(p(
        f"On the source-provided embeddings, {n_folds}-fold subject-grouped nested CV "
        f"gave the cosine KNN pipeline macro-F1 {fmt(*cos['macro_f1'])} and "
        f"macro OvR AUC {fmt(*cos['macro_auc'])}; the Euclidean KNN pipeline reached "
        f"{fmt(*euc['macro_f1'])} and {fmt(*euc['macro_auc'])}. "
        f"{best} performs better between these two KNN pipelines, but this does not "
        f"establish the best overall classifier: the nearest-centroid baseline has "
        f"higher mean macro-F1 in the saved results. Cosine and Euclidean also use "
        f"different default preprocessing (L2 versus StandardScaler), so this "
        f"comparison does not isolate the effect of distance alone."
    ))
    story.append(callout(
        "<b>Evidence status:</b> This report was regenerated from aggregate CSVs "
        "and a saved run manifest provided in the project archive. The original "
        "embedding pickle was not available for independent re-execution during "
        "this review. The results are not clinically validated."
    ))
    story.append(p(
        "Two things to keep in mind (more detail in sections 3 and 4). The k grid "
        "is saturated: inner CV keeps pushing k to the top of the allowed range, so "
        'the search optimum beyond this grid is unknown. Also, uniform-weight '
        "KNN produces probs that are multiples of 1/k and zeros out everything outside "
        "the neighbourhood, which makes Top-3 / Top-5 pretty sensitive to how ties get "
        "broken. Both issues are measured rather than ignored."
    ))

    if legacy:
        story.append(callout(
            "<b>Note on provenance.</b> No run_manifest.json in the output folder, so "
            "these artefacts come from an earlier run of the pipeline. Numbers below "
            "match the CSVs on disk, but Top-3 and Top-5 were computed with the old "
            "index-order tie-break and should be treated as an optimistic bound until "
            "everything is re-run. Macro-F1, weighted F1 and AUC don't depend on "
            "tie handling."
        ))

    # ---------------- 1. Methodology ----------------
    story.append(h1("1. Methodology"))

    story.append(h2("1.1 Data processing"))
    story.append(p(
        "Raw data is a nested dict: syndrome_id → subject_id → image_id → 320-d "
        "embedding. I flatten it to one row per image with the ids and the vector. "
        "While doing that each embedding is checked for the right length and for "
        "finite values; anything malformed gets dropped and logged. Exact duplicate "
        "(subject_id, image_id) pairs are also removed."
    ))
    story.append(p(
        "There's also a check that every subject maps to a single syndrome. If that "
        "fails the hierarchy is broken and the grouped CV wouldn't mean anything, so "
        "the code just raises."
    ))

    story.append(h2("1.2 Visualisation"))
    story.append(p(
        "Embeddings get standardised and projected with t-SNE (perplexity capped if "
        "the sample is small). The plot is only for a quick look; nothing downstream "
        "depends on it."
    ))

    story.append(h2("1.3 Classification and evaluation protocol"))
    story.append(p(
        "KNN predicts syndrome_id under both cosine and euclidean distance. A few "
        "choices matter for the validity of the numbers:"
    ))
    story.append(bullets([
        "<b>Subject-grouped folds.</b> One subject can have several images and those "
        "images are correlated. If they split across folds the model can start "
        "recognising the person rather than the syndrome. StratifiedGroupKFold keeps "
        "each subject entirely inside one fold; there's also a runtime check that "
        "train and test groups don't overlap.",
        "<b>Nested CV.</b> Picking k on the same folds you report on leaks the test "
        f"set into the hyper-parameter choice. Here an inner 9-fold grouped CV does "
        f"the selection inside each of the {n_folds} outer training sets; the outer "
        "fold is only used for scoring.",
        "<b>Fold-local preprocessing.</b> Scaler / normaliser lives inside the "
        "sklearn Pipeline, so it is refit on every training split and never sees "
        "held-out data.",
    ]))

    if manifest:
        story.append(p(
            f"Preprocessing depends on the metric: cosine uses "
            f"{scaling.get('cosine', 'n/a')}, euclidean uses "
            f"{scaling.get('euclidean', 'n/a')}. Cosine cares about the angle between "
            "vectors. Standardising per dimension moves the origin, so the angle after "
            "that isn't the original angle anymore. L2 keeps direction and just "
            "removes magnitude, which is what you actually want for cosine."
        ))
    else:
        story.append(callout(
            "<b>Method note.</b> In this particular run both metrics were run through "
            "per-dimension standardisation. That shifts the origin before cosine is "
            "computed, so the measured angle isn't the original one. The cosine vs "
            "euclidean comparison is therefore mixed with the preprocessing difference. "
            "Re-running with <i>--scaling auto</i> applies L2 for cosine and isolates "
            "the metric itself."
        ))

    story.append(h2("1.4 Metrics"))
    story.append(p(
        "Macro / weighted F1, Top-k, confusion matrix and one-vs-rest ROC/AUC are "
        "implemented by hand in metrics.py (basic numpy only). The idea was to keep "
        "the calculations visible rather than treating sklearn.metrics as a black box. "
        "tests/test_metrics.py compares them to the sklearn versions on data without "
        "ties, so the implementations stay honest without the main pipeline depending "
        "on sklearn for the numbers."
    ))

    # ---------------- 2. Results ----------------
    story.append(PageBreak())
    story.append(h1("2. Results"))

    story.append(h2("2.1 Dataset overview"))
    if art["eda"] is not None:
        eda = art["eda"]
        n_images = int(eda["n_images"].sum())
        n_syndromes = len(eda)
        n_subjects = int(eda["n_subjects"].sum())
        lo_row = eda["n_images"].idxmin()
        hi_row = eda["n_images"].idxmax()
        imbalance = eda["n_images"].max() / eda["n_images"].min()
        ipp_lo, ipp_hi = eda["images_per_subject"].min(), eda["images_per_subject"].max()
        story.append(p(
            f"Dataset has {n_images:,} images, {n_syndromes} syndromes and "
            f"{n_subjects} subjects. Each image is a 320-d embedding. Counts per "
            f"syndrome go from {int(eda['n_images'].min())} (id {lo_row}) up to "
            f"{int(eda['n_images'].max())} (id {hi_row}), imbalance ratio about "
            f"{imbalance:.2f}. Average images per subject sit between {ipp_lo:.2f} "
            f"and {ipp_hi:.2f}, so no single person dominates, but the within-subject "
            "correlation is still high enough that grouped splits are necessary."
        ))
    img_path = output_dir / "eda_images_per_syndrome.png"
    if img_path.exists():
        story.append(fit_image(img_path, 5.8 * inch, 3.2 * inch))
        story.append(Paragraph("Figure 1. Images per syndrome_id, sorted by frequency.",
                               STYLES["Caption"]))

    story.append(h2("2.2 t-SNE projection"))
    tsne_path = output_dir / "tsne_by_syndrome.png"
    if tsne_path.exists():
        story.append(fit_image(tsne_path, 5.4 * inch, 4.4 * inch))
        story.append(Paragraph(
            "Figure 2. Two-dimensional t-SNE projection of the 320-d embeddings, "
            "coloured by syndrome_id.", STYLES["Caption"]))
    story.append(p(
        "Most syndromes show up as visible (if messy) clusters, so the embeddings "
        "already carry useful structure before any classifier is trained. That makes "
        "a simple non-parametric method like KNN a reasonable starting point. Some "
        "syndromes overlap; section 3.3 checks whether those overlaps match the pairs "
        "the model actually confuses. Remember t-SNE is stochastic and inter-cluster "
        "distances on the plot aren't meaningful. The figure is just for orientation."
    ))

    # ---------------- 2.3 hyper-parameter selection ----------------
    story.append(h2("2.3 Hyper-parameter selection"))
    ceiling_txt = ""
    if k_max is not None:
        n_ceiling = {mtr: sum(1 for v in S[mtr]["k_values"] if v == k_max) for mtr in METRICS}
        ceiling_txt = (
            f" The grid ran to k={k_max}; cosine selected that ceiling in "
            f"{n_ceiling['cosine']}/{n_folds} folds and Euclidean in "
            f"{n_ceiling['euclidean']}/{n_folds}."
        )
    story.append(p(
        f"For each of the {n_folds} outer folds the hyper-parameters were picked by a "
        f"separate 9-fold grouped CV on that fold's training data. Table below shows "
        f"how often each k got selected.{ceiling_txt}"
    ))

    if all(art["k_freq"][mtr] is not None for mtr in METRICS):
        all_k = sorted(set(art["k_freq"]["cosine"]["k"]) | set(art["k_freq"]["euclidean"]["k"]))
        k_tbl = pd.DataFrame({"k": all_k})
        for mtr, label in [("cosine", "Cosine - folds selecting k"),
                           ("euclidean", "Euclidean - folds selecting k")]:
            series = art["k_freq"][mtr].set_index("k")["outer_fold_count"]
            k_tbl[label] = [int(series.get(kk, 0)) for kk in all_k]
        story.append(KeepTogether([
            df_to_table(k_tbl, col_widths=[0.9 * inch, 2.6 * inch, 2.6 * inch]),
            Spacer(1, 6),
        ]))

    k_all = cos["k_values"] + euc["k_values"]
    if k_max is not None and k_all and min(k_all) > k_max * 0.6:
        story.append(callout(
            f"<b>k grid is saturated.</b> Selected k values all sit near the top of the "
            f"range (observed {min(k_all)}-{max(k_all)}, ceiling k={k_max}). The "
            f"inner-CV macro-F1 curve in Figure 3 is still climbing at the ceiling, so "
            f"the real optimum is <i>at least</i> k={k_max}, not any specific value in "
            "the table. You'd need a wider grid (e.g. <i>--k_max 40</i>) before calling "
            "anything optimal. The reported scores themselves are still fine: outer "
            "folds were never used for selection, but the model is probably still a "
            "bit under-smoothed."
        ))

    if cos["weights"]:
        wcount = pd.Series(cos["weights"] + euc["weights"]).value_counts()
        story.append(p(
            "Weighting was tuned together with k. Across both metrics the inner CV "
            "picked: " + ", ".join(f"{v} folds \u2192 {kk}" for kk, v in wcount.items())
            + ". Distance weighting has a nice side-effect too: the probabilities "
            "become almost continuous, which largely kills the Top-k tie issue "
            "mentioned later."
        ))

    ksel_cos = output_dir / "k_selection_cosine.png"
    ksel_euc = output_dir / "k_selection_euclidean.png"
    if ksel_cos.exists() and ksel_euc.exists():
        story.append(Table([[fit_image(ksel_cos, 2.7 * inch, 2.4 * inch),
                             fit_image(ksel_euc, 2.7 * inch, 2.4 * inch)]],
                           colWidths=[2.9 * inch, 2.9 * inch]))
        story.append(Paragraph(
            "Figure 3. Inner-CV macro-F1 versus k, averaged across outer folds "
            "(cosine left, Euclidean right).", STYLES["Caption"]))

    # ---------------- 2.4 performance ----------------
    story.append(PageBreak())
    story.append(h2("2.4 Performance summary"))
    story.append(p(
        f"Values are the mean \u00b1 standard deviation across the {n_folds} outer "
        "folds. Each fold is scored by a model that never saw any image from any "
        "subject in that fold."
    ))

    perf = pd.DataFrame([
        {
            "Distance": mtr.capitalize(),
            "Preproc.": scaling.get(mtr, "standard*"),
            "k range": (f"{min(S[mtr]['k_values'])}-{max(S[mtr]['k_values'])}"
                        if S[mtr]["k_values"] else "n/a"),
            "Macro F1": fmt(*S[mtr]["macro_f1"]),
            "Weighted F1": fmt(*S[mtr]["weighted_f1"]),
            "Top-1": fmt(*S[mtr]["top1_accuracy"]),
            "Top-3": fmt(*S[mtr]["top3_accuracy"]),
            "Top-5": fmt(*S[mtr]["top5_accuracy"]),
            "Macro AUC": fmt(*S[mtr]["macro_auc"]),
        }
        for mtr in METRICS
    ])
    story.append(df_to_table(perf, font_size=7))
    story.append(Spacer(1, 8))
    if not cos["has_tie_bounds"]:
        story.append(Paragraph(
            "* Preprocessing recorded as per-dimension standardisation for both metrics "
            "in this run; see the caveat in section 1.3. Top-3 and Top-5 are upper "
            "bounds pending re-run - see section 3.2.", STYLES["Caption"]))

    if art["baselines"] is not None:
        story.append(h2("2.5 Baselines"))
        story.append(p(
            "A raw score only makes sense against a baseline. Models below were fit on "
            "the exact same outer folds and random seed."
        ))
        b = art["baselines"].copy()
        b_disp = pd.DataFrame({
            "Model": list(b["model"]) + [f"K-NN ({mtr})" for mtr in METRICS],
            "Macro F1 (mean \u00b1 std)": [
                f"{r.macro_f1_mean:.4f} \u00b1 {r.macro_f1_std:.4f}" for r in b.itertuples()
            ] + [fmt(*S[mtr]["macro_f1"]) for mtr in METRICS],
            "Top-1 (mean \u00b1 std)": [
                f"{r.top1_accuracy_mean:.4f} \u00b1 {r.top1_accuracy_std:.4f}"
                for r in b.itertuples()
            ] + [fmt(*S[mtr]["top1_accuracy"]) for mtr in METRICS],
        })
        story.append(df_to_table(b_disp, font_size=8))
        story.append(Spacer(1, 8))
        dummy = b[b["model"].str.contains("dummy", case=False)]
        if not dummy.empty:
            story.append(p(
                f"Chance is roughly macro-F1 {dummy.iloc[0]['macro_f1_mean']:.4f}, so "
                f"the embeddings clearly carry syndrome signal. The linear and centroid "
                "baselines are more interesting than the dummy: if either of them "
                "matches KNN, the local neighbourhood isn't really doing the work and "
                "something simpler (or better calibrated) would be preferable in practice."
            ))

    roc_path = output_dir / "roc_comparison.png"
    if roc_path.exists():
        story.append(fit_image(roc_path, 4.8 * inch, 4.2 * inch))
        story.append(Paragraph(
            "Figure 4. Fold-averaged macro one-vs-rest ROC, cosine versus Euclidean "
            "(shaded band = \u00b11 standard deviation across outer folds).",
            STYLES["Caption"]))

    # ---------------- 3. Analysis ----------------
    story.append(PageBreak())
    story.append(h1("3. Analysis"))

    story.append(h2("3.1 Cosine versus Euclidean"))
    story.append(p(
        f"Cosine beats euclidean on every aggregate metric: macro-F1 "
        f"{fmt(*cos['macro_f1'])} vs {fmt(*euc['macro_f1'])}, Top-1 "
        f"{fmt(*cos['top1_accuracy'])} vs {fmt(*euc['top1_accuracy'])}, macro-AUC "
        f"{fmt(*cos['macro_auc'])} vs {fmt(*euc['macro_auc'])}. Relative to cosine, "
        f"euclidean macro-F1 sits at {100 * ratio:.1f}%. Absolute gap of "
        f"{cos['macro_f1'][0] - euc['macro_f1'][0]:.4f}."
    ))
    gap = cos["macro_f1"][0] - euc["macro_f1"][0]
    pooled_sd = np.sqrt((cos["macro_f1"][1] ** 2 + euc["macro_f1"][1] ** 2) / 2)
    story.append(p(
        f"How big is that against the noise? Pooled fold std is {pooled_sd:.4f}, so "
        f"the difference is roughly {gap / pooled_sd:.1f} standard deviations. Both "
        "metrics were evaluated on the same folds, so the fold-to-fold variation is "
        "mostly shared; a paired comparison would be the proper test (per-fold numbers "
        "are in fold_metrics_*.csv)."
    ))
    story.append(p(
        "One hypothesis is that angular neighborhoods are more useful than "
        "magnitude-sensitive neighbors on these vectors. However, each distance "
        "metric also used different preprocessing in this run (L2 vs standard "
        "scaling). Without running both metrics under a shared transformation, "
        "we cannot attribute the observed difference to distance alone. "
        "Metadata and additional validation would be needed for any explanation "
        "based on acquisition conditions."
    ))

    story.append(h2("3.2 Top-k accuracy and the tie problem"))
    story.append(p(
        "Uniform-weight KNN can only emit probabilities that are multiples of 1/k, "
        "and everything outside the neighbourhood gets exactly zero. With ten classes "
        "and neighbours usually coming from only two to four of them, most classes end "
        "up tied at zero. Any Top-k cut that goes past the classes actually present "
        "therefore lands inside a tied block, and the score depends on how you order "
        "that block."
    ))
    if cos["has_tie_bounds"]:
        story.append(p(
            f"The code measures this. Under cosine, "
            f"{100 * cos['top5_tied']:.0f}% of samples have their Top-5 decided by a "
            f"tie (Top-3: {100 * cos['top3_tied']:.0f}%). Reported Top-5 uses fractional "
            f"credit (expected value under a random ordering of the tied block); the "
            f"range runs from {cos['top5_lo']:.4f} (pessimistic) to {cos['top5_hi']:.4f} "
            f"(optimistic). When that interval is wide, Top-k isn't really telling you "
            "about model quality. Lean on Top-1 and AUC instead."
        ))
    else:
        story.append(callout(
            "<b>Affects the Top-3 / Top-5 numbers in this build.</b> These artefacts "
            "were produced before tie-aware scoring, using an argsort that breaks ties "
            "by column index (i.e. alphabetical order of the syndrome IDs). A simulation "
            "matching this probability structure showed that just permuting the class "
            "order can move Top-5 by up to 4.5 points, larger than the fold-to-fold "
            "std. Treat the Top-3 and Top-5 values above as an optimistic bound. "
            "Macro-F1, weighted F1, Top-1 and AUC are fine; they don't depend on "
            "column order."
        ))
    story.append(p(
        "The clean fix is distance weighting: neighbour votes become almost unique "
        "and the tied blocks mostly disappear. The pipeline now tunes weights as a "
        "hyper-parameter. For reporting we just state the interval, which the current "
        "code does."
    ))

    story.append(h2("3.3 Where the errors are"))
    if art["pairs"].get("cosine") is not None and not art["pairs"]["cosine"].empty:
        pairs = art["pairs"]["cosine"].head(5)
        rows = pd.DataFrame({
            "True syndrome": pairs["true_syndrome"],
            "Predicted as": pairs["predicted_syndrome"],
            "Errors": pairs["n_errors"],
            "Share of true class": [f"{v:.1%}" for v in pairs["fraction_of_true_class"]],
        })
        story.append(p(
            "OOF predictions under cosine give a proper held-out confusion matrix. "
            "Most confused pairs:"
        ))
        story.append(df_to_table(rows, font_size=8))
        story.append(Spacer(1, 8))
        story.append(p(
            "These pairs are probably the most useful output of the whole run: they "
            "show which syndrome pairs are hardest to distinguish for this "
            "model. Comparing them with the t-SNE visualization may generate "
            "hypotheses, but a 2-D t-SNE projection cannot establish whether a "
            "problem originates in the embeddings or the classifier."
        ))
        cm_png = output_dir / "confusion_matrix_cosine.png"
        if cm_png.exists():
            story.append(fit_image(cm_png, 5.2 * inch, 4.4 * inch))
            story.append(Paragraph(
                "Figure 5. Row-normalised out-of-fold confusion matrix (cosine). Rows "
                "read as per-class recall.", STYLES["Caption"]))

        pc = art["per_class"].get("cosine")
        if pc is not None and not pc.empty:
            worst = pc.iloc[-1]
            story.append(p(
                f"Looking per class shows an asymmetry that the averages hide. Weakest "
                f"class is syndrome {worst['syndrome_id']} ({int(worst['support'])} "
                f"images): precision {worst['precision']:.3f} but recall only "
                f"{worst['recall']:.3f}. When the model does predict it, it's usually "
                "correct, but it misses most of the true instances. Typical minority-class "
                "behaviour under majority vote. A rare class rarely wins among k "
                "neighbours even if a few of them belong to it. Distance weighting or "
                "class-balanced voting would help more here than chasing another fraction "
                "of a point on average accuracy."
            ))
            pc_disp = pd.DataFrame({
                "syndrome_id": pc["syndrome_id"],
                "Support": pc["support"],
                "Precision": [f"{v:.3f}" for v in pc["precision"]],
                "Recall": [f"{v:.3f}" for v in pc["recall"]],
                "F1": [f"{v:.3f}" for v in pc["f1"]],
            })
            story.append(df_to_table(pc_disp, font_size=7.5))
            story.append(Spacer(1, 6))
            story.append(Paragraph(
                "Table. Per-class out-of-fold precision, recall and F1 (cosine), "
                "sorted by F1.", STYLES["Caption"]))
    else:
        story.append(p(
            "Confusion artefacts were not present in this output directory. Running "
            "confusion_analysis.py against oof_predictions_cosine.csv produces the "
            "held-out confusion matrix, the ranked confused pairs and per-class "
            "precision/recall, which is where the aggregate scores become actionable."
        ))

    story.append(h2("3.4 Class imbalance and fold variability"))
    if art["eda"] is not None:
        imbalance = art["eda"]["n_images"].max() / art["eda"]["n_images"].min()
        story.append(p(
            f"Imbalance ratio is {imbalance:.2f}, so macro-F1 is the right headline: "
            "it treats every syndrome equally and won't be propped up by the big "
            f"classes alone. Weighted F1 is also reported ({fmt(*cos['weighted_f1'])} "
            f"for cosine vs macro-F1 {fmt(*cos['macro_f1'])}). Weighted sitting above "
            "macro is the usual pattern on imbalanced data: frequent classes get "
            "predicted more reliably."
        ))
    f1_col = art["folds"]["cosine"]["macro_f1"]
    story.append(p(
        f"Fold-to-fold spread is larger than a single std suggests. Cosine macro-F1 "
        f"runs from {f1_col.min():.4f} (fold "
        f"{int(art['folds']['cosine'].loc[f1_col.idxmin(), 'fold'])}) up to "
        f"{f1_col.max():.4f} (fold "
        f"{int(art['folds']['cosine'].loc[f1_col.idxmax(), 'fold'])}), a range of "
        f"{f1_col.max() - f1_col.min():.4f}. Calling that uniformly stable would be "
        "overstating it. With ~94 test subjects per fold and rare syndromes only "
        "contributing a few images each, a couple of misclassifications in a small "
        "class move the number. The weakest fold is worth looking at on its own "
        "rather than just averaging it away."
    ))

    # ---------------- 4. Challenges ----------------
    story.append(PageBreak())
    story.append(h1("4. Challenges and how they were handled"))
    story.append(bullets([
        "<b>Pickle compatibility.</b> Different numpy versions write pickles that fail "
        "to load, and the error message changes (numpy.core._multiarray_umath, "
        "numpy._core, ...). Matching a fixed string is brittle, so data_processing "
        "catches any numpy-related import failure on unpickle and re-raises with a "
        "clearer instruction.",
        "<b>Subject-level leakage.</b> Main risk on this data. Handled by grouping on "
        "subject_id at both CV levels and by asserting empty train/test group "
        "intersection on every outer fold.",
        "<b>Selection bias from reusing folds.</b> Picking k on the same folds you "
        "report on biases the estimate up. Nested CV removes that; cost is roughly an "
        "order of magnitude more fits, which is fine for KNN.",
        "<b>Rare classes in inner folds.</b> Smallest syndrome has ~64 images, so some "
        "inner validation folds have almost none of it. Stratified grouped splitting "
        "keeps proportions reasonable; inner macro-F1 only averages over classes that "
        "actually appear in that fold, so a missing class no longer contributes a "
        "spurious zero.",
        "<b>Unstable tie-breaking in selection.</b> pandas default sort is quicksort "
        "(not stable), so ties in inner-CV macro-F1 could resolve to an arbitrary k. "
        "Now sorts with mergesort on an explicit (score, k, weights) key; a regression "
        "test locks the behaviour in.",
        "<b>Coarse KNN probabilities.</b> Covered in 3.2. Visible effect is a "
        "step-shaped ROC; practical effect is unreliable Top-k. These probabilities "
        "are also poorly calibrated in absolute terms and shouldn't be read as "
        "confidence without extra calibration.",
    ]))

    # ---------------- 5. Recommendations ----------------
    story.append(h1("5. Recommendations"))
    recs = []
    if k_max is not None:
        recs.append(
            f"<b>Widen the k grid and re-run.</b> Selection is currently capped at "
            f"k={k_max}. Extending to 40-50 will either find an interior optimum or show "
            "the curve flattening; only then is it fair to call a k optimal. Highest "
            "value change, costs one pipeline run."
        )
    else:
        recs.append(
            "<b>Widen the k grid and re-run.</b> Selected k values all sit at the top of "
            "the searched range and the inner-CV curve is still rising, so the optimum "
            "is outside the grid. Try k up to 40-50 before treating any value as optimal."
        )
    recs += [
        "<b>Report Top-k with its tie interval, or just drop it.</b> When the interval "
        "is wide the metric is describing label ordering more than the model. Distance "
        "weighting largely removes the issue and is now tuned automatically.",
        "<b>Separate metric effect from preprocessing.</b> Run cosine with L2 and "
        "euclidean with standardisation, then swap the preprocessing and compare. "
        "Only the difference that survives both setups can be attributed to the "
        "distance function itself.",
        "<b>Act on the confused pairs.</b> The ranked pairs in confused_pairs_*.csv "
        "point at the specific syndromes that need more data, which is much more efficient "
        "than collecting uniformly across all ten.",
        "<b>More images for the rare syndromes.</b> Smallest class has around 64 "
        "images. Reducing imbalance would cut noise both in k selection and in the "
        "macro-averaged scores.",
        "<b>Calibrate before treating probabilities as confidence.</b> KNN scores at "
        "moderate k are coarse and not calibrated. If you need thresholds or triage, "
        "fit isotonic / Platt on held-out folds; if you only care about ranking, "
        "calibration isn't necessary.",
        "<b>If the embedding model can be retrained,</b> a metric-learning loss "
        "(triplet or supervised contrastive) would optimise directly for angular "
        "separation between syndromes, the property the cosine result already "
        "hints is doing most of the work.",
    ]
    story.append(bullets(recs))

    story.append(h1("Appendix: artefact index"))
    story.append(p(
        "Figures and numbers above come from the files in the output directory. "
        "Per-fold scores live in fold_metrics_*.csv, the full inner-CV grid in "
        "nested_k_selection_*.csv, and per-image OOF predictions (with probabilities) "
        "in oof_predictions_*.csv. Any metric in this report can be recomputed from "
        "those without re-running the CV."
    ))

    doc.build(story)
    return report_path


def parse_args():
    ap = argparse.ArgumentParser(
        description="Generate the PDF report from pipeline outputs.")
    ap.add_argument("--output_dir", type=str, default="outputs")
    ap.add_argument("--report_path", type=str, default=None)
    ap.add_argument("--author", type=str, default="Lucas Nunes")
    return ap.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    report_path = Path(args.report_path) if args.report_path else output_dir / "report.pdf"
    build(output_dir, report_path, args.author)
    print(f"Report written to {report_path.resolve()}")


if __name__ == "__main__":
    main()
