from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re
import textwrap
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image as PDFImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ============================================================
# Configuration and paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
QIIME2_DIR = BASE_DIR / "qiime2"

FEATURE_TABLE_FILE = DATA_DIR / "feature-table.tsv"

TAXONOMY_CANDIDATES = [
    DATA_DIR / "taxonomy.tsv",
    DATA_DIR / "exported" / "taxonomy_vsearch" / "taxonomy.tsv",
    DATA_DIR / "taxonomy_vsearch" / "taxonomy.tsv",
]

st.set_page_config(
    page_title="COPD Microbiome Explorer",
    page_icon="🫁",
    layout="wide",
)


# ============================================================
# General data helpers
# ============================================================

@st.cache_data
def load_csv(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception as error:
        st.warning(f"Could not read {filename}: {error}")
        return pd.DataFrame()


def find_taxonomy_file() -> Path | None:
    for path in TAXONOMY_CANDIDATES:
        if path.exists():
            return path

    return None


def short_genus(taxonomy: str) -> str:
    """
    Extract a short genus name from a SILVA taxonomy lineage.
    """

    text = str(taxonomy)

    match = re.search(r"(?:^|;)g__([^;]+)", text)

    if match:
        genus = match.group(1).strip()

        if genus and genus not in {"__", "nan"}:
            return genus

    parts = [
        part.strip()
        for part in text.split(";")
        if part.strip()
    ]

    for part in reversed(parts):
        if "__" in part:
            value = part.split("__", 1)[1].strip()

            if value:
                return value

    return "Unclassified"


@st.cache_data(show_spinner="Loading ASV abundance table...")
def load_feature_table() -> pd.DataFrame:
    """
    Load BIOM-converted feature-table.tsv.

    Rows: ASVs
    Columns: samples
    """

    if not FEATURE_TABLE_FILE.exists():
        return pd.DataFrame()

    try:
        table = pd.read_csv(
            FEATURE_TABLE_FILE,
            sep="\t",
            skiprows=1,
            index_col=0,
            low_memory=False,
        )

        table.index = table.index.astype(str)
        table.index.name = "Feature ID"

        table = table.apply(
            pd.to_numeric,
            errors="coerce",
        ).fillna(0)

        return table

    except Exception:
        # Fallback in case the file has no BIOM comment line.
        table = pd.read_csv(
            FEATURE_TABLE_FILE,
            sep="\t",
            index_col=0,
            low_memory=False,
        )

        table.index = table.index.astype(str)
        table.index.name = "Feature ID"

        return table.apply(
            pd.to_numeric,
            errors="coerce",
        ).fillna(0)


@st.cache_data(show_spinner="Loading taxonomy assignments...")
def load_taxonomy_table() -> pd.DataFrame:
    path = find_taxonomy_file()

    if path is None:
        return pd.DataFrame()

    taxonomy = pd.read_csv(
        path,
        sep="\t",
        low_memory=False,
    )

    feature_column = None
    taxon_column = None
    confidence_column = None

    for column in taxonomy.columns:
        lower = str(column).lower().strip()

        if feature_column is None and (
            "feature" in lower
            or lower in {"id", "feature id"}
        ):
            feature_column = column

        if taxon_column is None and (
            lower == "taxon"
            or "taxonomy" in lower
        ):
            taxon_column = column

        if confidence_column is None and (
            "confidence" in lower
            or "consensus" in lower
        ):
            confidence_column = column

    if feature_column is None:
        feature_column = taxonomy.columns[0]

    if taxon_column is None and len(taxonomy.columns) >= 2:
        taxon_column = taxonomy.columns[1]

    selected_columns = [
        feature_column,
        taxon_column,
    ]

    if (
        confidence_column is not None
        and confidence_column not in selected_columns
    ):
        selected_columns.append(confidence_column)

    taxonomy = taxonomy[selected_columns].copy()

    rename_mapping = {
        feature_column: "Feature ID",
        taxon_column: "Taxonomy",
    }

    if confidence_column is not None:
        rename_mapping[confidence_column] = "Confidence / Consensus"

    taxonomy = taxonomy.rename(columns=rename_mapping)

    taxonomy["Feature ID"] = taxonomy["Feature ID"].astype(str)
    taxonomy["Genus"] = taxonomy["Taxonomy"].map(short_genus)

    return taxonomy


@st.cache_data(show_spinner="Preparing ASV summary...")
def prepare_asv_summary() -> pd.DataFrame:
    feature_table = load_feature_table()

    if feature_table.empty:
        return pd.DataFrame()

    summary = pd.DataFrame(
        {
            "Feature ID": feature_table.index,
            "Total abundance": feature_table.sum(axis=1).values,
            "Samples observed": (
                feature_table.gt(0).sum(axis=1).values
            ),
            "Mean abundance": feature_table.mean(axis=1).values,
            "Maximum abundance": feature_table.max(axis=1).values,
        }
    )

    taxonomy = load_taxonomy_table()

    if not taxonomy.empty:
        summary = summary.merge(
            taxonomy,
            on="Feature ID",
            how="left",
        )
    else:
        summary["Taxonomy"] = "Taxonomy unavailable"
        summary["Genus"] = "Unclassified"

    summary["Taxonomy"] = summary["Taxonomy"].fillna(
        "Unassigned"
    )

    summary["Genus"] = summary["Genus"].fillna(
        "Unclassified"
    )

    return summary.sort_values(
        "Total abundance",
        ascending=False,
    )


# ============================================================
# Figure PDF/download helpers
# ============================================================

@st.cache_data
def image_to_pdf_bytes(
    image_path_string: str,
    title: str,
) -> bytes:
    image_path = Path(image_path_string)
    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=1.2 * cm,
        leftMargin=1.2 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
    )

    styles = getSampleStyleSheet()

    story = [
        Paragraph(title, styles["Title"]),
        Spacer(1, 0.5 * cm),
    ]

    with Image.open(image_path) as image:
        width_pixels, height_pixels = image.size

    max_width = 25.5 * cm
    max_height = 16.0 * cm

    scale = min(
        max_width / width_pixels,
        max_height / height_pixels,
    )

    story.append(
        PDFImage(
            str(image_path),
            width=width_pixels * scale,
            height=height_pixels * scale,
        )
    )

    document.build(story)
    buffer.seek(0)

    return buffer.getvalue()


def show_image(
    filename: str,
    caption: str,
    download_name: str | None = None,
    key_prefix: str | None = None,
) -> None:
    path = FIGURES_DIR / filename

    if not path.exists():
        st.warning(f"Figure unavailable: {filename}")
        return

    st.image(
        str(path),
        caption=caption,
        use_container_width=True,
    )

    safe_name = download_name or path.stem
    unique_key = key_prefix or path.stem

    png_bytes = path.read_bytes()

    try:
        pdf_bytes = image_to_pdf_bytes(
            str(path),
            caption,
        )
    except Exception as error:
        pdf_bytes = None
        st.caption(f"PDF conversion unavailable: {error}")

    png_column, pdf_column = st.columns(2)

    with png_column:
        st.download_button(
            label="⬇️ Download PNG",
            data=png_bytes,
            file_name=f"{safe_name}.png",
            mime="image/png",
            key=f"png_{unique_key}",
            use_container_width=True,
        )

    with pdf_column:
        if pdf_bytes is not None:
            st.download_button(
                label="📄 Download PDF",
                data=pdf_bytes,
                file_name=f"{safe_name}.pdf",
                mime="application/pdf",
                key=f"pdf_{unique_key}",
                use_container_width=True,
            )


# ============================================================
# Table PDF/download helpers
# ============================================================

@st.cache_data
def dataframe_to_pdf_bytes(
    dataframe_csv: str,
    title: str,
    max_rows: int = 100,
) -> bytes:
    dataframe = pd.read_csv(
        BytesIO(dataframe_csv.encode("utf-8"))
    )

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=0.7 * cm,
        leftMargin=0.7 * cm,
        topMargin=0.8 * cm,
        bottomMargin=0.8 * cm,
    )

    styles = getSampleStyleSheet()

    story = [
        Paragraph(title, styles["Title"]),
        Spacer(1, 0.4 * cm),
    ]

    display_dataframe = dataframe.head(max_rows).copy()

    for column in display_dataframe.columns:
        display_dataframe[column] = display_dataframe[column].map(
            lambda value: "\n".join(
                textwrap.wrap(
                    str(value),
                    width=28,
                )
            )
            if len(str(value)) > 28
            else str(value)
        )

    table_data = [
        [
            Paragraph(
                str(column),
                styles["BodyText"],
            )
            for column in display_dataframe.columns
        ]
    ]

    for row in display_dataframe.itertuples(
        index=False,
        name=None,
    ):
        table_data.append(
            [
                Paragraph(
                    str(value),
                    styles["BodyText"],
                )
                for value in row
            ]
        )

    number_of_columns = max(
        len(display_dataframe.columns),
        1,
    )

    column_width = (
        27.5 * cm
        / number_of_columns
    )

    pdf_table = Table(
        table_data,
        colWidths=[
            column_width
            for _ in display_dataframe.columns
        ],
        repeatRows=1,
    )

    pdf_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#315f7d"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    6.5,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.35,
                    colors.HexColor("#b7c9d4"),
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        colors.HexColor("#edf5f8"),
                    ],
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    3,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    3,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    3,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    3,
                ),
            ]
        )
    )

    story.append(pdf_table)

    if len(dataframe) > max_rows:
        story.append(Spacer(1, 0.4 * cm))
        story.append(
            Paragraph(
                f"PDF contains the first {max_rows} "
                f"of {len(dataframe)} rows. "
                "Download CSV for the complete table.",
                styles["BodyText"],
            )
        )

    document.build(story)
    buffer.seek(0)

    return buffer.getvalue()


def show_table(
    dataframe: pd.DataFrame,
    title: str,
    filename_prefix: str,
    hide_index: bool = True,
    max_display_rows: int | None = None,
) -> None:
    if dataframe.empty:
        st.info(f"{title} is not available.")
        return

    st.subheader(title)

    display_dataframe = (
        dataframe.head(max_display_rows)
        if max_display_rows is not None
        else dataframe
    )

    st.dataframe(
        display_dataframe,
        use_container_width=True,
        hide_index=hide_index,
    )

    csv_text = dataframe.to_csv(
        index=not hide_index,
    )

    try:
        pdf_bytes = dataframe_to_pdf_bytes(
            dataframe.to_csv(index=False),
            title,
        )
    except Exception as error:
        pdf_bytes = None
        st.caption(
            f"PDF table conversion unavailable: {error}"
        )

    csv_column, pdf_column = st.columns(2)

    with csv_column:
        st.download_button(
            label="⬇️ Download CSV",
            data=csv_text.encode("utf-8"),
            file_name=f"{filename_prefix}.csv",
            mime="text/csv",
            key=f"csv_{filename_prefix}",
            use_container_width=True,
        )

    with pdf_column:
        if pdf_bytes is not None:
            st.download_button(
                label="📄 Download PDF",
                data=pdf_bytes,
                file_name=f"{filename_prefix}.pdf",
                mime="application/pdf",
                key=f"table_pdf_{filename_prefix}",
                use_container_width=True,
            )


# ============================================================
# Complete results ZIP
# ============================================================

@st.cache_data
def create_results_zip() -> bytes:
    buffer = BytesIO()

    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:

        if DATA_DIR.exists():
            for path in DATA_DIR.rglob("*"):
                if path.is_file():
                    archive.write(
                        path,
                        arcname=Path("data") / path.relative_to(DATA_DIR),
                    )

        if FIGURES_DIR.exists():
            for path in FIGURES_DIR.rglob("*"):
                if path.is_file():
                    archive.write(
                        path,
                        arcname=Path("figures") / path.relative_to(
                            FIGURES_DIR
                        ),
                    )

    buffer.seek(0)
    return buffer.getvalue()


def metric_card(
    column,
    label: str,
    value: str,
    help_text: str | None = None,
) -> None:
    column.metric(
        label=label,
        value=value,
        help=help_text,
    )


# ============================================================
# Styling
# ============================================================



st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       Main application background
    -------------------------------------------------------- */

    .stApp {
    background:
        radial-gradient(
            circle at top left,
            rgba(62, 174, 213, 0.12),
            transparent 34%
        ),
        radial-gradient(
            circle at top right,
            rgba(108, 92, 231, 0.12),
            transparent 34%
        ),
        linear-gradient(
            135deg,
            #0e1117 0%,
            #131722 50%,
            #101820 100%
        );
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 3rem;
        max-width: 1580px;
    }

    /* --------------------------------------------------------
       General typography
    -------------------------------------------------------- */

    h1 {
        margin-bottom: 0.25rem;
        font-weight: 850;
        color: #123c59;
        letter-spacing: -0.5px;
    }

    h2 {
        color: #1d4f70;
        font-weight: 800;
        margin-top: 1.2rem;
    }

    h3 {
        color: #2a5f7c;
        font-weight: 750;
    }

    /* --------------------------------------------------------
       Top navigation tabs
    -------------------------------------------------------- */

    div[data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(255, 255, 255, 0.92);
        padding: 12px;
        border-radius: 18px;
        box-shadow:
            0 8px 28px rgba(31, 74, 102, 0.12);
        margin-top: 18px;
        margin-bottom: 26px;
        flex-wrap: wrap;
        border: 1px solid rgba(91, 139, 166, 0.16);
    }

    button[data-baseweb="tab"] {
        border-radius: 13px;
        padding: 12px 17px;
        min-height: 52px;
        border: 1px solid rgba(73, 119, 148, 0.22);
        transition:
            transform 0.18s ease,
            box-shadow 0.18s ease,
            background-color 0.18s ease;
    }

    button[data-baseweb="tab"] p {
        font-size: 16px !important;
        font-weight: 850 !important;
        color: #173f5a;
        margin: 0;
        white-space: nowrap;
    }

    button[data-baseweb="tab"]:hover {
        transform: translateY(-2px);
        box-shadow:
            0 6px 14px rgba(44, 91, 122, 0.18);
        filter: brightness(0.98);
    }

    /* Different soft colour for each navigation button */

    button[data-baseweb="tab"]:nth-child(1) {
        background: #e8f3ff;
    }

    button[data-baseweb="tab"]:nth-child(2) {
        background: #eafaf2;
    }

    button[data-baseweb="tab"]:nth-child(3) {
        background: #fff4df;
    }

    button[data-baseweb="tab"]:nth-child(4) {
        background: #f1ecff;
    }

    button[data-baseweb="tab"]:nth-child(5) {
        background: #e9f8fa;
    }

    button[data-baseweb="tab"]:nth-child(6) {
        background: #fff0ee;
    }

    button[data-baseweb="tab"]:nth-child(7) {
        background: #f5ebff;
    }

    button[data-baseweb="tab"]:nth-child(8) {
        background: #fff6dc;
    }

    button[data-baseweb="tab"]:nth-child(9) {
        background: #e9f8ee;
    }

    button[data-baseweb="tab"]:nth-child(10) {
        background: #f1f3f5;
    }

    /* Selected navigation tab */

    button[data-baseweb="tab"][aria-selected="true"] {
        background:
            linear-gradient(
                135deg,
                #1479a8,
                #665bc7
            ) !important;
        border-color: transparent !important;
        transform: translateY(-2px);
        box-shadow:
            0 8px 18px rgba(71, 87, 171, 0.30);
    }

    button[data-baseweb="tab"][aria-selected="true"] p {
        color: white !important;
        font-weight: 900 !important;
    }

    div[data-baseweb="tab-highlight"] {
        display: none;
    }

    /* --------------------------------------------------------
       Hero section
    -------------------------------------------------------- */

    .hero-card {
        background:
            linear-gradient(
                120deg,
                rgba(17, 112, 156, 0.98),
                rgba(89, 80, 184, 0.94)
            );
        border-radius: 24px;
        padding: 34px 38px;
        color: white;
        box-shadow:
            0 14px 36px rgba(44, 74, 130, 0.22);
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }

    .hero-card::after {
        content: "";
        position: absolute;
        width: 280px;
        height: 280px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.08);
        right: -80px;
        top: -110px;
    }

    .hero-title {
        font-size: 2.25rem;
        line-height: 1.15;
        font-weight: 900;
        margin-bottom: 10px;
        position: relative;
        z-index: 2;
    }

    .hero-subtitle {
        max-width: 980px;
        font-size: 1.08rem;
        line-height: 1.65;
        opacity: 0.96;
        position: relative;
        z-index: 2;
    }

    .hero-tag {
        display: inline-block;
        background: rgba(255, 255, 255, 0.17);
        border: 1px solid rgba(255, 255, 255, 0.25);
        border-radius: 999px;
        padding: 6px 12px;
        margin-right: 7px;
        margin-top: 14px;
        font-size: 0.88rem;
        font-weight: 750;
        position: relative;
        z-index: 2;
    }

    /* --------------------------------------------------------
       Information cards
    -------------------------------------------------------- */

    .info-card {
        background: rgba(255, 255, 255, 0.91);
        border: 1px solid rgba(78, 127, 155, 0.16);
        border-radius: 18px;
        padding: 20px 22px;
        min-height: 180px;
        box-shadow:
            0 7px 20px rgba(37, 76, 102, 0.08);
        margin-bottom: 12px;
    }

    .info-card-title {
        font-size: 1.08rem;
        font-weight: 850;
        color: #174c6b;
        margin-bottom: 8px;
    }

    .info-card-text {
        font-size: 0.97rem;
        color: #486778;
        line-height: 1.55;
    }

    /* --------------------------------------------------------
       Workflow
    -------------------------------------------------------- */

    .workflow-wrapper {
        background: rgba(255, 255, 255, 0.88);
        border-radius: 18px;
        padding: 20px;
        border: 1px solid rgba(70, 117, 145, 0.15);
        box-shadow:
            0 6px 18px rgba(35, 74, 100, 0.07);
        margin-bottom: 20px;
    }

    .workflow-step {
        display: inline-block;
        background:
            linear-gradient(
                135deg,
                #e8f5fb,
                #eeeafd
            );
        border: 1px solid rgba(85, 121, 164, 0.18);
        border-radius: 999px;
        padding: 9px 14px;
        margin: 5px 4px;
        font-size: 0.91rem;
        font-weight: 760;
        color: #214e69;
    }

    .workflow-arrow {
        color: #6a75a7;
        font-weight: 900;
        padding: 0 2px;
    }

    /* --------------------------------------------------------
       Metrics
    -------------------------------------------------------- */

    div[data-testid="stMetric"] {
        border: 1px solid rgba(73, 126, 156, 0.18);
        border-radius: 17px;
        padding: 17px;
        background: rgba(255, 255, 255, 0.91);
        box-shadow:
            0 7px 20px rgba(32, 72, 98, 0.08);
        transition:
            transform 0.18s ease,
            box-shadow 0.18s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow:
            0 10px 24px rgba(32, 72, 98, 0.13);
    }

    div[data-testid="stMetricLabel"] {
        font-weight: 800;
        color: #4a6d82;
    }

    div[data-testid="stMetricValue"] {
        font-weight: 900;
        color: #153f5a;
    }

    /* --------------------------------------------------------
       Tables, expanders and images
    -------------------------------------------------------- */

    div[data-testid="stDataFrame"],
    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.88);
        border-radius: 15px;
        padding: 5px;
        box-shadow:
            0 4px 14px rgba(31, 70, 96, 0.07);
    }

    div[data-testid="stImage"] {
        background: rgba(255, 255, 255, 0.91);
        padding: 11px;
        border-radius: 16px;
        box-shadow:
            0 6px 18px rgba(31, 70, 96, 0.08);
    }

    /* --------------------------------------------------------
       Download buttons
    -------------------------------------------------------- */

    .stDownloadButton > button {
        background:
            linear-gradient(
                135deg,
                #dff3e9,
                #e5f4f8
            );
        color: #155841;
        border: 1px solid #9dcfba;
        border-radius: 11px;
        font-weight: 800;
        transition:
            transform 0.18s ease,
            box-shadow 0.18s ease;
    }

    .stDownloadButton > button:hover {
        background:
            linear-gradient(
                135deg,
                #c8ebd9,
                #d4edf4
            );
        color: #103f31;
        border-color: #67b995;
        transform: translateY(-2px);
        box-shadow:
            0 5px 12px rgba(47, 128, 93, 0.16);
    }

    /* --------------------------------------------------------
       Select boxes and inputs
    -------------------------------------------------------- */

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        border-radius: 11px;
        background-color: rgba(255, 255, 255, 0.95);
    }

    .small-note {
        font-size: 0.97rem;
        color: #557486;
        opacity: 0.96;
        line-height: 1.55;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Header
# ============================================================


st.title("🫁 COPD Microbiome Explorer")

st.markdown(
    """
    **Interactive end-to-end 16S rRNA microbiome analysis**

    This application integrates sequencing quality control, QIIME2,
    DADA2, taxonomy, microbial diversity, machine learning,
    SHAP and LIME explainability.
    """
)

st.caption(
    "16S Microbiome · QIIME2 · DADA2 · Machine Learning · "
    "Explainable AI · Interactive Analytics"
)

st.warning(
    "Important interpretation note: Healthy and COPD samples originated "
    "from different BioProjects. Observed discrimination may reflect both "
    "biological differences and study-specific batch effects."
)

# ============================================================
# Reordered navigation tabs
# ============================================================

tabs = st.tabs(
    [
        "🏠 Overview",
        "🧪 Quality Control",
        "🦠 Taxonomy",
        "🧬 Diversity",
        "🔬 ASV Explorer",
        "🤖 Models",
        "🔍 SHAP",
        "💡 LIME",
        "🎯 Serratia",
        "⚠️ Limitations",
    ]
)


# ============================================================
# Tab 1 — Overview
# ============================================================

with tabs[0]:
    st.header("Project overview")

    st.write(
        "This dashboard presents an exploratory comparison of sputum "
        "microbiome profiles from COPD and healthy cohorts. It combines "
        "bioinformatics processing, statistical analysis, microbiome "
        "visualisation, machine learning and explainable AI."
    )

    st.divider()

    # --------------------------------------------------------
    # Main project metrics
    # --------------------------------------------------------

    col1, col2, col3, col4, col5 = st.columns(5)

    metric_card(
        col1,
        "Total samples",
        "839",
        "All COPD and Healthy samples processed in QIIME2.",
    )

    metric_card(
        col2,
        "COPD samples",
        "715",
        "Samples belonging to the COPD cohort.",
    )

    metric_card(
        col3,
        "Healthy samples",
        "124",
        "Samples belonging to the Healthy cohort.",
    )

    metric_card(
        col4,
        "Detected ASVs",
        "5,035",
        "Exact sequence variants generated by DADA2.",
    )

    metric_card(
        col5,
        "Genus features",
        "375",
        "Genus-level features used for downstream analysis.",
    )

    st.divider()

    # --------------------------------------------------------
    # Project information
    # --------------------------------------------------------

    st.subheader("Scientific objective")

    st.info(
        "To identify microbial differences between COPD and healthy "
        "sputum samples and evaluate whether genus-level abundance "
        "profiles can distinguish the two cohorts."
    )

    information_left, information_middle, information_right = st.columns(3)

    with information_left:
        st.markdown("### 🧪 Bioinformatics")

        st.write(
            "Paired-end reads were processed using QIIME2 and DADA2 "
            "to generate an ASV abundance table."
        )

    with information_middle:
        st.markdown("### 🦠 Microbiome analysis")

        st.write(
            "Taxonomy was assigned using the SILVA database, followed by "
            "alpha diversity, beta diversity and relative-abundance analysis."
        )

    with information_right:
        st.markdown("### 🤖 Machine learning")

        st.write(
            "Logistic Regression, Random Forest, Gradient Boosting and "
            "XGBoost were compared. SHAP and LIME were used to explain "
            "model predictions."
        )

    st.divider()

    # --------------------------------------------------------
    # Simple workflow
    # --------------------------------------------------------

    st.subheader("End-to-end analysis workflow")

    st.markdown(
        """
        **Raw FASTQ → Quality control → QIIME2 import → DADA2 denoising  
        → ASV table → SILVA taxonomy → Alpha and beta diversity  
        → Relative abundance → Machine learning → SHAP and LIME  
        → Interactive Streamlit dashboard**
        """
    )

    st.caption(
        "The workflow starts with raw sequencing reads and ends with "
        "interactive biological and machine-learning interpretation."
    )

    st.divider()

    # --------------------------------------------------------
    # Analysis design
    # --------------------------------------------------------

    st.subheader("Analysis design")

    analysis_left, analysis_right = st.columns(2)

    with analysis_left:
        st.info(
            "### Analysis 1 — Balanced cohort\n\n"
            "**124 Healthy + 124 COPD samples**\n\n"
            "This dataset was used to compare machine-learning models "
            "with equal sample numbers in both disease groups."
        )

    with analysis_right:
        st.info(
            "### Analysis 2 — Full cohort\n\n"
            "**124 Healthy + 715 COPD samples**\n\n"
            "This analysis will use the complete cohort with class weighting "
            "and imbalance-aware evaluation metrics."
        )

    st.divider()

    # --------------------------------------------------------
    # Main exploratory observations
    # --------------------------------------------------------

    st.subheader("Main exploratory observations")

    observation_left, observation_middle, observation_right = st.columns(3)

    with observation_left:
        st.success(
            "**Microbial composition**\n\n"
            "COPD and Healthy samples showed different genus-level "
            "abundance patterns."
        )

    with observation_middle:
        st.success(
            "**Community diversity**\n\n"
            "The two cohorts showed strong separation in beta-diversity "
            "analysis."
        )

    with observation_right:
        st.success(
            "**Explainable machine learning**\n\n"
            "Serratia, Acinetobacter, Leptotrichia and related genera "
            "contributed strongly to model discrimination."
        )

    st.warning(
        "These genera should currently be described as cohort-discriminating "
        "features, not as clinically validated COPD biomarkers."
    )

    st.divider()

    # --------------------------------------------------------
    # Dashboard navigation information
    # --------------------------------------------------------

    st.subheader("Explore the dashboard")

    st.markdown(
        """
        Use the coloured buttons above to explore:

        **Quality Control** — sequencing depth and DADA2 retention  
        **Taxonomy** — phylum- and genus-level composition  
        **Diversity** — alpha diversity, PCoA and PERMANOVA  
        **ASV Explorer** — searchable ASV abundance and taxonomy  
        **Models** — performance of machine-learning classifiers  
        **SHAP** — global and individual XGBoost explanations  
        **LIME** — local sample-level explanations  
        **Serratia** — abundance and prevalence analysis  
        **Limitations** — study design and interpretation cautions
        """
    )

    st.subheader("Download complete results")

    st.download_button(
        label="📦 Download all figures and result tables",
        data=create_results_zip(),
        file_name="COPD_microbiome_dashboard_results.zip",
        mime="application/zip",
        key="download_complete_results",
        use_container_width=True,
    )


# ============================================================
# Tab 2 — Quality Control
# ============================================================

with tabs[1]:
    st.header("Sequencing quality control and DADA2 processing")

    qc_summary = load_csv(
        "qc_summary_metrics.csv"
    )

    if not qc_summary.empty:
        summary_dictionary = dict(
            zip(
                qc_summary["Metric"],
                qc_summary["Value"],
            )
        )

        qc_col1, qc_col2, qc_col3, qc_col4 = st.columns(4)

        metric_card(
            qc_col1,
            "Processed samples",
            f"{int(float(summary_dictionary.get('Samples', 839))):,}",
        )

        metric_card(
            qc_col2,
            "Median input reads",
            f"{float(summary_dictionary.get('Median input reads', 0)):,.0f}",
        )

        metric_card(
            qc_col3,
            "Median merged reads",
            f"{float(summary_dictionary.get('Median merged reads', 0)):,.0f}",
        )

        metric_card(
            qc_col4,
            "Median non-chimeric",
            f"{float(summary_dictionary.get('Median non-chimeric reads', 0)):,.0f}",
        )

    st.subheader("DADA2 denoising and read retention")

    qc_left, qc_right = st.columns(2)

    with qc_left:
        show_image(
            "00_qc_dada2_read_retention.png",
            "Median read counts retained during DADA2 processing",
            "dada2_read_retention",
            "qc_dada2_counts",
        )

    with qc_right:
        show_image(
            "00_qc_dada2_retention_percentage.png",
            "Percentage of input reads retained after filtering, merging and chimera removal",
            "dada2_retention_percentage",
            "qc_dada2_percent",
        )

    st.subheader("Feature-table sequencing depth")

    depth_left, depth_right = st.columns(2)

    with depth_left:
        show_image(
            "00_qc_sequencing_depth_histogram.png",
            "Distribution of sequencing depth across samples",
            "sequencing_depth_histogram",
            "qc_depth_histogram",
        )

    with depth_right:
        show_image(
            "00_qc_feature_table_summary.png",
            "Feature-table summary after DADA2",
            "feature_table_summary",
            "qc_table_summary",
        )

    if not qc_summary.empty:
        show_table(
            qc_summary,
            title="Quality-control summary metrics",
            filename_prefix="qc_summary_metrics",
        )

    denoising_stats = load_csv(
        "dada2_denoising_statistics.csv"
    )

    if not denoising_stats.empty:
        with st.expander(
            "View per-sample DADA2 denoising statistics"
        ):
            show_table(
                denoising_stats,
                title="DADA2 denoising statistics",
                filename_prefix="dada2_denoising_statistics",
                max_display_rows=100,
            )

    st.info(
        "DADA2 performed quality filtering, error correction, paired-read "
        "merging and chimera removal before generating the ASV table."
    )


# ============================================================
# Tab 3 — Taxonomy
# ============================================================

with tabs[2]:
    st.header("Taxonomic composition")

    tax_col1, tax_col2, tax_col3 = st.columns(3)

    metric_card(tax_col1, "Reference database", "SILVA 138")
    metric_card(tax_col2, "Assignment method", "VSEARCH consensus")
    metric_card(tax_col3, "Genus-level features", "375")

    st.subheader("Phylum-level relative abundance")

    show_image(
        "04_taxonomy_phylum_stacked_bar.png",
        "Mean phylum-level relative abundance",
        "taxonomy_phylum_stacked_bar",
        "taxonomy_phylum",
    )

    st.subheader("Genus-level relative abundance")

    genus_left, genus_right = st.columns(2)

    with genus_left:
        show_image(
            "05_taxonomy_top15_genera_stacked_bar.png",
            "Top 15 genera — stacked relative abundance",
            "taxonomy_top15_genera",
            "taxonomy_genus_stacked",
        )

    with genus_right:
        show_image(
            "06_taxonomy_group_mean_genera.png",
            "Top genera by mean relative abundance",
            "taxonomy_group_mean_genera",
            "taxonomy_group_mean",
        )

    st.subheader("Top-genus abundance heatmap")

    show_image(
        "07_taxonomy_top20_genera_heatmap.png",
        "Top 20 genera across representative Healthy and COPD samples",
        "taxonomy_top20_heatmap",
        "taxonomy_heatmap",
    )

    taxonomy_table = load_taxonomy_table()

    if not taxonomy_table.empty:
        st.subheader("Taxonomy assignment table")

        taxonomy_search = st.text_input(
            "Search taxonomy by ASV ID, genus or lineage",
            key="taxonomy_search",
            placeholder="Example: Serratia or 0277c58f",
        )

        filtered_taxonomy = taxonomy_table.copy()

        if taxonomy_search.strip():
            search_text = taxonomy_search.strip()

            mask = (
                filtered_taxonomy
                .astype(str)
                .apply(
                    lambda column: column.str.contains(
                        search_text,
                        case=False,
                        na=False,
                        regex=False,
                    )
                )
                .any(axis=1)
            )

            filtered_taxonomy = filtered_taxonomy[mask]

        show_table(
            filtered_taxonomy,
            title=f"Taxonomy results ({len(filtered_taxonomy):,} matches)",
            filename_prefix="taxonomy_assignments_filtered",
            max_display_rows=200,
        )
    else:
        st.warning(
            "taxonomy.tsv was not found. Copy it into data/ or "
            "data/exported/taxonomy_vsearch/."
        )


# ============================================================
# Tab 4 — Diversity
# ============================================================

with tabs[3]:
    st.header("Microbiome diversity")

    st.subheader("Alpha diversity")

    alpha_left, alpha_right = st.columns(2)

    with alpha_left:
        show_image(
            "01_alpha_shannon_boxplot.png",
            "Shannon diversity by disease group",
            "alpha_shannon_diversity",
            "alpha_shannon",
        )

    with alpha_right:
        show_image(
            "02_alpha_observed_features_boxplot.png",
            "Observed-feature richness by disease group",
            "alpha_observed_features",
            "alpha_observed",
        )

    st.markdown(
        """
        - **Shannon diversity** combines richness and evenness.
        - **Observed features** represent the number of detected ASVs.
        """
    )

    st.divider()

    st.subheader("Beta diversity")

    show_image(
        "03_beta_bray_curtis_pcoa.png",
        "Bray–Curtis PCoA of Healthy and COPD samples",
        "beta_bray_curtis_pcoa",
        "beta_pcoa",
    )

    beta_col1, beta_col2, beta_col3 = st.columns(3)

    metric_card(beta_col1, "PERMANOVA p-value", "0.001")
    metric_card(beta_col2, "Pseudo-F", "906.33")
    metric_card(beta_col3, "Permutations", "999")

    st.info(
        "Bray–Curtis PCoA visualizes abundance-based differences in "
        "microbial community composition. PERMANOVA statistically tested "
        "whether the two groups differed in multivariate distance space."
    )


# ============================================================
# Tab 5 — ASV Explorer
# ============================================================

with tabs[4]:
    st.header("Interactive ASV abundance explorer")

    feature_table = load_feature_table()
    asv_summary = prepare_asv_summary()
    taxonomy_table = load_taxonomy_table()

    if feature_table.empty:
        st.error(
            "feature-table.tsv was not found inside the dashboard data folder."
        )

    else:
        asv_count = feature_table.shape[0]
        sample_count = feature_table.shape[1]
        total_reads = feature_table.to_numpy().sum()
        nonzero_values = int(
            np.count_nonzero(
                feature_table.to_numpy()
            )
        )

        asv_col1, asv_col2, asv_col3, asv_col4 = st.columns(4)

        metric_card(asv_col1, "ASVs", f"{asv_count:,}")
        metric_card(asv_col2, "Samples", f"{sample_count:,}")
        metric_card(asv_col3, "Total ASV reads", f"{total_reads:,.0f}")
        metric_card(asv_col4, "Non-zero entries", f"{nonzero_values:,}")

        st.caption(
            "The feature table contains raw ASV counts. Rows represent "
            "ASVs, columns represent samples and values represent read counts."
        )

        st.subheader("Search ASVs and taxa")

        search_left, search_right = st.columns([2, 1])

        with search_left:
            asv_search = st.text_input(
                "Search by Feature ID, genus or taxonomy",
                key="asv_search",
                placeholder="Example: Serratia or 0277c58fe865",
            )

        with search_right:
            maximum_rows = st.selectbox(
                "Rows displayed",
                [25, 50, 100, 250, 500],
                index=2,
                key="asv_rows",
            )

        filtered_summary = asv_summary.copy()

        if asv_search.strip():
            search_text = asv_search.strip()

            mask = (
                filtered_summary
                .astype(str)
                .apply(
                    lambda column: column.str.contains(
                        search_text,
                        case=False,
                        na=False,
                        regex=False,
                    )
                )
                .any(axis=1)
            )

            filtered_summary = filtered_summary[mask]

        st.write(
            f"**Matching ASVs:** {len(filtered_summary):,}"
        )

        display_columns = [
            column
            for column in [
                "Feature ID",
                "Genus",
                "Taxonomy",
                "Confidence / Consensus",
                "Total abundance",
                "Samples observed",
                "Mean abundance",
                "Maximum abundance",
            ]
            if column in filtered_summary.columns
        ]

        show_table(
            filtered_summary[display_columns],
            title="ASV abundance and taxonomy results",
            filename_prefix="asv_explorer_results",
            max_display_rows=maximum_rows,
        )

        st.divider()

        st.subheader("Top abundant ASVs")

        top_n_asvs = st.slider(
            "Number of top ASVs",
            min_value=5,
            max_value=50,
            value=15,
            step=5,
        )

        top_asvs = asv_summary.head(top_n_asvs).copy()

        if "Genus" in top_asvs.columns:
            top_asvs["Display label"] = (
                top_asvs["Genus"].astype(str)
                + " | "
                + top_asvs["Feature ID"].str[:10]
            )
        else:
            top_asvs["Display label"] = (
                top_asvs["Feature ID"].str[:12]
            )

        chart_data = (
            top_asvs[
                ["Display label", "Total abundance"]
            ]
            .set_index("Display label")
            .sort_values(
                "Total abundance",
                ascending=True,
            )
        )

        st.bar_chart(
            chart_data,
            horizontal=True,
            use_container_width=True,
        )

        show_table(
            top_asvs[display_columns],
            title=f"Top {top_n_asvs} ASVs by total abundance",
            filename_prefix=f"top_{top_n_asvs}_abundant_asvs",
        )

        st.divider()

        st.subheader("Inspect one ASV across samples")

        available_asvs = filtered_summary["Feature ID"].tolist()

        if available_asvs:
            selected_asv = st.selectbox(
                "Select an ASV",
                available_asvs[:1000],
                key="selected_asv",
            )

            selected_counts = (
                feature_table.loc[selected_asv]
                .sort_values(ascending=False)
            )

            asv_sample_table = pd.DataFrame(
                {
                    "Sample ID": selected_counts.index,
                    "ASV abundance": selected_counts.values,
                }
            )

            positive_samples = asv_sample_table[
                asv_sample_table["ASV abundance"] > 0
            ]

            selected_taxonomy = asv_summary[
                asv_summary["Feature ID"] == selected_asv
            ]

            if not selected_taxonomy.empty:
                selected_row = selected_taxonomy.iloc[0]

                detail_col1, detail_col2, detail_col3 = st.columns(3)

                metric_card(
                    detail_col1,
                    "Genus",
                    str(selected_row.get("Genus", "Unclassified")),
                )

                metric_card(
                    detail_col2,
                    "Samples observed",
                    f"{int(selected_row['Samples observed']):,}",
                )

                metric_card(
                    detail_col3,
                    "Total abundance",
                    f"{selected_row['Total abundance']:,.0f}",
                )

                st.write(
                    "**Taxonomy:** "
                    f"{selected_row.get('Taxonomy', 'Unavailable')}"
                )

            top_sample_counts = positive_samples.head(30)

            if not top_sample_counts.empty:
                st.bar_chart(
                    top_sample_counts.set_index("Sample ID"),
                    use_container_width=True,
                )

                show_table(
                    positive_samples,
                    title=f"Sample abundances for ASV {selected_asv}",
                    filename_prefix=(
                        f"asv_{selected_asv[:12]}_sample_abundance"
                    ),
                    max_display_rows=200,
                )
            else:
                st.info(
                    "This ASV has no positive abundance values."
                )

        st.info(
            "ASV IDs are exact sequence variants produced by DADA2. "
            "Taxonomic annotations were assigned using consensus VSEARCH "
            "against SILVA 138."
        )


# ============================================================
# Tab 6 — Models
# ============================================================

with tabs[5]:
    st.header("Machine-learning models")

    st.info(
        "Models were trained on 248 balanced samples "
        "(124 Healthy and 124 COPD). Performance was evaluated on an "
        "independent 20% hold-out test set containing 50 unseen samples "
        "(25 Healthy and 25 COPD)."
    )

    model_results = pd.DataFrame(
        {
            "Model": [
                "Random Forest",
                "Logistic Regression",
                "Gradient Boosting",
                "XGBoost",
            ],
            "Accuracy": [
                1.00,
                0.96,
                1.00,
                1.00,
            ],
            "Balanced accuracy": [
                1.00,
                0.96,
                1.00,
                1.00,
            ],
            "Test ROC-AUC": [
                1.00,
                1.00,
                1.00,
                1.00,
            ],
            "CV ROC-AUC": [
                1.00,
                1.00,
                0.9958,
                1.00,
            ],
        }
    )

    show_table(
        model_results,
        title="Model performance comparison",
        filename_prefix="model_performance_comparison",
    )

    selected_model = st.selectbox(
        "Select model",
        [
            "Random Forest",
            "XGBoost",
        ],
    )

    if selected_model == "Random Forest":
        roc_file = "08_rf_roc_curve.png"
        confusion_file = "09_rf_confusion_matrix.png"
        importance_file = "10_rf_feature_importance.png"
        model_key = "random_forest"

    else:
        roc_file = "11_xgboost_roc_curve.png"
        confusion_file = "12_xgboost_confusion_matrix.png"
        importance_file = None
        model_key = "xgboost"

    model_left, model_right = st.columns(2)

    with model_left:
        show_image(
            roc_file,
            f"{selected_model} ROC curve — hold-out test set",
            f"{model_key}_roc_curve",
            f"{model_key}_roc",
        )

    with model_right:
        show_image(
            confusion_file,
            f"{selected_model} confusion matrix — hold-out test set (n = 50)",
            f"{model_key}_confusion_matrix_test_set",
            f"{model_key}_confusion",
        )

    st.caption(
        "The confusion matrix contains only the independent test samples. "
        "Training samples are not included because the model already saw "
        "them during fitting and they do not provide an unbiased measure "
        "of generalization."
    )

    if importance_file is not None:
        show_image(
            importance_file,
            "Top Random Forest genus features",
            "random_forest_genus_feature_importance",
            "rf_importance",
        )

    comparison = load_csv(
        "xgboost_feature_set_comparison.csv"
    )

    if not comparison.empty:
        show_table(
            comparison,
            title="XGBoost feature-set comparison",
            filename_prefix="xgboost_feature_set_comparison",
        )

    st.success(
        "Serratia alone, the top five genera and all 374 genera "
        "achieved ROC-AUC = 1.0 in the balanced cohort."
    )


# ============================================================
# Tab 7 — SHAP
# ============================================================

with tabs[6]:
    st.header("SHAP explainability")

    st.markdown(
        """
        SHAP quantifies how features influence model predictions.

        - Positive SHAP values push predictions toward **COPD**.
        - Negative SHAP values push predictions toward **Healthy**.
        """
    )

    shap_left, shap_right = st.columns(2)

    with shap_left:
        show_image(
            "shap_bar.png",
            "Global SHAP feature importance",
            "shap_global_importance",
            "shap_bar",
        )

    with shap_right:
        show_image(
            "shap_beeswarm.png",
            "SHAP beeswarm plot",
            "shap_beeswarm",
            "shap_beeswarm",
        )

    waterfall_left, waterfall_right = st.columns(2)

    with waterfall_left:
        show_image(
            "shap_waterfall_COPD.png",
            "SHAP waterfall — COPD sample",
            "shap_waterfall_copd",
            "shap_waterfall_copd",
        )

    with waterfall_right:
        show_image(
            "shap_waterfall_Healthy.png",
            "SHAP waterfall — Healthy sample",
            "shap_waterfall_healthy",
            "shap_waterfall_healthy",
        )

    shap_table = load_csv(
        "shap_top_features.csv"
    )

    if not shap_table.empty:
        if "Genus" in shap_table.columns:
            shap_table = shap_table.copy()
            shap_table["Genus"] = shap_table["Genus"].map(
                short_genus
            )

        shap_columns = [
            column
            for column in [
                "Genus",
                "Mean_absolute_SHAP",
            ]
            if column in shap_table.columns
        ]

        show_table(
            shap_table[shap_columns].head(20),
            title="Top 20 SHAP genus features",
            filename_prefix="top_20_shap_genus_features",
        )

    st.success(
        "Serratia was the dominant global predictor, followed by "
        "Acinetobacter, Leptotrichia, Escherichia–Shigella, "
        "Haemophilus and Moraxella."
    )


# ============================================================
# Tab 8 — LIME
# ============================================================

with tabs[7]:
    st.header("LIME local explanations")

    st.write(
        "LIME explains one individual prediction by fitting a simple "
        "local surrogate model around that sample."
    )

    lime_summary = load_csv(
        "lime_selected_samples_summary.csv"
    )

    if not lime_summary.empty:
        show_table(
            lime_summary,
            title="Selected LIME sample explanations",
            filename_prefix="lime_selected_samples",
        )

    lime_left, lime_right = st.columns(2)

    with lime_left:
        show_image(
            "lime_COPD_explanation.png",
            "LIME explanation — COPD sample",
            "lime_copd_explanation",
            "lime_copd",
        )

    with lime_right:
        show_image(
            "lime_Healthy_explanation.png",
            "LIME explanation — Healthy sample",
            "lime_healthy_explanation",
            "lime_healthy",
        )

    st.info(
        "High Serratia abundance supported the selected COPD prediction, "
        "whereas the absence of Serratia strongly supported the Healthy "
        "prediction."
    )


# ============================================================
# Tab 9 — Serratia
# ============================================================

with tabs[8]:
    st.header("Serratia cohort signal")

    serratia_col1, serratia_col2, serratia_col3, serratia_col4 = (
        st.columns(4)
    )

    metric_card(
        serratia_col1,
        "COPD prevalence",
        "100%",
    )

    metric_card(
        serratia_col2,
        "Healthy prevalence",
        "0.8%",
    )

    metric_card(
        serratia_col3,
        "COPD median abundance",
        "13.77%",
    )

    metric_card(
        serratia_col4,
        "Mann–Whitney p-value",
        "8.46 × 10⁻⁴⁸",
    )

    serratia_left, serratia_right = st.columns(2)

    with serratia_left:
        show_image(
            "serratia_boxplot.png",
            "Serratia relative abundance",
            "serratia_relative_abundance",
            "serratia_boxplot",
        )

    with serratia_right:
        show_image(
            "serratia_log_scatter.png",
            "Serratia abundance on a logarithmic scale",
            "serratia_log_abundance",
            "serratia_log",
        )

    group_summary = load_csv(
        "serratia_group_summary.csv"
    )

    prevalence_summary = load_csv(
        "serratia_prevalence_summary.csv"
    )

    summary_left, summary_right = st.columns(2)

    with summary_left:
        if not group_summary.empty:
            show_table(
                group_summary,
                title="Serratia abundance summary",
                filename_prefix="serratia_abundance_summary",
            )

    with summary_right:
        if not prevalence_summary.empty:
            show_table(
                prevalence_summary,
                title="Serratia prevalence summary",
                filename_prefix="serratia_prevalence_summary",
            )

    st.warning(
        "Serratia should be described as a cohort-discriminating feature, "
        "not as a clinically validated COPD biomarker."
    )


# ============================================================
# Tab 10 — Limitations
# ============================================================

with tabs[9]:
    st.header("Interpretation and limitations")

    st.markdown(
        """
        ### Important limitations

        1. Healthy and COPD samples originated from different BioProjects.
        2. Disease status is completely confounded with study origin.
        3. DNA extraction, primer design, sequencing platform, laboratory
           environment and cohort characteristics may contribute to the
           observed separation.
        4. Perfect model performance must not be interpreted as validated
           clinical diagnostic performance.
        5. Serratia requires independent validation in a study containing
           both COPD and Healthy subjects processed under the same protocol.
        6. Negative controls and contamination assessment would be important
           because Serratia showed a nearly cohort-specific distribution.
        7. A leave-one-study-out validation design would be preferable when
           multiple independent studies become available.
        8. Prospective and external validation would be necessary before
           translational or clinical use.
        """
    )

    st.info(
        "This application demonstrates an end-to-end workflow including "
        "sequencing QC, DADA2, ASV exploration, taxonomy, diversity, "
        "machine learning and explainable AI."
    )
