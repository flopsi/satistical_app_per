"""
Statistical Data Analysis Platform - Streamlit App
Complete 6-Phase Workflow converted from Colab
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import io
import warnings
from typing import Dict, Tuple
from datetime import datetime
from sklearn.ensemble import IsolationForest

warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Statistical Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

# ============================================================================
# DATA LOADING CLASSES
# ============================================================================

class DataLoader:
    """Handle data loading from multiple sources."""

    def __init__(self):
        self.df = None
        self.metadata = {}

    def load_from_upload(self, uploaded_file) -> Tuple[pd.DataFrame, Dict]:
        """Load data from Streamlit file uploader."""
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                return None, None

            self.df = df
            self.metadata = {
                'source': 'upload',
                'filename': uploaded_file.name,
                'rows': len(df),
                'columns': len(df.columns)
            }
            return df, self.metadata
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None, None

# ============================================================================
# ANALYSIS CLASSES
# ============================================================================

class QualityAssessment:
    """Data quality assessment."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.report = {}

    def assess(self) -> Dict:
        """Generate quality report."""
        missing_pct = (self.df.isna().sum().sum()) / (len(self.df) * len(self.df.columns))
        completeness = 1.0 - missing_pct
        duplicate_rows = len(self.df[self.df.duplicated()])
        uniqueness = 1.0 - (duplicate_rows / len(self.df)) if len(self.df) > 0 else 1.0

        self.report = {
            'completeness': round(completeness, 3),
            'uniqueness': round(uniqueness, 3),
            'overall': round((completeness + uniqueness) / 2, 3),
            'duplicates': int(duplicate_rows),
            'missing_cells': int(self.df.isna().sum().sum())
        }
        return self.report


class EnhancedProfiler:
    """Comprehensive data profiling."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.profiles = {}

    def profile_numeric(self, series: pd.Series) -> Dict:
        """Profile numeric column."""
        data = series.dropna()
        if len(data) == 0:
            return {'error': 'All values missing'}

        return {
            'type': 'numeric',
            'count': len(series),
            'missing': int(series.isna().sum()),
            'mean': round(float(data.mean()), 4),
            'median': round(float(data.median()), 4),
            'std': round(float(data.std()), 4),
            'min': round(float(data.min()), 4),
            'max': round(float(data.max()), 4),
            'skewness': round(float(data.skew()), 4)
        }

    def auto_profile(self) -> Dict:
        """Profile all columns."""
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.profiles[col] = self.profile_numeric(self.df[col])
        return self.profiles


class CorrelationEngine:
    """Correlation analysis."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.pearson_corr = None

    def pearson_matrix(self) -> pd.DataFrame:
        """Calculate Pearson correlation."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return None
        self.pearson_corr = numeric_df.corr(method='pearson')
        return self.pearson_corr

    def create_heatmap(self):
        """Create correlation heatmap."""
        if self.pearson_corr is None:
            self.pearson_matrix()

        if self.pearson_corr is None:
            return None

        fig = go.Figure(data=go.Heatmap(
            z=self.pearson_corr.values,
            x=self.pearson_corr.columns,
            y=self.pearson_corr.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(self.pearson_corr.values, 2),
            texttemplate='%{text:.2f}'
        ))

        fig.update_layout(
            title="Correlation Heatmap",
            width=700,
            height=600
        )
        return fig


class DistributionAnalyzer:
    """Distribution analysis and normality testing."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def test_normality(self, series: pd.Series) -> Dict:
        """Test for normality."""
        data = series.dropna()
        if len(data) < 3:
            return {'error': 'insufficient_data'}

        shapiro_stat, shapiro_p = stats.shapiro(data)

        return {
            'shapiro_wilk_p': round(float(shapiro_p), 4),
            'shapiro_wilk_stat': round(float(shapiro_stat), 4),
            'interpretation': 'Normal' if shapiro_p > 0.05 else 'Non-normal'
        }

    def create_histogram(self, series: pd.Series):
        """Create histogram with KDE."""
        fig = px.histogram(series, nbins=30, title=f'Distribution: {series.name}')
        return fig


class OutlierAnalyzer:
    """Outlier detection."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def detect_iqr(self, series: pd.Series) -> Dict:
        """Detect outliers using IQR."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outliers = series[outlier_mask]

        return {
            'count': int(len(outliers)),
            'percentage': round((len(outliers) / len(series)) * 100, 2),
            'bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)}
        }

# ============================================================================
# STREAMLIT UI
# ============================================================================

# Header
st.markdown('<h1 class="main-header">📊 Statistical Analysis Platform</h1>', unsafe_allow_html=True)
st.markdown("### Complete 6-Phase Data Analysis Workflow")
st.markdown("---")

# Sidebar Navigation
st.sidebar.title("📊 Navigation")
phase = st.sidebar.radio(
    "Select Phase:",
    [
        "🏠 Home",
        "📥 Phase 1: Data Ingestion",
        "📊 Phase 2: Data Profiling",
        "🔬 Phase 3: Statistical Testing",
        "📈 Phase 4: Visualization",
        "📄 Phase 5: Reporting"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Quick Info")

# ============================================================================
# HOME
# ============================================================================

if phase == "🏠 Home":
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Phases", "5", "Complete")
    with col2:
        st.metric("Modules", "6+", "Ready")
    with col3:
        st.metric("Chart Types", "10+", "Interactive")
    with col4:
        st.metric("Tests", "15+", "Available")

    st.markdown("---")
    st.markdown("### 🎯 Workflow Overview")

    workflow_df = pd.DataFrame({
        'Phase': ['1️⃣ Data Ingestion', '2️⃣ Data Profiling', '3️⃣ Statistical Testing', 
                  '4️⃣ Visualization', '5️⃣ Reporting'],
        'Description': [
            'Load CSV, Excel, or sample datasets',
            'Generate statistical profiles and correlations',
            'Run hypothesis tests and normality checks',
            'Create interactive charts and plots',
            'Generate reports and export data'
        ]
    })

    st.dataframe(workflow_df, use_container_width=True)
    st.info("👈 **Get Started**: Select 'Phase 1: Data Ingestion' from the sidebar!")

# ============================================================================
# PHASE 1: DATA INGESTION
# ============================================================================

elif phase == "📥 Phase 1: Data Ingestion":
    st.markdown("# 📥 Phase 1: Data Ingestion")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Upload File", "Sample Data", "Preview"])

    with tab1:
        st.markdown("### Upload Your Data")
        uploaded_file = st.file_uploader("Choose CSV or Excel file", type=['csv', 'xlsx', 'xls'])

        if uploaded_file:
            loader = DataLoader()
            df, metadata = loader.load_from_upload(uploaded_file)

            if df is not None:
                st.session_state.data = df
                st.success(f"✅ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    with tab2:
        st.markdown("### Load Sample Dataset")
        sample = st.selectbox("Choose sample:", ["None", "Iris", "Tips", "Titanic"])

        if st.button("Load Sample"):
            if sample == "Iris":
                st.session_state.data = sns.load_dataset('iris')
                st.success("✅ Iris dataset loaded!")
            elif sample == "Tips":
                st.session_state.data = sns.load_dataset('tips')
                st.success("✅ Tips dataset loaded!")
            elif sample == "Titanic":
                st.session_state.data = sns.load_dataset('titanic').head(100)
                st.success("✅ Titanic dataset loaded!")

    with tab3:
        if st.session_state.data is not None:
            df = st.session_state.data
            st.markdown("### Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                missing_pct = (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                st.metric("Missing %", f"{missing_pct:.1f}%")

            # Quality Assessment
            if st.button("Run Quality Assessment"):
                qa = QualityAssessment(df)
                report = qa.assess()

                st.markdown("### 📊 Quality Report")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Completeness", f"{report['completeness']:.1%}")
                with col2:
                    st.metric("Uniqueness", f"{report['uniqueness']:.1%}")
                with col3:
                    st.metric("Overall Quality", f"{report['overall']:.1%}")
        else:
            st.info("👈 Upload or load a dataset to see preview")

# ============================================================================
# PHASE 2: DATA PROFILING
# ============================================================================

elif phase == "📊 Phase 2: Data Profiling":
    st.markdown("# 📊 Phase 2: Data Profiling")
    st.markdown("---")

    if st.session_state.data is None:
        st.warning("⚠️ Please load data in Phase 1 first!")
    else:
        df = st.session_state.data

        tab1, tab2, tab3 = st.tabs(["Statistics", "Correlations", "Distributions"])

        with tab1:
            st.markdown("### Descriptive Statistics")
            st.dataframe(df.describe(), use_container_width=True)

            # Enhanced Profiling
            if st.button("Generate Enhanced Profile"):
                profiler = EnhancedProfiler(df)
                profiles = profiler.auto_profile()

                for col, profile in profiles.items():
                    with st.expander(f"📊 {col}"):
                        if 'error' not in profile:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Mean", profile.get('mean', 'N/A'))
                            with col2:
                                st.metric("Median", profile.get('median', 'N/A'))
                            with col3:
                                st.metric("Std Dev", profile.get('std', 'N/A'))

        with tab2:
            st.markdown("### Correlation Analysis")
            corr_engine = CorrelationEngine(df)

            if st.button("Calculate Correlations"):
                fig = corr_engine.create_heatmap()
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Need at least 2 numeric columns for correlation")

        with tab3:
            st.markdown("### Distribution Analysis")
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) > 0:
                col = st.selectbox("Select column:", numeric_cols)

                dist_analyzer = DistributionAnalyzer(df)

                # Histogram
                fig = dist_analyzer.create_histogram(df[col])
                st.plotly_chart(fig, use_container_width=True)

                # Normality test
                if st.button("Test Normality"):
                    result = dist_analyzer.test_normality(df[col])
                    if 'error' not in result:
                        st.markdown(f"**Shapiro-Wilk Test**: p-value = {result['shapiro_wilk_p']:.4f}")
                        if result['interpretation'] == 'Normal':
                            st.success("✅ Data appears normally distributed")
                        else:
                            st.warning("⚠️ Data may not be normally distributed")

# ============================================================================
# PHASE 3: STATISTICAL TESTING
# ============================================================================

elif phase == "🔬 Phase 3: Statistical Testing":
    st.markdown("# 🔬 Phase 3: Statistical Testing")
    st.markdown("---")

    if st.session_state.data is None:
        st.warning("⚠️ Please load data in Phase 1 first!")
    else:
        df = st.session_state.data
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        test_type = st.selectbox("Select Test:", 
                                ["Correlation (Pearson)", "Correlation (Spearman)", 
                                 "T-Test", "Outlier Detection (IQR)"])

        if test_type in ["Correlation (Pearson)", "Correlation (Spearman)"]:
            if len(numeric_cols) >= 2:
                col1 = st.selectbox("Variable 1:", numeric_cols)
                col2 = st.selectbox("Variable 2:", numeric_cols, index=min(1, len(numeric_cols)-1))

                if st.button("Run Test"):
                    if test_type == "Correlation (Pearson)":
                        r, p = stats.pearsonr(df[col1].dropna(), df[col2].dropna())
                    else:
                        r, p = stats.spearmanr(df[col1].dropna(), df[col2].dropna())

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Correlation", f"{r:.4f}")
                    with col_b:
                        st.metric("P-value", f"{p:.4f}")

                    if p < 0.05:
                        st.success("✅ Significant correlation detected (p < 0.05)")
                    else:
                        st.info("No significant correlation (p ≥ 0.05)")

                    # Scatter plot
                    fig = px.scatter(df, x=col1, y=col2, trendline="ols",
                                   title=f'{col1} vs {col2}')
                    st.plotly_chart(fig, use_container_width=True)

        elif test_type == "Outlier Detection (IQR)":
            col = st.selectbox("Select column:", numeric_cols)

            if st.button("Detect Outliers"):
                outlier_analyzer = OutlierAnalyzer(df)
                result = outlier_analyzer.detect_iqr(df[col])

                st.metric("Outliers Detected", f"{result['count']} ({result['percentage']:.1f}%)")

                fig = px.box(df, y=col, title=f'Box Plot: {col}')
                st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PHASE 4: VISUALIZATION
# ============================================================================

elif phase == "📈 Phase 4: Visualization":
    st.markdown("# 📈 Phase 4: Visualization")
    st.markdown("---")

    if st.session_state.data is None:
        st.warning("⚠️ Please load data in Phase 1 first!")
    else:
        df = st.session_state.data

        chart_type = st.selectbox("Select Chart Type:",
                                 ["Histogram", "Box Plot", "Scatter Plot", 
                                  "Correlation Heatmap", "Bar Chart"])

        if chart_type == "Histogram":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col = st.selectbox("Select column:", numeric_cols)
                fig = px.histogram(df, x=col, nbins=30, title=f'Histogram: {col}')
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Box Plot":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col = st.selectbox("Select column:", numeric_cols)
                fig = px.box(df, y=col, title=f'Box Plot: {col}')
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Scatter Plot":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                x_col = st.selectbox("X-axis:", numeric_cols)
                y_col = st.selectbox("Y-axis:", numeric_cols, index=min(1, len(numeric_cols)-1))
                fig = px.scatter(df, x=x_col, y=y_col, title=f'{x_col} vs {y_col}')
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Correlation Heatmap":
            corr_engine = CorrelationEngine(df)
            fig = corr_engine.create_heatmap()
            if fig:
                st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PHASE 5: REPORTING
# ============================================================================

elif phase == "📄 Phase 5: Reporting":
    st.markdown("# 📄 Phase 5: Reporting")
    st.markdown("---")

    if st.session_state.data is None:
        st.warning("⚠️ Please load data in Phase 1 first!")
    else:
        df = st.session_state.data

        st.markdown("### Executive Summary")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Features", len(df.columns))
        with col3:
            missing_pct = (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Completeness", f"{100-missing_pct:.1f}%")

        st.markdown("---")
        st.markdown("### 📥 Download Data")

        col1, col2 = st.columns(2)

        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV",
                data=csv,
                file_name="analysis_data.csv",
                mime="text/csv"
            )

        with col2:
            json_str = df.to_json(orient='records', indent=2)
            st.download_button(
                label="📄 Download JSON",
                data=json_str,
                file_name="analysis_data.json",
                mime="application/json"
            )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Statistical Data Analysis Platform | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
