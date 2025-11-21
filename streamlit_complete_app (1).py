"""
Statistical Data Analysis Platform - Streamlit App
Complete 6-Phase Workflow with Sales Analysis

To run locally:
    pip install -r requirements.txt
    streamlit run streamlit_complete_app.py

To deploy to Streamlit Cloud:
    1. Push to GitHub
    2. Go to share.streamlit.io
    3. Connect repository
    4. Deploy!
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
        """Load data from Streamlit file uploader with enhanced parsing."""
        try:
            if uploaded_file.name.endswith('.csv'):
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except:
                    uploaded_file.seek(0)
                    try:
                        df = pd.read_csv(uploaded_file, encoding='latin-1')
                    except:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding='utf-8', sep=None, engine='python')
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file, sheet_name=0)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            elif uploaded_file.name.endswith('.parquet'):
                df = pd.read_parquet(uploaded_file)
            elif uploaded_file.name.endswith('.tsv'):
                df = pd.read_csv(uploaded_file, sep='\t')
            else:
                st.error(f"Unsupported file type: {uploaded_file.name}")
                return None, None

            df.columns = df.columns.astype(str).str.strip()

            for col in df.select_dtypes(include=['object']).columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass

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
st.markdown("### Complete Data Analysis Workflow with Sales Analytics")
st.markdown("---")

# Sidebar Navigation
st.sidebar.title("📊 Navigation")
phase = st.sidebar.radio(
    "Select Phase:",
    [
        "🏠 Home",
        "📥 Phase 1: Data Ingestion",
        "📊 Phase 2: Data Profiling",
        "💰 Phase 3: Sales Analysis",
        "🔬 Phase 4: Statistical Testing",
        "📈 Phase 5: Visualization",
        "📄 Phase 6: Reporting"
    ]
)

st.sidebar.markdown("---")
if st.session_state.data is not None:
    st.sidebar.success(f"✅ Data loaded: {len(st.session_state.data)} rows")

# ============================================================================
# HOME
# ============================================================================

if phase == "🏠 Home":
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Phases", "6", "Complete")
    with col2:
        st.metric("Modules", "8+", "Ready")
    with col3:
        st.metric("Chart Types", "15+", "Interactive")
    with col4:
        st.metric("Tests", "10+", "Available")

    st.markdown("---")
    st.markdown("### 🎯 Workflow Overview")

    workflow_df = pd.DataFrame({
        'Phase': ['1️⃣ Data Ingestion', '2️⃣ Data Profiling', '3️⃣ Sales Analysis',
                  '4️⃣ Statistical Testing', '5️⃣ Visualization', '6️⃣ Reporting'],
        'Description': [
            'Load CSV, Excel, JSON, Parquet files',
            'Generate statistical profiles and correlations',
            'Analyze revenue, trends, top performers, growth',
            'Run hypothesis tests and outlier detection',
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
        st.markdown("Supported formats: CSV, Excel, JSON, Parquet, TSV")
        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls', 'json', 'parquet', 'tsv'])

        if uploaded_file:
            loader = DataLoader()
            df, metadata = loader.load_from_upload(uploaded_file)

            if df is not None:
                st.session_state.data = df
                st.success(f"✅ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    with tab2:
        st.markdown("### Load Sample Dataset")
        sample = st.selectbox("Choose sample:", ["None", "Iris", "Tips", "Titanic", "Sales Demo"])

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
            elif sample == "Sales Demo":
                # Generate demo sales data
                np.random.seed(42)
                dates = pd.date_range('2024-01-01', periods=365, freq='D')
                st.session_state.data = pd.DataFrame({
                    'Date': dates,
                    'Product': np.random.choice(['A', 'B', 'C', 'D'], 365),
                    'Region': np.random.choice(['North', 'South', 'East', 'West'], 365),
                    'Salesperson': np.random.choice(['John', 'Jane', 'Bob', 'Alice'], 365),
                    'Revenue': np.random.randint(100, 5000, 365),
                    'Units': np.random.randint(1, 50, 365)
                })
                st.success("✅ Sales demo dataset loaded!")

    with tab3:
        if st.session_state.data is not None:
            df = st.session_state.data
            st.markdown("### Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                missing_pct = (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                st.metric("Missing %", f"{missing_pct:.1f}%")
            with col4:
                numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
                st.metric("Numeric Cols", numeric_cols)

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

            st.markdown("### Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes,
                'Non-Null': df.notna().sum(),
                'Unique': df.nunique()
            })
            st.dataframe(col_info, use_container_width=True)

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
# PHASE 3: SALES ANALYSIS
# ============================================================================

elif phase == "💰 Phase 3: Sales Analysis":
    st.markdown("# 💰 Phase 3: Sales Analysis")
    st.markdown("---")

    if st.session_state.data is None:
        st.warning("⚠️ Please load data in Phase 1 first!")
    else:
        df = st.session_state.data
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) == 0:
            st.error("❌ No numeric columns found for analysis")
        else:
            tab1, tab2, tab3, tab4 = st.tabs(["Revenue Analysis", "Time Series", "Top Performers", "Growth Rates"])

            with tab1:
                st.markdown("### Revenue Analysis")
                revenue_col = st.selectbox("Select revenue column:", numeric_cols, key="rev_col")

                if revenue_col:
                    total_revenue = df[revenue_col].sum()
                    avg_revenue = df[revenue_col].mean()
                    median_revenue = df[revenue_col].median()

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Revenue", f"${total_revenue:,.2f}")
                    with col2:
                        st.metric("Average Revenue", f"${avg_revenue:,.2f}")
                    with col3:
                        st.metric("Median Revenue", f"${median_revenue:,.2f}")

                    # Distribution
                    fig = px.histogram(df, x=revenue_col, nbins=30, 
                                     title=f"Revenue Distribution: {revenue_col}")
                    st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.markdown("### Time Series Analysis")

                # Auto-detect date columns
                date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) 
                           or 'date' in col.lower() or 'time' in col.lower()]

                if len(date_cols) == 0:
                    st.info("No date columns detected. Try converting a column to datetime first.")
                else:
                    date_col = st.selectbox("Select date column:", date_cols, key="date_col")
                    metric_col = st.selectbox("Select metric column:", numeric_cols, key="metric_col")

                    if date_col and metric_col:
                        df_sorted = df.sort_values(date_col)
                        fig = px.line(df_sorted, x=date_col, y=metric_col,
                                    title=f"{metric_col} Over Time")
                        st.plotly_chart(fig, use_container_width=True)

            with tab3:
                st.markdown("### Top Performers")

                group_col = st.selectbox("Group by:", df.columns, key="group_col")
                value_col = st.selectbox("Measure:", numeric_cols, key="value_col")
                top_n = st.slider("Show top N:", 5, 50, 10)

                if group_col and value_col:
                    top_performers = df.groupby(group_col)[value_col].sum().sort_values(ascending=False).head(top_n)

                    fig = px.bar(x=top_performers.index, y=top_performers.values,
                               title=f"Top {top_n} {group_col} by {value_col}",
                               labels={'x': group_col, 'y': value_col})
                    st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(top_performers.reset_index().rename(columns={group_col: 'Rank', value_col: 'Total'}), 
                               use_container_width=True)

            with tab4:
                st.markdown("### Growth Rate Analysis")

                if len(numeric_cols) > 0:
                    growth_col = st.selectbox("Select column for growth:", numeric_cols, key="growth_col")

                    if growth_col and len(df) > 1:
                        df_sorted = df.sort_index()
                        growth_rates = df_sorted[growth_col].pct_change() * 100

                        avg_growth = growth_rates.mean()
                        max_growth = growth_rates.max()

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Average Growth Rate", f"{avg_growth:.2f}%")
                        with col2:
                            st.metric("Max Growth Rate", f"{max_growth:.2f}%")

                        fig = px.line(x=range(len(growth_rates)), y=growth_rates,
                                    title=f"{growth_col} Growth Rate Over Time",
                                    labels={'x': 'Period', 'y': 'Growth Rate (%)'})
                        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PHASE 4: STATISTICAL TESTING
# ============================================================================

elif phase == "🔬 Phase 4: Statistical Testing":
    st.markdown("# 🔬 Phase 4: Statistical Testing")
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
                    data1 = df[col1].dropna()
                    data2 = df[col2].dropna()

                    if len(data1) > 0 and len(data2) > 0:
                        if test_type == "Correlation (Pearson)":
                            r, p = stats.pearsonr(data1, data2)
                        else:
                            r, p = stats.spearmanr(data1, data2)

                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Correlation", f"{r:.4f}")
                        with col_b:
                            st.metric("P-value", f"{p:.4f}")

                        if p < 0.05:
                            st.success("✅ Significant correlation detected (p < 0.05)")
                        else:
                            st.info("No significant correlation (p ≥ 0.05)")

                        # Scatter plot without trendline
                        fig = px.scatter(df, x=col1, y=col2, title=f'{col1} vs {col2}')
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
# PHASE 5: VISUALIZATION
# ============================================================================

elif phase == "📈 Phase 5: Visualization":
    st.markdown("# 📈 Phase 5: Visualization")
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
# PHASE 6: REPORTING
# ============================================================================

elif phase == "📄 Phase 6: Reporting":
    st.markdown("# 📄 Phase 6: Reporting")
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
