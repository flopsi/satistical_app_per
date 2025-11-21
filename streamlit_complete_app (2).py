"""
Statistical Data Analysis Platform - Streamlit App
Complete 6-Phase Workflow with Advanced Multi-Filter Sales Analysis

To run locally:
    pip install -r requirements.txt
    streamlit run streamlit_complete_app.py
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
    .filter-box {
        background: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'filtered_data' not in st.session_state:
    st.session_state.filtered_data = None

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
st.markdown("### Complete Data Analysis with Advanced Multi-Filter System")
st.markdown("---")

# Sidebar Navigation
st.sidebar.title("📊 Navigation")
phase = st.sidebar.radio(
    "Select Phase:",
    [
        "🏠 Home",
        "📥 Phase 1: Data Ingestion",
        "📊 Phase 2: Data Profiling",
        "🔍 Phase 3: Advanced Filtering & Analysis",
        "🔬 Phase 4: Statistical Testing",
        "📈 Phase 5: Visualization",
        "📄 Phase 6: Reporting"
    ]
)

st.sidebar.markdown("---")
if st.session_state.data is not None:
    st.sidebar.success(f"✅ Data loaded: {len(st.session_state.data)} rows")
    if st.session_state.filtered_data is not None:
        st.sidebar.info(f"🔍 Filtered: {len(st.session_state.filtered_data)} rows")

# ============================================================================
# HOME
# ============================================================================

if phase == "🏠 Home":
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Phases", "6", "Complete")
    with col2:
        st.metric("Analysis Types", "15+", "Ready")
    with col3:
        st.metric("Chart Types", "12+", "Interactive")
    with col4:
        st.metric("Filters", "Unlimited", "Multi-Select")

    st.markdown("---")
    st.markdown("### 🎯 Workflow Overview")

    workflow_df = pd.DataFrame({
        'Phase': ['1️⃣ Data Ingestion', '2️⃣ Data Profiling', '3️⃣ Advanced Filtering',
                  '4️⃣ Statistical Testing', '5️⃣ Visualization', '6️⃣ Reporting'],
        'Description': [
            'Load CSV, Excel, JSON, Parquet files',
            'Generate statistical profiles and correlations',
            'Multi-criteria filtering (Stage, Product, Year, etc.)',
            'Run hypothesis tests and outlier detection',
            'Create interactive charts and plots',
            'Generate reports and export filtered data'
        ]
    })

    st.dataframe(workflow_df, use_container_width=True)

    st.markdown("---")
    st.markdown("### 💡 Example Filter Query")
    st.markdown("""
    **"Analyze Stage=Won of Product Line=LSMS for years 2020 and 2021"**

    In Phase 3, you can:
    - Select multiple values from any column
    - Combine filters (Stage AND Product AND Year)
    - See results update in real-time
    - Export filtered data
    """)

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
                st.session_state.filtered_data = None  # Reset filters
                st.success(f"✅ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    with tab2:
        st.markdown("### Load Sample Dataset")
        sample = st.selectbox("Choose sample:", ["None", "Iris", "Tips", "Titanic", "SFDC Sales Demo"])

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
            elif sample == "SFDC Sales Demo":
                # Generate demo SFDC data matching your schema
                np.random.seed(42)
                n_rows = 500
                years = [2020, 2021, 2022, 2023]
                stages = ['Won', 'Lost', 'Open', 'Closed']
                products = ['LSMS', 'Proteomics', 'Genomics', 'Clinical']

                st.session_state.data = pd.DataFrame({
                    'SFDC Project No.': [f'SFDC-{i:05d}' for i in range(n_rows)],
                    'Accounts Name': [f'Account_{np.random.choice(["Alpha", "Beta", "Gamma", "Delta"])}' for _ in range(n_rows)],
                    'Current Account Manager': np.random.choice(['John', 'Jane', 'Bob', 'Alice', 'Charlie'], n_rows),
                    'Product Line': np.random.choice(products, n_rows),
                    'Total Price': np.random.randint(10000, 500000, n_rows),
                    'Close Year': pd.to_datetime([f'{np.random.choice(years)}-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}' for _ in range(n_rows)]),
                    'Past Account Manager': np.random.choice(['Mike', 'Sarah', 'Tom', 'Emma', 'David'], n_rows),
                    'Stage': np.random.choice(stages, n_rows)
                })
                st.success("✅ SFDC Sales demo dataset loaded!")
            st.session_state.filtered_data = None

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
# PHASE 3: ADVANCED FILTERING & ANALYSIS
# ============================================================================

elif phase == "🔍 Phase 3: Advanced Filtering & Analysis":
    st.markdown("# 🔍 Phase 3: Advanced Filtering & Analysis")
    st.markdown("### Apply multiple filters to analyze specific data segments")
    st.markdown("---")

    if st.session_state.data is None:
        st.warning("⚠️ Please load data in Phase 1 first!")
    else:
        df = st.session_state.data

        # Create filter section
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        st.markdown("### 🔍 Multi-Criteria Filters")
        st.markdown("**Select values from multiple columns to filter your data**")

        # Create columns for filter controls
        filter_cols = st.columns(3)

        filters = {}
        active_filters = []

        # Get all columns
        all_columns = df.columns.tolist()

        # Dynamic filter creation
        for idx, col in enumerate(all_columns):
            with filter_cols[idx % 3]:
                unique_vals = df[col].dropna().unique()

                # Different widget based on data type and unique values
                if len(unique_vals) <= 50:  # Categorical or low-cardinality
                    selected = st.multiselect(
                        f"🔹 {col}",
                        options=sorted(unique_vals.tolist(), key=str),
                        key=f"filter_{col}"
                    )
                    if selected:
                        filters[col] = selected
                        active_filters.append(f"{col} in {selected}")

                elif pd.api.types.is_numeric_dtype(df[col]):  # Numeric
                    min_val = float(df[col].min())
                    max_val = float(df[col].max())
                    range_vals = st.slider(
                        f"🔹 {col}",
                        min_val, max_val, (min_val, max_val),
                        key=f"filter_{col}"
                    )
                    if range_vals != (min_val, max_val):
                        filters[col] = range_vals
                        active_filters.append(f"{col} between {range_vals[0]:.0f} and {range_vals[1]:.0f}")

                elif pd.api.types.is_datetime64_any_dtype(df[col]):  # Date
                    min_date = df[col].min()
                    max_date = df[col].max()
                    date_range = st.date_input(
                        f"🔹 {col}",
                        value=(min_date, max_date),
                        key=f"filter_{col}"
                    )
                    if len(date_range) == 2 and date_range != (min_date, max_date):
                        filters[col] = date_range
                        active_filters.append(f"{col} between {date_range[0]} and {date_range[1]}")

        st.markdown('</div>', unsafe_allow_html=True)

        # Apply filters button
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            apply_button = st.button("🔍 Apply Filters", type="primary", use_container_width=True)
        with col2:
            reset_button = st.button("🔄 Reset Filters", use_container_width=True)

        if reset_button:
            st.session_state.filtered_data = None
            st.rerun()

        if apply_button or filters:
            # Apply all filters
            filtered_df = df.copy()

            for col, values in filters.items():
                if isinstance(values, list):  # Multiselect
                    filtered_df = filtered_df[filtered_df[col].isin(values)]
                elif isinstance(values, tuple) and len(values) == 2:  # Range
                    if pd.api.types.is_numeric_dtype(df[col]):
                        filtered_df = filtered_df[(filtered_df[col] >= values[0]) & (filtered_df[col] <= values[1])]
                    elif pd.api.types.is_datetime64_any_dtype(df[col]):
                        filtered_df = filtered_df[(filtered_df[col] >= pd.to_datetime(values[0])) & 
                                                  (filtered_df[col] <= pd.to_datetime(values[1]))]

            st.session_state.filtered_data = filtered_df

            # Show active filters
            if active_filters:
                st.markdown("### 🎯 Active Filters")
                for i, filter_desc in enumerate(active_filters, 1):
                    st.markdown(f"**{i}.** {filter_desc}")

            # Show results
            st.markdown("---")
            st.markdown("### 📊 Filtered Results")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Rows", f"{len(df):,}")
            with col2:
                st.metric("Filtered Rows", f"{len(filtered_df):,}", 
                         delta=f"{len(filtered_df) - len(df):,}")
            with col3:
                pct = (len(filtered_df) / len(df)) * 100 if len(df) > 0 else 0
                st.metric("Percentage", f"{pct:.1f}%")

            # Show filtered data
            st.markdown("### 📋 Filtered Data Preview")
            st.dataframe(filtered_df.head(20), use_container_width=True)

            # Analysis tabs for filtered data
            st.markdown("---")
            st.markdown("### 📈 Filtered Data Analysis")

            tab1, tab2, tab3 = st.tabs(["Summary Statistics", "Charts", "Export"])

            with tab1:
                numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    st.dataframe(filtered_df[numeric_cols].describe(), use_container_width=True)

                    # Group analysis
                    if len(filters) > 0:
                        st.markdown("### Group Breakdown")
                        for col in filters.keys():
                            if col in filtered_df.columns:
                                group_summary = filtered_df.groupby(col).size().reset_index(name='Count')
                                st.markdown(f"**By {col}:**")
                                st.dataframe(group_summary, use_container_width=True)

            with tab2:
                numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) > 0:
                    chart_col = st.selectbox("Select column to visualize:", numeric_cols)

                    col1, col2 = st.columns(2)
                    with col1:
                        fig1 = px.histogram(filtered_df, x=chart_col, title=f"Distribution: {chart_col}")
                        st.plotly_chart(fig1, use_container_width=True)
                    with col2:
                        fig2 = px.box(filtered_df, y=chart_col, title=f"Box Plot: {chart_col}")
                        st.plotly_chart(fig2, use_container_width=True)

                    # If categorical column exists, show breakdown
                    cat_cols = [c for c in filters.keys() if c in filtered_df.columns]
                    if cat_cols:
                        group_col = st.selectbox("Group by:", cat_cols)
                        fig3 = px.bar(filtered_df.groupby(group_col)[chart_col].sum().reset_index(),
                                     x=group_col, y=chart_col,
                                     title=f"{chart_col} by {group_col}")
                        st.plotly_chart(fig3, use_container_width=True)

            with tab3:
                st.markdown("### 💾 Export Filtered Data")

                col1, col2 = st.columns(2)
                with col1:
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="📄 Download Filtered CSV",
                        data=csv,
                        file_name="filtered_data.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with col2:
                    json_str = filtered_df.to_json(orient='records', indent=2)
                    st.download_button(
                        label="📄 Download Filtered JSON",
                        data=json_str,
                        file_name="filtered_data.json",
                        mime="application/json",
                        use_container_width=True
                    )
        else:
            st.info("👆 Select filter values above and click 'Apply Filters' to analyze specific data segments")
            st.markdown("""
            ### 💡 Example Queries You Can Build:
            - Stage = "Won" AND Product Line = "LSMS" AND Close Year in [2020, 2021]
            - Total Price between $50,000 and $200,000
            - Current Account Manager in ["John", "Jane"]
            - Any combination of multiple filters!
            """)

# ============================================================================
# PHASE 4: STATISTICAL TESTING
# ============================================================================

elif phase == "🔬 Phase 4: Statistical Testing":
    st.markdown("# 🔬 Phase 4: Statistical Testing")
    st.markdown("---")

    # Use filtered data if available
    df = st.session_state.filtered_data if st.session_state.filtered_data is not None else st.session_state.data

    if df is None:
        st.warning("⚠️ Please load data in Phase 1 first!")
    else:
        if st.session_state.filtered_data is not None:
            st.info(f"ℹ️ Analyzing filtered data ({len(df)} rows)")

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

    # Use filtered data if available
    df = st.session_state.filtered_data if st.session_state.filtered_data is not None else st.session_state.data

    if df is None:
        st.warning("⚠️ Please load data in Phase 1 first!")
    else:
        if st.session_state.filtered_data is not None:
            st.info(f"ℹ️ Visualizing filtered data ({len(df)} rows)")

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

    # Use filtered data if available
    df = st.session_state.filtered_data if st.session_state.filtered_data is not None else st.session_state.data

    if df is None:
        st.warning("⚠️ Please load data in Phase 1 first!")
    else:
        if st.session_state.filtered_data is not None:
            st.info(f"ℹ️ Reporting on filtered data ({len(df)} rows)")

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
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            json_str = df.to_json(orient='records', indent=2)
            st.download_button(
                label="📄 Download JSON",
                data=json_str,
                file_name="analysis_data.json",
                mime="application/json",
                use_container_width=True
            )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Statistical Data Analysis Platform with Advanced Multi-Filter System | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
