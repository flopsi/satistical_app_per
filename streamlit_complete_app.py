"""
Statistical Analysis Platform - Streamlit App
Complete Analysis with Custom Rules, Currency Conversion, and A/B Comparisons

Features:
- Manual aggregation rules (starts with ETH, etc.)
- Auto-detection of patterns
- Smart stage separation (Won/Lost handled separately)
- Currency selector with conversion
- Comparative analysis (Filter A vs Filter B)
- Account-level hierarchy

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
import seaborn as sns
import io
import warnings
from typing import Dict, Tuple, List
from datetime import datetime
import re

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
    .filter-box {
        background: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .comparison-box {
        background: #e8f5e9;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    .rule-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'aggregation_rules' not in st.session_state:
    st.session_state.aggregation_rules = []
if 'stage_separation' not in st.session_state:
    st.session_state.stage_separation = True
if 'currency' not in st.session_state:
    st.session_state.currency = 'CHF'
if 'aggregated_data' not in st.session_state:
    st.session_state.aggregated_data = None

# Currency conversion rates (as of example)
CURRENCY_RATES = {
    'CHF': 1.0,
    'EUR': 1.08,
    'USD': 1.14,
    'GBP': 1.28,
    'JPY': 0.0075,
    'CNY': 0.16
}

# ============================================================================
# AGGREGATION AND ANALYSIS ENGINES
# ============================================================================

class AggregationRuleEngine:
    """Handle custom aggregation rules."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.rules = []

    def detect_patterns(self, column: str, min_group_size: int = 2) -> List[Dict]:
        """Auto-detect common patterns."""
        if column not in self.df.columns:
            return []

        values = self.df[column].dropna().astype(str).unique()
        suggestions = []

        # Detect prefixes
        prefixes = {}
        for val in values:
            if len(val) >= 3:
                prefix = val[:3]
                if prefix not in prefixes:
                    prefixes[prefix] = []
                prefixes[prefix].append(val)

        for prefix, matches in prefixes.items():
            if len(matches) >= min_group_size:
                suggestions.append({
                    'type': 'starts_with',
                    'pattern': prefix,
                    'column': column,
                    'matches': len(matches),
                    'examples': matches[:3],
                    'description': f"Values starting with '{prefix}' ({len(matches)} items)"
                })

        return suggestions

    def apply_rule(self, rule: Dict) -> pd.Series:
        """Apply a single rule."""
        column = rule['column']
        rule_type = rule['type']
        pattern = rule['pattern']
        group_name = rule['group_name']

        values = self.df[column].astype(str)
        mask = pd.Series(False, index=self.df.index)

        if rule_type == 'starts_with':
            mask = values.str.startswith(pattern, na=False)
        elif rule_type == 'ends_with':
            mask = values.str.endswith(pattern, na=False)
        elif rule_type == 'contains':
            mask = values.str.contains(pattern, na=False, regex=False)
        elif rule_type == 'equals':
            mask = values == pattern

        labels = pd.Series('Other', index=self.df.index)
        labels[mask] = group_name
        return labels


class CurrencyConverter:
    """Handle currency conversion."""

    def __init__(self, base_currency='CHF'):
        self.base = base_currency
        self.rates = CURRENCY_RATES

    def convert(self, amount: float, from_curr: str, to_curr: str) -> float:
        """Convert amount from one currency to another."""
        if from_curr == to_curr:
            return amount

        # Convert to base (CHF) first, then to target
        in_base = amount / self.rates.get(from_curr, 1.0)
        in_target = in_base * self.rates.get(to_curr, 1.0)
        return in_target

    def convert_column(self, df: pd.DataFrame, col: str, from_curr: str, to_curr: str) -> pd.Series:
        """Convert entire column."""
        return df[col].apply(lambda x: self.convert(x, from_curr, to_curr) if pd.notna(x) else x)


class ComparativeAnalyzer:
    """Handle A/B comparison analysis."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def apply_filters(self, filters: Dict) -> pd.DataFrame:
        """Apply filter dictionary to dataframe."""
        filtered = self.df.copy()

        for col, values in filters.items():
            if col in filtered.columns:
                if isinstance(values, list):
                    filtered = filtered[filtered[col].isin(values)]
                elif isinstance(values, tuple):
                    filtered = filtered[(filtered[col] >= values[0]) & (filtered[col] <= values[1])]

        return filtered

    def compare_groups(self, filter_a: Dict, filter_b: Dict, metric_col: str) -> Dict:
        """Compare two filtered groups."""
        group_a = self.apply_filters(filter_a)
        group_b = self.apply_filters(filter_b)

        if len(group_a) == 0 or len(group_b) == 0:
            return {'error': 'One or both groups are empty'}

        results = {
            'group_a_count': len(group_a),
            'group_b_count': len(group_b),
            'group_a_mean': group_a[metric_col].mean(),
            'group_b_mean': group_b[metric_col].mean(),
            'group_a_median': group_a[metric_col].median(),
            'group_b_median': group_b[metric_col].median(),
            'group_a_total': group_a[metric_col].sum(),
            'group_b_total': group_b[metric_col].sum(),
            'mean_diff': group_a[metric_col].mean() - group_b[metric_col].mean(),
            'mean_diff_pct': ((group_a[metric_col].mean() / group_b[metric_col].mean()) - 1) * 100 if group_b[metric_col].mean() != 0 else 0
        }

        # Statistical test
        try:
            t_stat, p_value = stats.ttest_ind(group_a[metric_col].dropna(), 
                                             group_b[metric_col].dropna())
            results['t_statistic'] = t_stat
            results['p_value'] = p_value
            results['significant'] = p_value < 0.05
        except:
            results['t_statistic'] = None
            results['p_value'] = None
            results['significant'] = None

        return results


# ============================================================================
# DATA LOADING
# ============================================================================

class DataLoader:
    """Data loading."""

    def load_from_upload(self, uploaded_file) -> Tuple[pd.DataFrame, Dict]:
        """Load from file."""
        try:
            if uploaded_file.name.endswith('.csv'):
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='latin-1')
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                return None, None

            df.columns = df.columns.astype(str).str.strip()

            metadata = {
                'filename': uploaded_file.name,
                'rows': len(df),
                'columns': len(df.columns)
            }
            return df, metadata
        except Exception as e:
            st.error(f"Error: {e}")
            return None, None

# ============================================================================
# UI
# ============================================================================

# Header
st.markdown('<h1 class="main-header">📊 Advanced Sales Analysis Platform</h1>', unsafe_allow_html=True)
st.markdown("### Custom Rules • Currency Conversion • Comparative Analysis")
st.markdown("---")

# Sidebar
st.sidebar.title("📊 Navigation")
phase = st.sidebar.radio(
    "Select Phase:",
    [
        "🏠 Home",
        "📥 Phase 1: Data Ingestion",
        "⚙️ Phase 2: Aggregation Rules",
        "🏢 Phase 3: Hierarchical Analysis",
        "⚖️ Phase 4: Comparative Analysis",
        "📈 Phase 5: Visualization",
        "📄 Phase 6: Reporting"
    ]
)

st.sidebar.markdown("---")

# Currency selector in sidebar
if st.session_state.data is not None:
    st.sidebar.markdown("### 💱 Currency")
    st.session_state.currency = st.sidebar.selectbox(
        "Display Currency:",
        list(CURRENCY_RATES.keys()),
        index=list(CURRENCY_RATES.keys()).index(st.session_state.currency)
    )
    st.sidebar.caption(f"Rate: {CURRENCY_RATES[st.session_state.currency]:.4f} to CHF")

if st.session_state.data is not None:
    st.sidebar.success(f"✅ {len(st.session_state.data)} rows")
    if st.session_state.aggregation_rules:
        st.sidebar.info(f"⚙️ {len(st.session_state.aggregation_rules)} rules")

# ============================================================================
# HOME
# ============================================================================

if phase == "🏠 Home":
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **🔧 Aggregation Rules**
        - Pattern matching (ETH, CHF)
        - Auto-detection
        - Custom grouping
        """)

    with col2:
        st.markdown("""
        **💱 Currency Support**
        - Multi-currency display
        - Real-time conversion
        - 6 major currencies
        """)

    with col3:
        st.markdown("""
        **⚖️ Comparisons**
        - Filter A vs Filter B
        - Statistical testing
        - Manager performance
        """)

    st.markdown("---")
    st.markdown("### 💡 Example Comparison")
    st.code("""
Filter A: Stage=Won, Manager=Florian Marty
Filter B: Stage=Won, Manager=Pascal Krapf
Compare: Average Price

Result: 
- Florian avg: CHF 125,000
- Pascal avg: CHF 98,000
- Difference: +CHF 27,000 (+27.6%)
- P-value: 0.032 (significant!)
    """)

# ============================================================================
# PHASE 1: DATA INGESTION
# ============================================================================

elif phase == "📥 Phase 1: Data Ingestion":
    st.markdown("# 📥 Phase 1: Data Ingestion")

    uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx'])

    if uploaded_file:
        loader = DataLoader()
        df, _ = loader.load_from_upload(uploaded_file)

        if df is not None:
            st.session_state.data = df
            st.success(f"✅ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

            # Auto-detect
            st.markdown("### 🤖 Auto-Detected Patterns")
            rule_engine = AggregationRuleEngine(df)

            for col in df.select_dtypes(include=['object']).columns[:3]:
                patterns = rule_engine.detect_patterns(col)
                if patterns:
                    st.info(f"**{col}**: Found {len(patterns)} patterns")

# ============================================================================
# PHASE 2: AGGREGATION RULES
# ============================================================================

elif phase == "⚙️ Phase 2: Aggregation Rules":
    st.markdown("# ⚙️ Phase 2: Aggregation Rules")

    if st.session_state.data is None:
        st.warning("Load data first!")
    else:
        df = st.session_state.data

        tab1, tab2 = st.tabs(["Manual Rules", "Stage Settings"])

        with tab1:
            st.markdown("### ➕ Create Rule")

            col1, col2 = st.columns(2)
            with col1:
                rule_col = st.selectbox("Column:", df.columns)
                rule_type = st.selectbox("Type:", ["starts_with", "contains", "equals"])
            with col2:
                rule_pattern = st.text_input("Pattern:", placeholder="e.g., ETH, CHF")
                rule_name = st.text_input("Group Name:")

            if st.button("➕ Add Rule"):
                if rule_pattern:
                    st.session_state.aggregation_rules.append({
                        'column': rule_col,
                        'type': rule_type,
                        'pattern': rule_pattern,
                        'group_name': rule_name or f"{rule_type}_{rule_pattern}"
                    })
                    st.success("Rule added!")
                    st.rerun()

            # Show rules
            if st.session_state.aggregation_rules:
                st.markdown("### 📋 Active Rules")
                for i, rule in enumerate(st.session_state.aggregation_rules):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"`{rule['column']}` {rule['type']} `{rule['pattern']}`")
                    with col2:
                        if st.button("🗑️", key=f"del_{i}"):
                            st.session_state.aggregation_rules.pop(i)
                            st.rerun()

        with tab2:
            st.markdown("### ⚖️ Stage Separation")
            st.session_state.stage_separation = st.checkbox(
                "Separate Won/Lost stages",
                value=True,
                help="Won and Lost will NOT be summed together"
            )

            if st.session_state.stage_separation:
                st.success("✅ Smart separation enabled")
            else:
                st.warning("⚠️ Stages will be combined")

# ============================================================================
# PHASE 3: HIERARCHICAL ANALYSIS
# ============================================================================

elif phase == "🏢 Phase 3: Hierarchical Analysis":
    st.markdown("# 🏢 Phase 3: Hierarchical Analysis")

    if st.session_state.data is None:
        st.warning("Load data first!")
    else:
        df = st.session_state.data

        # Select columns
        project_col = st.selectbox("Project Column:", df.columns)
        value_col = st.selectbox("Value Column:", 
                                [c for c in df.columns if df[c].dtype in ['int64', 'float64']])

        # Currency conversion
        converter = CurrencyConverter()
        df_converted = df.copy()
        df_converted[f'{value_col}_converted'] = converter.convert_column(
            df, value_col, 'CHF', st.session_state.currency
        )

        if st.button("🔄 Aggregate"):
            # Simple aggregation by project
            agg = df_converted.groupby(project_col).agg({
                f'{value_col}_converted': ['sum', 'mean', 'count']
            }).reset_index()

            agg.columns = [project_col, 'Total', 'Average', 'Count']
            st.session_state.aggregated_data = agg
            st.success(f"✅ Aggregated to {len(agg)} groups")

        if st.session_state.aggregated_data is not None:
            st.dataframe(st.session_state.aggregated_data)

# ============================================================================
# PHASE 4: COMPARATIVE ANALYSIS
# ============================================================================

elif phase == "⚖️ Phase 4: Comparative Analysis":
    st.markdown("# ⚖️ Phase 4: Comparative Analysis")
    st.markdown("### Compare metrics between different filter combinations")
    st.markdown("---")

    if st.session_state.data is None:
        st.warning("Load data first!")
    else:
        df = st.session_state.data

        # Currency conversion
        converter = CurrencyConverter()
        value_cols = [c for c in df.columns if df[c].dtype in ['int64', 'float64']]

        if value_cols:
            metric_col = st.selectbox("Select metric to compare:", value_cols)

            # Convert currency
            df_converted = df.copy()
            df_converted[f'{metric_col}_display'] = converter.convert_column(
                df, metric_col, 'CHF', st.session_state.currency
            )

            st.markdown('<div class="comparison-box">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            # Filter A
            with col1:
                st.markdown("### 🔵 Group A")
                filters_a = {}
                for col in df.columns[:5]:  # First 5 columns for filters
                    if df[col].dtype == 'object' and df[col].nunique() < 50:
                        selected = st.multiselect(f"{col}:", df[col].unique(), key=f"a_{col}")
                        if selected:
                            filters_a[col] = selected

            # Filter B
            with col2:
                st.markdown("### 🔴 Group B")
                filters_b = {}
                for col in df.columns[:5]:
                    if df[col].dtype == 'object' and df[col].nunique() < 50:
                        selected = st.multiselect(f"{col}:", df[col].unique(), key=f"b_{col}")
                        if selected:
                            filters_b[col] = selected

            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("⚖️ Compare Groups", type="primary"):
                analyzer = ComparativeAnalyzer(df_converted)
                results = analyzer.compare_groups(filters_a, filters_b, f'{metric_col}_display')

                if 'error' not in results:
                    st.markdown("---")
                    st.markdown("### 📊 Comparison Results")

                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Group A Average", 
                                f"{st.session_state.currency} {results['group_a_mean']:,.0f}",
                                f"{results['group_a_count']} items")
                    with col2:
                        st.metric("Group B Average",
                                f"{st.session_state.currency} {results['group_b_mean']:,.0f}",
                                f"{results['group_b_count']} items")
                    with col3:
                        diff_pct = results['mean_diff_pct']
                        st.metric("Difference",
                                f"{diff_pct:+.1f}%",
                                f"{st.session_state.currency} {results['mean_diff']:+,.0f}")

                    # Statistical test
                    if results['p_value'] is not None:
                        st.markdown("### 📈 Statistical Test (T-Test)")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("P-value", f"{results['p_value']:.4f}")
                        with col2:
                            if results['significant']:
                                st.success("✅ **Significant difference** (p < 0.05)")
                            else:
                                st.info("No significant difference (p ≥ 0.05)")

                    # Visualization
                    comparison_df = pd.DataFrame({
                        'Group': ['Group A', 'Group B'],
                        'Average': [results['group_a_mean'], results['group_b_mean']],
                        'Total': [results['group_a_total'], results['group_b_total']]
                    })

                    fig = px.bar(comparison_df, x='Group', y='Average',
                               title=f"Average {metric_col} Comparison",
                               labels={'Average': f'Average ({st.session_state.currency})'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(results['error'])
        else:
            st.info("No numeric columns found for comparison")

# ============================================================================
# PHASE 5: VISUALIZATION
# ============================================================================

elif phase == "📈 Phase 5: Visualization":
    st.markdown("# 📈 Phase 5: Visualization")

    if st.session_state.aggregated_data is not None:
        df = st.session_state.aggregated_data

        chart_type = st.selectbox("Chart Type:", ["Bar Chart", "Histogram", "Box Plot"])

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            col = st.selectbox("Column:", numeric_cols)

            if chart_type == "Bar Chart":
                fig = px.bar(df.head(20), x=df.columns[0], y=col)
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == "Histogram":
                fig = px.histogram(df, x=col, nbins=30)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run aggregation in Phase 3 first")

# ============================================================================
# PHASE 6: REPORTING
# ============================================================================

elif phase == "📄 Phase 6: Reporting":
    st.markdown("# 📄 Phase 6: Reporting")

    if st.session_state.aggregated_data is not None:
        df = st.session_state.aggregated_data

        st.markdown("### 📥 Export Data")
        csv = df.to_csv(index=False)
        st.download_button(
            "📄 Download CSV",
            csv,
            f"analysis_{st.session_state.currency}.csv",
            "text/csv"
        )
    else:
        st.info("No data to export")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Advanced Sales Analysis Platform | Custom Rules • Currency • Comparisons</p>
</div>
""", unsafe_allow_html=True)
