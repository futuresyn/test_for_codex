import streamlit as st
import pandas as pd
import numpy as np
from data_processor import DataProcessor
from visualization import ChartGenerator
from utils import save_data_to_file, load_data_from_file
import config
import time
import os

st.set_page_config(page_title="Data Analysis Dashboard", layout="wide")

def main():
    st.title("📊 Advanced Data Analysis Dashboard")
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    
    # Sidebar
    st.sidebar.header("Data Input")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file", 
        type=['csv']
    )
    
    # Sample data generation
    if st.sidebar.button("Generate Sample Data"):
        data = generate_sample_data()
        st.session_state.data = data
        st.success("Sample data generated!")
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data
            st.success(f"Data loaded successfully! Shape: {data.shape}")
        except Exception as e:
            st.error(f"Error loading file: {e}")
    
    if st.session_state.data is not None:
        display_main_content()
    else:
        st.info("Please upload a CSV file or generate sample data to begin.")

def generate_sample_data():
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'date': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
        'sales': np.random.normal(1000, 200, n_samples),
        'customers': np.random.poisson(50, n_samples),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], n_samples),
        'revenue': np.random.normal(5000, 1000, n_samples)
    }
    
    df = pd.DataFrame(data)
    # Introduce some correlations
    df['revenue'] = df['sales'] * np.random.uniform(3, 7, n_samples) + np.random.normal(0, 500, n_samples)
    df['customers'] = (df['sales'] / 20) + np.random.normal(0, 10, n_samples)
    
    return df

def display_main_content():
    data = st.session_state.data
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Data Overview")
        st.dataframe(data.head(10))
        
        st.header("Basic Statistics")
        st.dataframe(data.describe())
    
    with col2:
        st.header("Data Info")
        st.write(f"**Shape:** {data.shape}")
        st.write(f"**Columns:** {list(data.columns)}")
        st.write(f"**Memory Usage:** {data.memory_usage().sum() / 1024:.2f} KB")
        
        # Missing values
        missing = data.isnull().sum()
        if missing.sum() > 0:
            st.write("**Missing Values:**")
            for col, count in missing.items():
                if count > 0:
                    st.write(f"- {col}: {count}")
        else:
            st.write("**No missing values found**")
    
    # Data processing section
    st.header("Data Processing")
    
    processor = DataProcessor()
    
    # Processing options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        clean_data = st.checkbox("Clean Missing Values")
    with col2:
        normalize_data = st.checkbox("Normalize Numerical Data")
    with col3:
        filter_outliers = st.checkbox("Filter Outliers")
    
    if st.button("Process Data"):
        with st.spinner("Processing data..."):
            processed_data = processor.process_dataset(
                data.copy(), 
                clean_missing=clean_data,
                normalize=normalize_data,
                remove_outliers=filter_outliers
            )
            st.session_state.processed_data = processed_data
            st.success("Data processed successfully!")
    
    # Visualization section
    if st.session_state.processed_data is not None:
        st.header("Data Visualization")
        
        chart_gen = ChartGenerator()
        processed_data = st.session_state.processed_data
        
        # Chart type selection
        chart_type = st.selectbox(
            "Select Chart Type",
            ["Line Chart", "Bar Chart", "Scatter Plot", "Histogram", "Correlation Heatmap"]
        )
        
        # Column selection for charts
        if chart_type in ["Line Chart", "Bar Chart", "Scatter Plot"]:
            col1, col2 = st.columns(2)
            with col1:
                x_column = st.selectbox("X-axis", processed_data.columns)
            with col2:
                y_column = st.selectbox("Y-axis", processed_data.select_dtypes(include=[np.number]).columns)
        
        if st.button("Generate Chart"):
            try:
                if chart_type == "Line Chart":
                    fig = chart_gen.create_line_chart(processed_data, x_column, y_column)
                elif chart_type == "Bar Chart":
                    fig = chart_gen.create_bar_chart(processed_data, x_column, y_column)
                elif chart_type == "Scatter Plot":
                    fig = chart_gen.create_scatter_plot(processed_data, x_column, y_column)
                elif chart_type == "Histogram":
                    numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
                    fig = chart_gen.create_histogram(processed_data, numeric_cols[0])
                elif chart_type == "Correlation Heatmap":
                    fig = chart_gen.create_correlation_heatmap(processed_data)
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error generating chart: {e}")
    
    # Export section
    st.header("Export Data")
    
    if st.session_state.processed_data is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Save Processed Data"):
                filename = f"processed_data_{int(time.time())}.csv"
                save_data_to_file(st.session_state.processed_data, filename)
                st.success(f"Data saved as {filename}")
        
        with col2:
            csv = st.session_state.processed_data.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name="processed_data.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()