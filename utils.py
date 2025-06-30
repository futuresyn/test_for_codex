import pandas as pd
import numpy as np
import json
import os
import pickle
from datetime import datetime, timedelta
import logging
import sys
import hashlib
import requests
from typing import List, Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_data_to_file(data, filename, format='csv'):
    """Save data to file in various formats"""
    
    try:
        if format.lower() == 'csv':
            data.to_csv(filename, index=False)
        elif format.lower() == 'json':
            data.to_json(filename, orient='records', indent=2)
        elif format.lower() == 'excel':
            data.to_excel(filename, index=False)
        elif format.lower() == 'pickle':
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Data saved successfully to {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving data to {filename}: {str(e)}")
        return False

def load_data_from_file(filename, format=None):
    """Load data from file"""
    
    if not os.path.exists(filename):
        logger.error(f"File {filename} does not exist")
        return None
    
    # Auto-detect format if not specified
    if format is None:
        _, ext = os.path.splitext(filename)
        format = ext[1:].lower()  # Remove the dot
    
    try:
        if format.lower() in ['csv', 'txt']:
            data = pd.read_csv(filename)
        elif format.lower() == 'json':
            data = pd.read_json(filename)
        elif format.lower() in ['xlsx', 'xls', 'excel']:
            data = pd.read_excel(filename)
        elif format.lower() == 'pickle':
            with open(filename, 'rb') as f:
                data = pickle.load(f)
        else:
            logger.warning(f"Unknown format {format}, trying CSV")
            data = pd.read_csv(filename)
        
        logger.info(f"Data loaded successfully from {filename}")
        return data
        
    except Exception as e:
        logger.error(f"Error loading data from {filename}: {str(e)}")
        return None

def validate_dataframe(df):
    """Validate dataframe structure and content"""
    
    issues = []
    
    # Check if empty
    if df.empty:
        issues.append("DataFrame is empty")
        return issues
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    if total_missing > 0:
        issues.append(f"Found {total_missing} missing values across {len(missing_counts[missing_counts > 0])} columns")
    
    # Check for duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        issues.append(f"Found {duplicate_count} duplicate rows")
    
    # Check for columns with single unique value
    single_value_cols = []
    for col in df.columns:
        if df[col].nunique() == 1:
            single_value_cols.append(col)
    
    if single_value_cols:
        issues.append(f"Columns with single unique value: {single_value_cols}")
    
    # Check for extremely high cardinality in categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    high_cardinality_cols = []
    for col in categorical_cols:
        if df[col].nunique() > len(df) * 0.8:  # More than 80% unique values
            high_cardinality_cols.append(col)
    
    if high_cardinality_cols:
        issues.append(f"High cardinality categorical columns: {high_cardinality_cols}")
    
    return issues

def calculate_data_hash(df):
    """Calculate hash of dataframe for change detection"""
    
    # Convert dataframe to string and calculate hash
    df_string = df.to_string()
    hash_object = hashlib.md5(df_string.encode())
    return hash_object.hexdigest()

def memory_usage_analysis(df):
    """Analyze memory usage of dataframe"""
    
    memory_info = {}
    
    # Total memory usage
    total_memory = df.memory_usage(deep=True).sum()
    memory_info['total_mb'] = total_memory / (1024 * 1024)
    
    # Memory usage by column
    column_memory = df.memory_usage(deep=True)
    memory_info['by_column'] = {
        col: mem / (1024 * 1024) for col, mem in column_memory.items()
    }
    
    # Memory usage by data type
    dtype_memory = df.groupby(df.dtypes).apply(lambda x: x.memory_usage(deep=True).sum())
    memory_info['by_dtype'] = {
        str(dtype): mem / (1024 * 1024) for dtype, mem in dtype_memory.items()
    }
    
    return memory_info

def optimize_dataframe_memory(df):
    """Optimize dataframe memory usage"""
    
    optimized_df = df.copy()
    
    for col in optimized_df.columns:
        col_type = optimized_df[col].dtype
        
        if col_type != 'object':
            c_min = optimized_df[col].min()
            c_max = optimized_df[col].max()
            
            if str(col_type)[:3] == 'int':
                # Integer optimization
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    optimized_df[col] = optimized_df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    optimized_df[col] = optimized_df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    optimized_df[col] = optimized_df[col].astype(np.int32)
            
            elif str(col_type)[:5] == 'float':
                # Float optimization
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    optimized_df[col] = optimized_df[col].astype(np.float32)
        
        else:
            # Object/string optimization
            num_unique_values = len(optimized_df[col].unique())
            num_total_values = len(optimized_df[col])
            if num_unique_values / num_total_values < 0.5:
                optimized_df[col] = optimized_df[col].astype('category')
    
    return optimized_df

def generate_data_profile(df):
    """Generate comprehensive data profile"""
    
    profile = {
        'basic_info': {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
        },
        'missing_data': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'numerical_summary': {},
        'categorical_summary': {}
    }
    
    # Numerical columns analysis
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        profile['numerical_summary'][col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'q25': df[col].quantile(0.25),
            'q75': df[col].quantile(0.75),
            'skewness': df[col].skew(),
            'kurtosis': df[col].kurtosis(),
            'zeros': (df[col] == 0).sum(),
            'negative': (df[col] < 0).sum()
        }
    
    # Categorical columns analysis
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        profile['categorical_summary'][col] = {
            'unique_count': df[col].nunique(),
            'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
            'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
            'top_5_values': value_counts.head().to_dict()
        }
    
    return profile

def export_data_report(df, filename='data_report.html'):
    """Export comprehensive data report as HTML"""
    
    profile = generate_data_profile(df)
    validation_issues = validate_dataframe(df)
    memory_info = memory_usage_analysis(df)
    
    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .section {{ margin-bottom: 30px; }}
            .header {{ color: #333; border-bottom: 2px solid #007acc; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metric {{ background-color: #f9f9f9; padding: 10px; margin: 5px 0; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1 class="header">Data Analysis Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="section">
            <h2>Basic Information</h2>
            <div class="metric">Dataset Shape: {profile['basic_info']['shape']}</div>
            <div class="metric">Memory Usage: {profile['basic_info']['memory_usage_mb']:.2f} MB</div>
            <div class="metric">Total Columns: {len(profile['basic_info']['columns'])}</div>
        </div>
        
        <div class="section">
            <h2>Data Quality Issues</h2>
            {'<ul>' + ''.join([f'<li>{issue}</li>' for issue in validation_issues]) + '</ul>' if validation_issues else '<p>No major data quality issues found.</p>'}
        </div>
        
        <div class="section">
            <h2>Missing Data Summary</h2>
            <table>
                <tr><th>Column</th><th>Missing Count</th></tr>
                {''.join([f'<tr><td>{col}</td><td>{count}</td></tr>' for col, count in profile['missing_data'].items() if count > 0])}
            </table>
        </div>
    </body>
    </html>
    """
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"Report exported to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error exporting report: {str(e)}")
        return False

def fetch_external_data(url, format='json'):
    """Fetch data from external URL"""
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        if format.lower() == 'json':
            data = response.json()
            return pd.DataFrame(data)
        elif format.lower() == 'csv':
            from io import StringIO
            data = pd.read_csv(StringIO(response.text))
            return data
        else:
            logger.warning(f"Unsupported format {format}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data from {url}: {str(e)}")
        return None