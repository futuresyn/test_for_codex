import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from typing import Optional
import warnings

warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self, scaler: Optional[StandardScaler] = None,
                 imputer: Optional[SimpleImputer] = None):
        """Initialize the processor with optional scaler and imputer."""
        self.scaler = scaler or StandardScaler()
        self.imputer = imputer or SimpleImputer(strategy='mean')
        self.processed_columns = []
    
    def process_dataset(self, df, clean_missing=True, normalize=False, remove_outliers=False):
        """Process the entire dataset with various options"""
        
        # Make a copy to avoid modifying original
        processed_df = df.copy()
        
        # Clean missing values
        if clean_missing:
            processed_df = self.clean_missing_values(processed_df)
        
        # Remove outliers
        if remove_outliers:
            processed_df = self.remove_outliers(processed_df)
        
        # Normalize data
        if normalize:
            processed_df = self.normalize_data(processed_df)
        
        # Additional processing
        processed_df = self.additional_processing(processed_df)
        
        return processed_df
    
    def clean_missing_values(self, df):
        """Clean missing values from dataset"""
        
        # Handle numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                # Fill with mean
                mean_value = df[col].mean()
                df[col].fillna(mean_value, inplace=True)
        
        # Handle categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                # Fill with mode
                mode_value = df[col].mode()[0]
                df[col].fillna(mode_value, inplace=True)
        
        # Handle datetime columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        for col in datetime_cols:
            if df[col].isnull().sum() > 0:
                # Forward fill for datetime
                df[col].ffill(inplace=True)
        
        return df
    
    def remove_outliers(self, df):
        """Remove outliers using IQR method"""
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Calculate bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Remove outliers
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        return df
    
    def normalize_data(self, df):
        """Normalize numerical data"""
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            # Get column data
            col_data = df[col].values.reshape(-1, 1)
            
            # Fit and transform
            self.scaler.fit(col_data)
            normalized_data = self.scaler.transform(col_data)
            
            # Replace original data
            df[col] = normalized_data.flatten()
        
        return df
    
    def additional_processing(self, df):
        """Additional data processing and feature engineering"""
        
        # Create derived features
        if 'sales' in df.columns and 'customers' in df.columns:
            df['sales_per_customer'] = df['sales'] / df['customers']
            
            # Handle division by zero
            df['sales_per_customer'] = df['sales_per_customer'].replace([np.inf, -np.inf], 0)
        
        if 'revenue' in df.columns and 'sales' in df.columns:
            df['revenue_per_sale'] = df['revenue'] / df['sales']
            df['revenue_per_sale'] = df['revenue_per_sale'].replace([np.inf, -np.inf], 0)
        
        # Date processing
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'])
                df['year'] = df['date'].dt.year
                df['month'] = df['date'].dt.month
                df['day_of_week'] = df['date'].dt.dayofweek
                df['quarter'] = df['date'].dt.quarter
            except:
                pass
        
        # Categorical encoding
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'date':  # Skip date column
                # One-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)
        
        return df
    
    def calculate_statistics(self, df):
        """Calculate basic statistics for the dataset"""
        
        stats = {}
        
        # Numerical statistics
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            stats[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis()
            }
        
        return stats
