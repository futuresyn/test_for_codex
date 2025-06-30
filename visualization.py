import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

class ChartGenerator:
    def __init__(self):
        self.default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        self.theme = 'plotly_white'
        
    def create_line_chart(self, df, x_col, y_col, color_col=None):
        """Create a line chart"""
        
        # Check if columns exist
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError("Specified columns not found in dataframe")
        
        if color_col and color_col not in df.columns:
            color_col = None
        
        # Create figure
        if color_col:
            fig = px.line(df, x=x_col, y=y_col, color=color_col,
                         title=f'{y_col} vs {x_col}',
                         template=self.theme)
        else:
            fig = px.line(df, x=x_col, y=y_col,
                         title=f'{y_col} vs {x_col}',
                         template=self.theme)
        
        # Update layout
        fig.update_layout(
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title(),
            showlegend=True,
            height=500
        )
        
        return fig
    
    def create_bar_chart(self, df, x_col, y_col, orientation='v'):
        """Create a bar chart"""
        
        # Aggregate data if needed
        if df[x_col].dtype == 'object':
            # Group by categorical column
            grouped_data = df.groupby(x_col)[y_col].agg(['mean', 'sum', 'count']).reset_index()
            
            # Use mean by default
            fig = px.bar(grouped_data, x=x_col, y='mean',
                        title=f'Average {y_col} by {x_col}',
                        template=self.theme)
        else:
            # For numerical x, bin the data
            df_copy = df.copy()
            df_copy[f'{x_col}_binned'] = pd.cut(df_copy[x_col], bins=20)
            grouped_data = df_copy.groupby(f'{x_col}_binned')[y_col].mean().reset_index()
            
            fig = px.bar(grouped_data, x=f'{x_col}_binned', y=y_col,
                        title=f'{y_col} by {x_col} (Binned)',
                        template=self.theme)
        
        # Update layout
        fig.update_layout(
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title(),
            showlegend=False,
            height=500
        )
        
        return fig
    
    def create_scatter_plot(self, df, x_col, y_col, color_col=None, size_col=None):
        """Create a scatter plot"""
        
        # Sample data if too large
        if len(df) > 5000:
            df_sample = df.sample(n=5000, random_state=42)
        else:
            df_sample = df
        
        # Create scatter plot
        fig = px.scatter(df_sample, x=x_col, y=y_col, 
                        color=color_col, size=size_col,
                        title=f'{y_col} vs {x_col}',
                        template=self.theme,
                        opacity=0.7)
        
        # Add trendline
        try:
            # Calculate correlation
            correlation = df_sample[x_col].corr(df_sample[y_col])
            
            # Add trendline manually
            z = np.polyfit(df_sample[x_col], df_sample[y_col], 1)
            p = np.poly1d(z)
            
            fig.add_trace(go.Scatter(x=df_sample[x_col], y=p(df_sample[x_col]),
                                   mode='lines', name=f'Trendline (r={correlation:.3f})',
                                   line=dict(color='red', dash='dash')))
        except:
            pass
        
        # Update layout
        fig.update_layout(
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title(),
            height=500
        )
        
        return fig
    
    def create_histogram(self, df, col, bins=30):
        """Create a histogram"""
        
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in dataframe")
        
        # Create histogram
        fig = px.histogram(df, x=col, nbins=bins,
                          title=f'Distribution of {col}',
                          template=self.theme)
        
        # Add statistics
        mean_val = df[col].mean()
        median_val = df[col].median()
        std_val = df[col].std()
        
        # Add vertical lines for mean and median
        fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {mean_val:.2f}")
        fig.add_vline(x=median_val, line_dash="dash", line_color="green",
                     annotation_text=f"Median: {median_val:.2f}")
        
        # Update layout
        fig.update_layout(
            xaxis_title=col.replace('_', ' ').title(),
            yaxis_title='Frequency',
            height=500,
            annotations=[
                dict(x=0.7, y=0.9, xref='paper', yref='paper',
                     text=f'Mean: {mean_val:.2f}<br>Median: {median_val:.2f}<br>Std: {std_val:.2f}',
                     showarrow=False, bgcolor='rgba(255,255,255,0.8)',
                     bordercolor='black', borderwidth=1)
            ]
        )
        
        return fig
    
    def create_correlation_heatmap(self, df):
        """Create a correlation heatmap"""
        
        # Select only numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) < 2:
            raise ValueError("Need at least 2 numerical columns for correlation heatmap")
        
        # Calculate correlation matrix
        corr_matrix = df[numerical_cols].corr()
        
        # Create heatmap
        fig = px.imshow(corr_matrix, 
                       x=corr_matrix.columns,
                       y=corr_matrix.columns,
                       color_continuous_scale='RdBu',
                       aspect='auto',
                       title='Correlation Heatmap',
                       text_auto=True)
        
        # Update layout
        fig.update_layout(
            height=600,
            xaxis_title='Features',
            yaxis_title='Features'
        )
        
        return fig
    
    def create_box_plot(self, df, x_col, y_col):
        """Create a box plot"""
        
        fig = px.box(df, x=x_col, y=y_col,
                    title=f'{y_col} Distribution by {x_col}',
                    template=self.theme)
        
        # Update layout
        fig.update_layout(
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title(),
            height=500
        )
        
        return fig
    
    def create_pie_chart(self, df, col, max_categories=10):
        """Create a pie chart"""
        
        if df[col].dtype not in ['object', 'category']:
            raise ValueError("Pie chart requires categorical data")
        
        # Count values
        value_counts = df[col].value_counts()
        
        # Limit categories
        if len(value_counts) > max_categories:
            top_categories = value_counts.head(max_categories-1)
            other_count = value_counts.tail(len(value_counts) - max_categories + 1).sum()
            
            # Create new series with 'Other' category
            plot_data = top_categories.copy()
            plot_data['Other'] = other_count
        else:
            plot_data = value_counts
        
        # Create pie chart
        fig = px.pie(values=plot_data.values, names=plot_data.index,
                    title=f'Distribution of {col}',
                    template=self.theme)
        
        # Update layout
        fig.update_layout(height=500)
        
        return fig
    
    def create_time_series(self, df, date_col, value_col, freq='D'):
        """Create a time series plot"""
        
        if date_col not in df.columns or value_col not in df.columns:
            raise ValueError("Specified columns not found")
        
        # Ensure date column is datetime
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        
        # Sort by date
        df_copy = df_copy.sort_values(date_col)
        
        # Resample if needed
        if freq != 'D':
            df_copy = df_copy.set_index(date_col).resample(freq)[value_col].mean().reset_index()
        
        # Create time series plot
        fig = px.line(df_copy, x=date_col, y=value_col,
                     title=f'{value_col} Over Time',
                     template=self.theme)
        
        # Add moving average
        window_size = min(30, len(df_copy) // 10)
        if window_size > 1:
            df_copy[f'{value_col}_ma'] = df_copy[value_col].rolling(window=window_size).mean()
            
            fig.add_trace(go.Scatter(x=df_copy[date_col], y=df_copy[f'{value_col}_ma'],
                                   mode='lines', name=f'{window_size}-period MA',
                                   line=dict(color='orange', width=2)))
        
        # Update layout
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title=value_col.replace('_', ' ').title(),
            height=500
        )
        
        return fig
    
    def create_multi_line_chart(self, df, x_col, y_cols):
        """Create a multi-line chart"""
        
        fig = go.Figure()
        
        colors = self.default_colors
        
        for i, y_col in enumerate(y_cols):
            if y_col in df.columns:
                color = colors[i % len(colors)]
                fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col],
                                       mode='lines+markers',
                                       name=y_col.replace('_', ' ').title(),
                                       line=dict(color=color)))
        
        # Update layout
        fig.update_layout(
            title='Multi-Line Chart',
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title='Values',
            template=self.theme,
            height=500,
            showlegend=True
        )
        
        return fig
    
    def create_dashboard_layout(self, df):
        """Create a comprehensive dashboard with multiple charts"""
        
        from plotly.subplots import make_subplots
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sales Distribution', 'Sales vs Revenue', 
                          'Monthly Trends', 'Regional Performance'),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Add charts
        if 'sales' in df.columns:
            fig.add_trace(go.Histogram(x=df['sales'], name='Sales'), row=1, col=1)
        
        if 'sales' in df.columns and 'revenue' in df.columns:
            fig.add_trace(go.Scatter(x=df['sales'], y=df['revenue'], 
                                   mode='markers', name='Sales vs Revenue'), row=1, col=2)
        
        if 'date' in df.columns and 'sales' in df.columns:
            monthly_data = df.groupby(df['date'].dt.month)['sales'].mean()
            fig.add_trace(go.Scatter(x=monthly_data.index, y=monthly_data.values,
                                   mode='lines+markers', name='Monthly Sales'), row=2, col=1)
        
        if 'region' in df.columns and 'sales' in df.columns:
            regional_data = df.groupby('region')['sales'].mean()
            fig.add_trace(go.Bar(x=regional_data.index, y=regional_data.values,
                               name='Regional Sales'), row=2, col=2)
        
        # Update layout
        fig.update_layout(height=800, showlegend=False, title_text="Sales Dashboard")
        
        return fig