import pandas as pd
import numpy as np
import json
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self):
        pass
    
    def analyze_dataset(self, filepath):
        """Perform comprehensive analysis on the dataset"""
        
        try:
            df = pd.read_csv(filepath)
            
            analysis = {
                'basic_info': self._get_basic_info(df),
                'statistical_summary': self._get_statistical_summary(df),
                'missing_values': self._get_missing_values(df),
                'data_types': self._get_data_types(df),
                'correlation_analysis': self._get_correlation_analysis(df),
                'data_quality': self._calculate_data_quality(df),
                'outlier_analysis': self._detect_outliers(df)
            }
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing dataset: {str(e)}")
            raise e
    
    def comprehensive_analysis(self, filepath):
        """Perform advanced comprehensive data analysis"""
        
        df = pd.read_csv(filepath)
        
        analysis = self.analyze_dataset(filepath)
        
        # Add advanced analysis
        analysis.update({
            'distribution_analysis': self._analyze_distributions(df),
            'pairwise_correlations': self._get_pairwise_correlations(df),
            'multivariate_analysis': self._multivariate_analysis(df),
            'time_series_analysis': self._time_series_analysis(df),
            'data_patterns': self._detect_data_patterns(df)
        })
        
        # Generate visualization data
        analysis['visualizations'] = self._generate_visualizations(df)
        
        return analysis
    
    def _get_basic_info(self, df):
        """Get basic information about the dataset"""
        
        basic_info = {
            'shape': {
                'rows': df.shape[0],
                'columns': df.shape[1]
            },
            'columns': df.columns.tolist(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 ** 2),
            'duplicate_rows': df.duplicated().sum(),
            'total_cells': df.shape[0] * df.shape[1],
            'size_category': self._get_size_category(df)
        }
        
        return basic_info
    
    def _get_statistical_summary(self, df):
        """Get comprehensive statistical summary"""
        
        numerical_df = df.select_dtypes(include=[np.number])
        categorical_df = df.select_dtypes(include=['object', 'category'])
        
        summary = {
            'numerical': {},
            'categorical': {}
        }
        
        # Numerical columns summary
        if not numerical_df.empty:
            numerical_summary = numerical_df.describe().to_dict()
            
            # Add additional statistics
            for col in numerical_df.columns:
                numerical_summary[col].update({
                    'variance': numerical_df[col].var(),
                    'skewness': numerical_df[col].skew(),
                    'kurtosis': numerical_df[col].kurtosis(),
                    'median': numerical_df[col].median(),
                    'mode': numerical_df[col].mode().iloc[0] if not numerical_df[col].mode().empty else None,
                    'range': numerical_df[col].max() - numerical_df[col].min(),
                    'coefficient_of_variation': numerical_df[col].std() / numerical_df[col].mean() if numerical_df[col].mean() != 0 else 0,
                    'iqr': numerical_df[col].quantile(0.75) - numerical_df[col].quantile(0.25)
                })
            
            summary['numerical'] = numerical_summary
        
        # Categorical columns summary
        if not categorical_df.empty:
            for col in categorical_df.columns:
                value_counts = categorical_df[col].value_counts()
                summary['categorical'][col] = {
                    'unique_count': categorical_df[col].nunique(),
                    'most_frequent': value_counts.index[0] if not value_counts.empty else None,
                    'most_frequent_count': value_counts.iloc[0] if not value_counts.empty else 0,
                    'least_frequent': value_counts.index[-1] if not value_counts.empty else None,
                    'least_frequent_count': value_counts.iloc[-1] if not value_counts.empty else 0,
                    'entropy': self._calculate_entropy(value_counts),
                    'top_categories': value_counts.head(10).to_dict()
                }
        
        return summary
    
    def _get_missing_values(self, df):
        """Comprehensive missing values analysis"""
        
        missing_count = df.isnull().sum()
        missing_percentage = (missing_count / len(df)) * 100
        
        # Pattern analysis
        missing_patterns = self._analyze_missing_patterns(df)
        
        return {
            'count_by_column': missing_count.to_dict(),
            'percentage_by_column': missing_percentage.to_dict(),
            'total_missing': missing_count.sum(),
            'total_cells': df.shape[0] * df.shape[1],
            'completeness_score': ((df.shape[0] * df.shape[1] - missing_count.sum()) / (df.shape[0] * df.shape[1])) * 100,
            'columns_with_missing': [col for col in df.columns if missing_count[col] > 0],
            'missing_patterns': missing_patterns,
            'severity_assessment': self._assess_missing_severity(missing_percentage)
        }
    
    def _get_data_types(self, df):
        """Detailed data types analysis"""
        
        dtype_counts = df.dtypes.value_counts().to_dict()
        dtype_details = {}
        
        for col in df.columns:
            col_info = {
                'dtype': str(df[col].dtype),
                'unique_values': df[col].nunique(),
                'memory_usage_kb': df[col].memory_usage(deep=True) / 1024,
                'sample_values': df[col].dropna().head(5).tolist() if df[col].dtype == 'object' else None
            }
            
            # Add specific analysis based on data type
            if np.issubdtype(df[col].dtype, np.number):
                col_info.update({
                    'type': 'numerical',
                    'has_negative': (df[col] < 0).any(),
                    'has_zero': (df[col] == 0).any(),
                    'is_integer': all(df[col].dropna().apply(lambda x: x == int(x)) if not df[col].dropna().empty else False)
                })
            else:
                col_info.update({
                    'type': 'categorical',
                    'max_length': df[col].astype(str).str.len().max(),
                    'min_length': df[col].astype(str).str.len().min()
                })
            
            dtype_details[col] = col_info
        
        return {
            'type_counts': dtype_counts,
            'column_details': dtype_details,
            'type_recommendations': self._get_type_recommendations(df)
        }
    
    def _get_correlation_analysis(self, df):
        """Comprehensive correlation analysis"""
        
        numerical_df = df.select_dtypes(include=[np.number])
        
        if numerical_df.empty or len(numerical_df.columns) < 2:
            return {
                'matrix': {},
                'high_correlations': [],
                'correlation_heatmap': None
            }
        
        # Calculate correlation matrix
        correlation_matrix = numerical_df.corr().round(3)
        
        # Find highly correlated pairs
        high_corr_pairs = []
        moderate_corr_pairs = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if not np.isnan(corr_value):
                    pair_info = {
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_value,
                        'strength': self._get_correlation_strength(abs(corr_value))
                    }
                    
                    if abs(corr_value) > 0.7:
                        high_corr_pairs.append(pair_info)
                    elif abs(corr_value) > 0.5:
                        moderate_corr_pairs.append(pair_info)
        
        # Sort by absolute correlation
        high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        moderate_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return {
            'matrix': correlation_matrix.to_dict(),
            'high_correlations': high_corr_pairs[:20],  # Top 20
            'moderate_correlations': moderate_corr_pairs[:20],
            'most_correlated_features': self._get_most_correlated_features(correlation_matrix),
            'correlation_heatmap': self._generate_correlation_heatmap(numerical_df)
        }
    
    def _calculate_data_quality(self, df):
        """Calculate comprehensive data quality score"""
        
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        duplicate_rows = df.duplicated().sum()
        
        # Completeness score
        completeness_score = ((total_cells - missing_cells) / total_cells) * 100
        
        # Uniqueness score
        uniqueness_score = ((df.shape[0] - duplicate_rows) / df.shape[0]) * 100
        
        # Consistency score
        consistency_score = self._calculate_consistency_score(df)
        
        # Validity score
        validity_score = self._calculate_validity_score(df)
        
        # Accuracy score (estimated)
        accuracy_score = self._estimate_accuracy_score(df)
        
        # Overall quality score (weighted average)
        overall_quality = (
            completeness_score * 0.3 +
            uniqueness_score * 0.2 +
            consistency_score * 0.25 +
            validity_score * 0.15 +
            accuracy_score * 0.1
        )
        
        return {
            'completeness': round(completeness_score, 2),
            'uniqueness': round(uniqueness_score, 2),
            'consistency': round(consistency_score, 2),
            'validity': round(validity_score, 2),
            'accuracy': round(accuracy_score, 2),
            'overall': round(overall_quality, 2),
            'quality_grade': self._get_quality_grade(overall_quality)
        }
    
    def _detect_outliers(self, df):
        """Comprehensive outlier detection using multiple methods"""
        
        numerical_df = df.select_dtypes(include=[np.number])
        outlier_analysis = {}
        
        for col in numerical_df.columns:
            col_data = numerical_df[col].dropna()
            
            if len(col_data) < 3:
                continue
            
            # IQR method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound_iqr = Q1 - 1.5 * IQR
            upper_bound_iqr = Q3 + 1.5 * IQR
            
            outliers_iqr = col_data[(col_data < lower_bound_iqr) | (col_data > upper_bound_iqr)]
            
            # Z-score method
            z_scores = np.abs(stats.zscore(col_data))
            outliers_zscore = col_data[z_scores > 3]
            
            # Modified Z-score method (more robust)
            median = np.median(col_data)
            mad = np.median(np.abs(col_data - median))
            modified_z_scores = 0.6745 * (col_data - median) / mad if mad != 0 else 0
            outliers_modified_z = col_data[np.abs(modified_z_scores) > 3.5]
            
            outlier_analysis[col] = {
                'iqr_method': {
                    'outlier_count': len(outliers_iqr),
                    'outlier_percentage': (len(outliers_iqr) / len(col_data)) * 100,
                    'lower_bound': lower_bound_iqr,
                    'upper_bound': upper_bound_iqr
                },
                'zscore_method': {
                    'outlier_count': len(outliers_zscore),
                    'outlier_percentage': (len(outliers_zscore) / len(col_data)) * 100
                },
                'modified_zscore_method': {
                    'outlier_count': len(outliers_modified_z),
                    'outlier_percentage': (len(outliers_modified_z) / len(col_data)) * 100
                },
                'consensus_outliers': len(set(outliers_iqr.index) & set(outliers_zscore.index) & set(outliers_modified_z.index)),
                'outlier_indices': outliers_iqr.index.tolist()
            }
        
        return outlier_analysis
    
    def _analyze_distributions(self, df):
        """Analyze distributions of numerical columns"""
        
        numerical_df = df.select_dtypes(include=[np.number])
        distribution_analysis = {}
        
        for col in numerical_df.columns:
            col_data = numerical_df[col].dropna()
            
            if len(col_data) < 3:
                continue
            
            # Normality tests
            shapiro_test = stats.shapiro(col_data) if len(col_data) < 5000 else (0, 0)
            normality_pvalue = shapiro_test[1] if len(col_data) < 5000 else None
            
            distribution_analysis[col] = {
                'is_normal': normality_pvalue > 0.05 if normality_pvalue else False,
                'skewness': col_data.skew(),
                'kurtosis': col_data.kurtosis(),
                'modality': self._detect_modality(col_data),
                'distribution_type': self._classify_distribution(col_data),
                'normality_pvalue': normality_pvalue,
                'has_heavy_tails': abs(col_data.kurtosis()) > 3
            }
        
        return distribution_analysis
    
    def _get_pairwise_correlations(self, df):
        """Get detailed pairwise correlations with statistical significance"""
        
        numerical_df = df.select_dtypes(include=[np.number])
        
        if numerical_df.empty or len(numerical_df.columns) < 2:
            return []
        
        correlations = []
        columns = numerical_df.columns
        
        for i in range(len(columns)):
            for j in range(i+1, len(columns)):
                col1 = numerical_df[columns[i]]
                col2 = numerical_df[columns[j]]
                
                # Remove pairs with missing values
                valid_data = pd.concat([col1, col2], axis=1).dropna()
                if len(valid_data) < 3:
                    continue
                
                corr = valid_data[columns[i]].corr(valid_data[columns[j]])
                
                # Calculate p-value for correlation
                if not np.isnan(corr) and abs(corr) < 1.0:
                    n = len(valid_data)
                    t_stat = corr * np.sqrt((n - 2) / (1 - corr**2))
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                else:
                    p_value = None
                
                correlations.append({
                    'feature1': columns[i],
                    'feature2': columns[j],
                    'correlation': round(corr, 3),
                    'strength': self._get_correlation_strength(abs(corr)),
                    'p_value': p_value,
                    'significant': p_value < 0.05 if p_value else False,
                    'sample_size': len(valid_data)
                })
        
        # Sort by absolute correlation value
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return correlations[:50]  # Return top 50 correlations
    
    def _multivariate_analysis(self, df):
        """Perform multivariate analysis"""
        
        numerical_df = df.select_dtypes(include=[np.number])
        
        if numerical_df.empty or len(numerical_df.columns) < 2:
            return {}
        
        # Principal Component Analysis (simplified)
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numerical_df.dropna())
            
            # Perform PCA
            pca = PCA()
            pca_result = pca.fit_transform(scaled_data)
            
            pca_analysis = {
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
                'components_to_explain_95': len([x for x in np.cumsum(pca.explained_variance_ratio_) if x < 0.95]) + 1,
                'total_variance_explained': sum(pca.explained_variance_ratio_)
            }
        except:
            pca_analysis = {}
        
        return {
            'pca_analysis': pca_analysis,
            'multicollinearity_assessment': self._assess_multicollinearity(numerical_df)
        }
    
    def _time_series_analysis(self, df):
        """Analyze time series patterns if date columns exist"""
        
        date_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col], errors='raise')
                    date_columns.append(col)
                except:
                    pass
        
        if not date_columns:
            return {'has_temporal_data': False}
        
        time_analysis = {'has_temporal_data': True, 'date_columns': date_columns}
        
        # Basic time series analysis for the first date column
        date_col = date_columns[0]
        try:
            df_temp = df.copy()
            df_temp[date_col] = pd.to_datetime(df_temp[date_col])
            time_analysis['date_range'] = {
                'start': df_temp[date_col].min().strftime('%Y-%m-%d'),
                'end': df_temp[date_col].max().strftime('%Y-%m-%d'),
                'days_span': (df_temp[date_col].max() - df_temp[date_col].min()).days
            }
        except:
            pass
        
        return time_analysis
    
    def _detect_data_patterns(self, df):
        """Detect patterns and anomalies in data"""
        
        patterns = {
            'constant_columns': [],
            'highly_skewed_columns': [],
            'high_cardinality_columns': [],
            'suspicious_patterns': []
        }
        
        # Detect constant columns
        for col in df.columns:
            if df[col].nunique() == 1:
                patterns['constant_columns'].append(col)
        
        # Detect highly skewed numerical columns
        numerical_df = df.select_dtypes(include=[np.number])
        for col in numerical_df.columns:
            if abs(numerical_df[col].skew()) > 5:
                patterns['highly_skewed_columns'].append(col)
        
        # Detect high cardinality categorical columns
        categorical_df = df.select_dtypes(include=['object'])
        for col in categorical_df.columns:
            if categorical_df[col].nunique() > 100:
                patterns['high_cardinality_columns'].append({
                    'column': col,
                    'unique_values': categorical_df[col].nunique()
                })
        
        return patterns
    
    def _generate_visualizations(self, df):
        """Generate visualization data for frontend"""
        
        visualizations = {}
        
        # Numerical columns distributions
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numerical_cols:
            visualizations['numerical_columns'] = numerical_cols
        
        # Categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            visualizations['categorical_columns'] = categorical_cols
        
        return visualizations
    
    # Helper methods
    def _get_size_category(self, df):
        rows, cols = df.shape
        if rows < 1000:
            return 'Small'
        elif rows < 10000:
            return 'Medium'
        elif rows < 100000:
            return 'Large'
        else:
            return 'Very Large'
    
    def _calculate_entropy(self, value_counts):
        """Calculate entropy of a categorical distribution"""
        probabilities = value_counts / value_counts.sum()
        return -np.sum(probabilities * np.log2(probabilities))
    
    def _analyze_missing_patterns(self, df):
        """Analyze patterns in missing data"""
        missing_matrix = df.isnull()
        return {
            'rows_with_missing': missing_matrix.any(axis=1).sum(),
            'columns_with_missing': missing_matrix.any(axis=0).sum(),
            'complete_cases': (~missing_matrix.any(axis=1)).sum()
        }
    
    def _assess_missing_severity(self, missing_percentage):
        severity = {}
        for col, percentage in missing_percentage.items():
            if percentage == 0:
                severity[col] = 'None'
            elif percentage < 5:
                severity[col] = 'Low'
            elif percentage < 20:
                severity[col] = 'Moderate'
            elif percentage < 50:
                severity[col] = 'High'
            else:
                severity[col] = 'Critical'
        return severity
    
    def _get_type_recommendations(self, df):
        recommendations = []
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Check if it's actually a date
                    pd.to_datetime(df[col], errors='raise')
                    recommendations.append(f"Column '{col}' appears to be a date. Consider converting to datetime.")
                except:
                    # Check if it's actually numerical
                    try:
                        pd.to_numeric(df[col], errors='raise')
                        recommendations.append(f"Column '{col}' contains numerical data stored as text. Consider converting to numeric.")
                    except:
                        pass
        return recommendations
    
    def _get_correlation_strength(self, corr_value):
        if corr_value >= 0.7:
            return 'Strong'
        elif corr_value >= 0.5:
            return 'Moderate'
        elif corr_value >= 0.3:
            return 'Weak'
        else:
            return 'Very Weak'
    
    def _get_most_correlated_features(self, correlation_matrix):
        """Get features with highest average correlation"""
        avg_correlations = correlation_matrix.abs().mean().sort_values(ascending=False)
        return avg_correlations.head(10).to_dict()
    
    def _generate_correlation_heatmap(self, numerical_df):
        """Generate correlation heatmap as base64 image"""
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', center=0)
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            
            plt.close()
            return base64.b64encode(image_png).decode()
        except:
            return None
    
    def _calculate_consistency_score(self, df):
        """Calculate data consistency score"""
        score = 100
        numerical_df = df.select_dtypes(include=[np.number])
        
        # Check for negative values where inappropriate
        for col in numerical_df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['age', 'price', 'salary', 'amount', 'cost']):
                if (numerical_df[col] < 0).any():
                    score -= 5
        
        # Check for unrealistic values
        for col in numerical_df.columns:
            col_lower = col.lower()
            if 'age' in col_lower:
                if (numerical_df[col] > 150).any():
                    score -= 10
            elif 'price' in col_lower or 'salary' in col_lower:
                if (numerical_df[col] > 1e9).any():  # Unrealistically high values
                    score -= 10
        
        return max(score, 0)
    
    def _calculate_validity_score(self, df):
        """Calculate data validity score"""
        score = 100
        
        # Check for obvious data entry errors
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for mixed case inconsistencies
                unique_values = df[col].dropna().unique()
                if len(unique_values) > 1:
                    lower_values = [str(x).lower() for x in unique_values]
                    if len(set(lower_values)) < len(unique_values):
                        score -= 5
        
        return max(score, 0)
    
    def _estimate_accuracy_score(self, df):
        """Estimate data accuracy (simplified)"""
        # This is a simplified estimation - in real scenarios, you'd need domain knowledge
        score = 85  # Base score
        
        # Deduct points for common data quality issues
        if df.isnull().sum().sum() > 0:
            score -= 5
        
        numerical_df = df.select_dtypes(include=[np.number])
        for col in numerical_df.columns:
            if numerical_df[col].skew() > 10:  # Extremely skewed data
                score -= 2
        
        return max(score, 0)
    
    def _get_quality_grade(self, score):
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def _detect_modality(self, series):
        """Detect modality of distribution"""
        if len(series) < 3:
            return 'Unknown'
        
        # Simple modality detection
        value_counts = series.value_counts()
        if len(value_counts) < 2:
            return 'Unimodal'
        
        sorted_counts = value_counts.sort_values(ascending=False)
        if len(sorted_counts) > 1 and sorted_counts.iloc[1] > sorted_counts.iloc[0] * 0.8:
            return 'Bimodal'
        elif len(sorted_counts) > 2 and sorted_counts.iloc[2] > sorted_counts.iloc[0] * 0.6:
            return 'Multimodal'
        else:
            return 'Unimodal'
    
    def _classify_distribution(self, series):
        """Classify distribution type"""
        skewness = series.skew()
        kurt = series.kurtosis()
        
        if abs(skewness) < 0.5 and abs(kurt) < 1:
            return 'Normal'
        elif skewness > 1:
            return 'Right Skewed'
        elif skewness < -1:
            return 'Left Skewed'
        elif kurt > 3:
            return 'Heavy-tailed'
        elif kurt < -1:
            return 'Light-tailed'
        else:
            return 'Unknown'
    
    def _assess_multicollinearity(self, numerical_df):
        """Assess multicollinearity using VIF (simplified)"""
        if numerical_df.empty or len(numerical_df.columns) < 2:
            return {}
        
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            from statsmodels.tools.tools import add_constant
            
            # Remove constant columns
            numerical_df = numerical_df.loc[:, numerical_df.std() != 0]
            
            if len(numerical_df.columns) < 2:
                return {}
            
            # Calculate VIF
            X = add_constant(numerical_df)
            vif_data = {}
            for i, col in enumerate(X.columns):
                if col != 'const':
                    vif = variance_inflation_factor(X.values, i)
                    vif_data[col] = vif
            
            # Classify multicollinearity
            multicollinearity_assessment = {}
            for col, vif in vif_data.items():
                if vif > 10:
                    multicollinearity_assessment[col] = 'High'
                elif vif > 5:
                    multicollinearity_assessment[col] = 'Moderate'
                else:
                    multicollinearity_assessment[col] = 'Low'
            
            return {
                'vif_scores': vif_data,
                'assessment': multicollinearity_assessment,
                'high_vif_features': [col for col, vif in vif_data.items() if vif > 10]
            }
        except:
            return {}

# Utility function for external use
def create_sample_dataset():
    """Create a sample dataset for testing"""
    np.random.seed(42)
    
    data = {
        'age': np.random.normal(35, 10, 1000),
        'income': np.random.lognormal(10, 1, 1000),
        'education_years': np.random.randint(8, 20, 1000),
        'city': np.random.choice(['New York', 'London', 'Tokyo', 'Paris'], 1000),
        'satisfaction_score': np.random.randint(1, 6, 1000),
        'purchase_amount': np.random.exponential(100, 1000)
    }
    
    # Introduce some missing values
    for col in ['age', 'income']:
        missing_indices = np.random.choice(1000, size=50, replace=False)
        data[col][missing_indices] = np.nan
    
    # Introduce some outliers
    outlier_indices = np.random.choice(1000, size=20, replace=False)
    data['income'][outlier_indices] *= 10
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    # Test the data processor
    sample_df = create_sample_dataset()
    processor = DataProcessor()
    
    analysis = processor.analyze_dataset(sample_df)
    print("Data Processor Test Completed Successfully!")
    print(f"Dataset Shape: {sample_df.shape}")
    print(f"Data Quality Score: {analysis['data_quality']['overall']}")