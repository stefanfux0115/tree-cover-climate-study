"""
Feature Selection with Comprehensive Visualizations
Analyzes feature importance, correlations, and selects optimal feature subset
for interpretable ML analysis
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(str(Path(__file__).parent.parent))
from config_loader import load_ml_analysis_config

class FeatureSelector:
    """Comprehensive feature selection with visualizations for urban climate-energy ML"""
    
    def __init__(self):
        self.config = load_ml_analysis_config()
        self.data_dir = Path(__file__).parent.parent.parent / "data" / "processed" / "ml_analysis_grid"
        self.output_dir = Path(__file__).parent.parent.parent / "output" / "feature_selection" / "grid_level"
        
        # Create organized subdirectories
        self.eda_dir = self.output_dir / "exploratory_data_analysis"
        self.correlation_dir = self.output_dir / "correlation_analysis"
        self.importance_dir = self.output_dir / "feature_importance"
        self.selection_dir = self.output_dir / "final_selection"
        
        # Create all directories
        for dir_path in [self.output_dir, self.eda_dir, self.correlation_dir, self.importance_dir, self.selection_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        self.setup_logging()
        
        # Set style for better visualizations
        plt.style.use('default')
        sns.set_palette("husl")
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(__file__).parent.parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"feature_selection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_engineered_data(self) -> Dict[str, pd.DataFrame]:
        """Load the feature-engineered datasets"""
        self.logger.info("Loading feature-engineered datasets...")
        
        engineered_dir = self.data_dir / "engineered"
        datasets = {}
        
        for split in ['train', 'validation', 'test']:
            file_path = engineered_dir / f"{split}_engineered.parquet"
            if file_path.exists():
                datasets[split] = pd.read_parquet(file_path)
                self.logger.info(f"Loaded {split}: {datasets[split].shape[0]:,} rows, {datasets[split].shape[1]} features")
            else:
                self.logger.warning(f"Dataset not found: {file_path}")
                
        return datasets
        
    def identify_feature_groups(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Identify and categorize features"""
        self.logger.info("Identifying feature groups...")
        
        feature_groups = {
            'target': ['elec_kwh', 'ln_elec_kwh'],
            'spatial': ['grid_row', 'grid_col', 'x_albers', 'y_albers', 'lon', 'lat', 'grid_id'],
            'temporal': ['year', 'month', 'year_month'],
            'location': ['city_id', 'city_name', 'province'],
            'temperature_bins': [col for col in df.columns if 'days_in_bin_' in col and '_x_canopy' not in col],
            'temperature_interactions': [col for col in df.columns if 'days_in_bin_' in col and '_x_canopy' in col],
            'temperature_derived': [col for col in df.columns if any(x in col for x in ['extreme_', 'moderate_', 'entropy', 'total_days'])],
            'climate_raw': [col for col in df.columns if col in ['prec', 'pres', 'rhu', 'wind']],
            'climate_categorical': [col for col in df.columns if any(x in col for x in ['humidity_', 'precipitation_'])],
            'socioeconomic_raw': [col for col in df.columns if col in ['gdp_per_capita', 'population', 'nightlight']],
            'socioeconomic_categorical': [col for col in df.columns if any(x in col for x in ['gdp_', 'population_', 'nightlight_'])],
            'canopy': [col for col in df.columns if 'canopy' in col and '_x_' not in col and 'days_in_bin' not in col],
            'canopy_categorical': [col for col in df.columns if 'canopy_' in col and '_x_' not in col],
            'seasonal': [col for col in df.columns if any(x in col for x in ['season_', 'heating_', 'cooling_'])],
            'interactions': [col for col in df.columns if '_x_' in col and 'days_in_bin' not in col]
        }
        
        # Log feature group sizes
        for group, features in feature_groups.items():
            if features:
                self.logger.info(f"  {group}: {len(features)} features")
                
        return feature_groups
        
    def prepare_ml_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Prepare data for ML analysis"""
        self.logger.info("Preparing data for ML analysis...")
        
        # Identify features to exclude (technical/identifier variables not for ML)
        exclude_features = [
            # Target variables
            'elec_kwh', 'ln_elec_kwh', 
            # Identifiers
            'grid_id', 'city_id', 'city_name', 'province', 'year_month',
            # Technical spatial coordinates (not policy-relevant)
            'x_albers', 'y_albers', 'grid_row', 'grid_col'
        ]
        
        # Find common features between train and validation
        train_cols = set(train_df.columns) - set(exclude_features)
        val_cols = set(val_df.columns) - set(exclude_features)
        common_features = list(train_cols.intersection(val_cols))
        
        self.logger.info(f"Found {len(common_features)} common features between train and validation")
        if len(train_cols) != len(val_cols):
            missing_in_val = train_cols - val_cols
            missing_in_train = val_cols - train_cols
            if missing_in_val:
                self.logger.warning(f"Features in train but not validation: {list(missing_in_val)}")
            if missing_in_train:
                self.logger.warning(f"Features in validation but not train: {list(missing_in_train)}")
        
        # Select features for analysis
        feature_cols = common_features
        
        # Handle missing values (fill with median for numeric, mode for categorical)
        train_features = train_df[feature_cols].copy()
        val_features = val_df[feature_cols].copy()
        
        for col in feature_cols:
            if train_features[col].dtype in ['object', 'category']:
                mode_val = train_features[col].mode().iloc[0] if len(train_features[col].mode()) > 0 else train_features[col].iloc[0]
                train_features[col] = train_features[col].fillna(mode_val)
                val_features[col] = val_features[col].fillna(mode_val)
            else:
                median_val = train_features[col].median()
                train_features[col] = train_features[col].fillna(median_val)
                val_features[col] = val_features[col].fillna(median_val)
        
        # Target variable (use log-transformed electricity consumption)
        train_target = train_df['ln_elec_kwh'].copy()
        val_target = val_df['ln_elec_kwh'].copy()
        
        self.logger.info(f"Prepared features: {len(feature_cols)} columns")
        self.logger.info(f"Training samples: {len(train_features):,}")
        self.logger.info(f"Validation samples: {len(val_features):,}")
        
        return train_features, train_target, val_features, val_target
        
    def visualize_feature_distributions(self, train_df: pd.DataFrame, feature_groups: Dict[str, List[str]]) -> None:
        """Create comprehensive feature distribution visualizations"""
        self.logger.info("Creating feature distribution visualizations...")
        
        # 1. Target variable distribution
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Raw electricity consumption
        axes[0].hist(train_df['elec_kwh'], bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Electricity Consumption (kWh)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Raw Electricity Consumption')
        axes[0].set_yscale('log')
        
        # Log-transformed electricity consumption
        axes[1].hist(train_df['ln_elec_kwh'], bins=50, alpha=0.7, edgecolor='black', color='orange')
        axes[1].set_xlabel('Log Electricity Consumption')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Log-Transformed Electricity Consumption')
        
        plt.tight_layout()
        plt.savefig(self.eda_dir / 'target_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Temperature bins distribution
        temp_bins = feature_groups['temperature_bins']
        if temp_bins:
            fig, ax = plt.subplots(figsize=(16, 8))
            temp_data = train_df[temp_bins].mean()
            
            bars = ax.bar(range(len(temp_data)), temp_data.values, 
                         color='skyblue', edgecolor='navy', alpha=0.7)
            ax.set_xlabel('Temperature Bins')
            ax.set_ylabel('Average Days per Month')
            ax.set_title('Average Days per Temperature Bin (Training Data)')
            ax.set_xticks(range(len(temp_data)))
            ax.set_xticklabels([col.replace('days_in_bin_', '') for col in temp_data.index], rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, temp_data.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                       f'{value:.1f}', ha='center', va='bottom')
                       
            plt.tight_layout()
            plt.savefig(self.eda_dir / 'temperature_bins_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Climate variables distribution
        climate_vars = feature_groups['climate_raw']
        if climate_vars:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            for i, var in enumerate(climate_vars):
                if var in train_df.columns:
                    axes[i].hist(train_df[var], bins=50, alpha=0.7, edgecolor='black')
                    axes[i].set_xlabel(var.upper())
                    axes[i].set_ylabel('Frequency')
                    axes[i].set_title(f'Distribution of {var.upper()}')
                    axes[i].set_yscale('log')
            
            plt.tight_layout()
            plt.savefig(self.eda_dir / 'climate_variables_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Socioeconomic variables distribution  
        socio_vars = feature_groups['socioeconomic_raw']
        if socio_vars:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            for i, var in enumerate(socio_vars):
                if var in train_df.columns:
                    # Use log scale for better visualization
                    data = train_df[var][train_df[var] > 0]  # Remove zeros for log
                    axes[i].hist(np.log10(data + 1), bins=50, alpha=0.7, edgecolor='black')
                    axes[i].set_xlabel(f'Log10({var})')
                    axes[i].set_ylabel('Frequency')
                    axes[i].set_title(f'Distribution of {var} (Log Scale)')
            
            plt.tight_layout()
            plt.savefig(self.eda_dir / 'socioeconomic_variables_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        self.logger.info("Feature distribution visualizations saved")
        
    def compute_correlation_analysis(self, X_train: pd.DataFrame, y_train: pd.Series, feature_groups: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
        """Compute correlation analysis between features and target"""
        self.logger.info("Computing correlation analysis...")
        
        # Compute correlations with target
        correlations = X_train.corrwith(y_train).abs().sort_values(ascending=False)
        correlations_df = pd.DataFrame({
            'feature': correlations.index,
            'correlation': correlations.values
        })
        
        # Add feature group information
        def get_feature_group(feature_name):
            for group, features in feature_groups.items():
                if feature_name in features:
                    return group
            return 'other'
        
        correlations_df['group'] = correlations_df['feature'].apply(get_feature_group)
        
        # Create correlation visualization
        plt.figure(figsize=(12, 20))
        top_features = correlations_df.head(50)  # Top 50 features
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_features['group'].unique())))
        group_colors = dict(zip(top_features['group'].unique(), colors))
        bar_colors = [group_colors[group] for group in top_features['group']]
        
        bars = plt.barh(range(len(top_features)), top_features['correlation'], color=bar_colors)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Absolute Correlation with Log Electricity Consumption')
        plt.title('Top 50 Features by Correlation with Target Variable')
        plt.gca().invert_yaxis()
        
        # Add legend
        handles = [plt.Rectangle((0,0),1,1, color=group_colors[group]) for group in group_colors]
        plt.legend(handles, group_colors.keys(), loc='lower right', bbox_to_anchor=(1.3, 0))
        
        plt.tight_layout()
        plt.savefig(self.correlation_dir / 'feature_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Compute feature-feature correlation matrix for top features
        top_feature_names = top_features.head(30)['feature'].tolist()
        feature_corr_matrix = X_train[top_feature_names].corr()
        
        # Visualize feature-feature correlations
        plt.figure(figsize=(16, 14))
        mask = np.triu(np.ones_like(feature_corr_matrix, dtype=bool))
        sns.heatmap(feature_corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Feature-Feature Correlation Matrix (Top 30 Features)')
        plt.tight_layout()
        plt.savefig(self.correlation_dir / 'feature_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'target_correlations': correlations_df,
            'feature_correlations': feature_corr_matrix
        }
        
    def compute_feature_importance_methods(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                         X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, pd.DataFrame]:
        """Compute feature importance using multiple methods"""
        self.logger.info("Computing feature importance using multiple methods...")
        
        # Sample data for faster computation (if too large)
        if len(X_train) > 100000:
            self.logger.info("Sampling data for feature importance computation...")
            sample_idx = np.random.choice(len(X_train), 100000, replace=False)
            X_train_sample = X_train.iloc[sample_idx]
            y_train_sample = y_train.iloc[sample_idx]
        else:
            X_train_sample = X_train
            y_train_sample = y_train
        
        importance_results = {}
        
        # 1. Random Forest Feature Importance
        self.logger.info("Computing Random Forest feature importance...")
        try:
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train_sample, y_train_sample)
            
            rf_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            importance_results['random_forest'] = rf_importance
            
            # Compute validation score
            y_pred = rf.predict(X_val)
            rf_r2 = r2_score(y_val, y_pred)
            self.logger.info(f"Random Forest R²: {rf_r2:.4f}")
            
        except Exception as e:
            self.logger.warning(f"Random Forest failed: {str(e)}")
        
        # 2. LightGBM Feature Importance
        self.logger.info("Computing LightGBM feature importance...")
        try:
            lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
            lgb_model.fit(X_train_sample, y_train_sample)
            
            lgb_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': lgb_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            importance_results['lightgbm'] = lgb_importance
            
            # Compute validation score
            y_pred = lgb_model.predict(X_val)
            lgb_r2 = r2_score(y_val, y_pred)
            self.logger.info(f"LightGBM R²: {lgb_r2:.4f}")
            
        except Exception as e:
            self.logger.warning(f"LightGBM failed: {str(e)}")
        
        # 3. Lasso Feature Selection
        self.logger.info("Computing Lasso feature importance...")
        try:
            # Standardize features for Lasso
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_sample)
            
            lasso = LassoCV(cv=5, random_state=42)
            lasso.fit(X_train_scaled, y_train_sample)
            
            lasso_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': np.abs(lasso.coef_)
            }).sort_values('importance', ascending=False)
            
            importance_results['lasso'] = lasso_importance
            
            # Compute validation score
            X_val_scaled = scaler.transform(X_val)
            y_pred = lasso.predict(X_val_scaled)
            lasso_r2 = r2_score(y_val, y_pred)
            self.logger.info(f"Lasso R²: {lasso_r2:.4f}")
            
        except Exception as e:
            self.logger.warning(f"Lasso failed: {str(e)}")
        
        # 4. Mutual Information
        self.logger.info("Computing Mutual Information...")
        try:
            mi_scores = mutual_info_regression(X_train_sample, y_train_sample, random_state=42)
            
            mi_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': mi_scores
            }).sort_values('importance', ascending=False)
            
            importance_results['mutual_info'] = mi_importance
            
        except Exception as e:
            self.logger.warning(f"Mutual Information failed: {str(e)}")
        
        return importance_results
        
    def visualize_feature_importance(self, importance_results: Dict[str, pd.DataFrame], 
                                   feature_groups: Dict[str, List[str]]) -> None:
        """Visualize feature importance from multiple methods"""
        self.logger.info("Creating feature importance visualizations...")
        
        # Create subplots for each method
        n_methods = len(importance_results)
        if n_methods == 0:
            self.logger.warning("No feature importance results to visualize")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        # Define feature group colors
        def get_feature_group(feature_name):
            for group, features in feature_groups.items():
                if feature_name in features:
                    return group
            return 'other'
        
        for i, (method, importance_df) in enumerate(importance_results.items()):
            if i >= 4:  # Only show first 4 methods
                break
                
            top_features = importance_df.head(25)  # Top 25 features
            top_features['group'] = top_features['feature'].apply(get_feature_group)
            
            # Color by feature group
            unique_groups = top_features['group'].unique()
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_groups)))
            group_colors = dict(zip(unique_groups, colors))
            bar_colors = [group_colors[group] for group in top_features['group']]
            
            axes[i].barh(range(len(top_features)), top_features['importance'], color=bar_colors)
            axes[i].set_yticks(range(len(top_features)))
            axes[i].set_yticklabels(top_features['feature'], fontsize=8)
            axes[i].set_xlabel('Feature Importance')
            axes[i].set_title(f'Top 25 Features - {method.title().replace("_", " ")}')
            axes[i].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.importance_dir / 'feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create consensus ranking
        self.create_consensus_ranking(importance_results, feature_groups)
        
    def create_consensus_ranking(self, importance_results: Dict[str, pd.DataFrame], 
                               feature_groups: Dict[str, List[str]]) -> pd.DataFrame:
        """Create consensus feature ranking across methods"""
        self.logger.info("Creating consensus feature ranking...")
        
        # Normalize importance scores to 0-1 range for each method
        normalized_results = {}
        for method, importance_df in importance_results.items():
            normalized_scores = importance_df['importance'] / importance_df['importance'].max()
            normalized_results[method] = pd.DataFrame({
                'feature': importance_df['feature'],
                'importance': normalized_scores
            })
        
        # Combine all methods
        all_features = set()
        for method_df in normalized_results.values():
            all_features.update(method_df['feature'])
        
        consensus_data = []
        for feature in all_features:
            scores = []
            for method, method_df in normalized_results.items():
                feature_score = method_df[method_df['feature'] == feature]
                if not feature_score.empty:
                    scores.append(feature_score['importance'].iloc[0])
                else:
                    scores.append(0.0)
            
            consensus_data.append({
                'feature': feature,
                'mean_importance': np.mean(scores),
                'std_importance': np.std(scores),
                'min_importance': np.min(scores),
                'max_importance': np.max(scores),
                'methods_count': sum(1 for s in scores if s > 0)
            })
        
        consensus_df = pd.DataFrame(consensus_data).sort_values('mean_importance', ascending=False)
        
        # Add feature group information
        def get_feature_group(feature_name):
            for group, features in feature_groups.items():
                if feature_name in features:
                    return group
            return 'other'
        
        consensus_df['group'] = consensus_df['feature'].apply(get_feature_group)
        
        # Visualize consensus ranking
        plt.figure(figsize=(14, 20))
        top_consensus = consensus_df.head(50)
        
        # Color by feature group
        unique_groups = top_consensus['group'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_groups)))
        group_colors = dict(zip(unique_groups, colors))
        bar_colors = [group_colors[group] for group in top_consensus['group']]
        
        bars = plt.barh(range(len(top_consensus)), top_consensus['mean_importance'], 
                       color=bar_colors, alpha=0.7)
        
        # Add error bars
        plt.errorbar(top_consensus['mean_importance'], range(len(top_consensus)), 
                    xerr=top_consensus['std_importance'], fmt='none', color='black', alpha=0.5)
        
        plt.yticks(range(len(top_consensus)), top_consensus['feature'])
        plt.xlabel('Consensus Importance Score')
        plt.title('Top 50 Features - Consensus Ranking Across All Methods')
        plt.gca().invert_yaxis()
        
        # Add legend
        handles = [plt.Rectangle((0,0),1,1, color=group_colors[group]) for group in group_colors]
        plt.legend(handles, group_colors.keys(), loc='lower right', bbox_to_anchor=(1.4, 0))
        
        plt.tight_layout()
        plt.savefig(self.importance_dir / 'consensus_feature_ranking.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save consensus ranking
        consensus_df.to_csv(self.importance_dir / 'consensus_feature_ranking.csv', index=False)
        self.logger.info(f"Consensus ranking saved with {len(consensus_df)} features")
        
        return consensus_df
        
    def select_optimal_features(self, consensus_df: pd.DataFrame, 
                              cumulative_threshold: float = 0.95) -> List[str]:
        """Select optimal feature subset based on cumulative importance"""
        self.logger.info(f"Selecting optimal features with {cumulative_threshold:.0%} cumulative threshold...")
        
        # Calculate cumulative importance
        consensus_df = consensus_df.sort_values('mean_importance', ascending=False)
        consensus_df['cumulative_importance'] = consensus_df['mean_importance'].cumsum() / consensus_df['mean_importance'].sum()
        
        # Select features up to threshold
        selected_features = consensus_df[consensus_df['cumulative_importance'] <= cumulative_threshold]['feature'].tolist()
        
        # Ensure we have at least 20 features and at most 60 features
        if len(selected_features) < 20:
            selected_features = consensus_df.head(20)['feature'].tolist()
        elif len(selected_features) > 60:
            selected_features = consensus_df.head(60)['feature'].tolist()
        
        self.logger.info(f"Selected {len(selected_features)} features ({len(selected_features)/len(consensus_df)*100:.1f}% of total)")
        
        # Visualize cumulative importance
        plt.figure(figsize=(12, 8))
        plt.plot(range(1, len(consensus_df) + 1), consensus_df['cumulative_importance'], 
                linewidth=2, color='blue')
        plt.axhline(y=cumulative_threshold, color='red', linestyle='--', 
                   label=f'{cumulative_threshold:.0%} Threshold')
        plt.axvline(x=len(selected_features), color='red', linestyle='--', alpha=0.7,
                   label=f'Selected Features: {len(selected_features)}')
        plt.xlabel('Number of Features (Ranked by Importance)')
        plt.ylabel('Cumulative Importance')
        plt.title('Cumulative Feature Importance for Optimal Feature Selection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.selection_dir / 'cumulative_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return selected_features
        
    def save_feature_selection_results(self, selected_features: List[str], 
                                     consensus_df: pd.DataFrame,
                                     feature_groups: Dict[str, List[str]],
                                     correlation_results: Dict) -> None:
        """Save comprehensive feature selection results"""
        self.logger.info("Saving feature selection results...")
        
        # Create feature selection summary
        def get_feature_group(feature_name):
            for group, features in feature_groups.items():
                if feature_name in features:
                    return group
            return 'other'
        
        selected_df = consensus_df[consensus_df['feature'].isin(selected_features)].copy()
        selected_df['group'] = selected_df['feature'].apply(get_feature_group)
        
        # Group summary
        group_summary = selected_df['group'].value_counts().to_dict()
        
        # Create comprehensive results
        results = {
            'feature_selection_summary': {
                'total_original_features': len(consensus_df),
                'selected_features_count': len(selected_features),
                'selection_ratio': len(selected_features) / len(consensus_df),
                'group_distribution': group_summary,
                'selection_date': datetime.now().isoformat()
            },
            'selected_features': selected_features,
            'feature_details': selected_df.to_dict('records'),
            'feature_groups': {group: [f for f in features if f in selected_features] 
                             for group, features in feature_groups.items()},
            'correlation_summary': {
                'top_correlated_features': correlation_results['target_correlations'].head(20).to_dict('records')
            }
        }
        
        # Save results
        import json
        with open(self.selection_dir / 'feature_selection_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save selected features list for easy import
        selected_df.to_csv(self.selection_dir / 'selected_features.csv', index=False)
        
        # Save simple feature list
        with open(self.selection_dir / 'selected_features_list.txt', 'w', encoding='utf-8') as f:
            for feature in selected_features:
                f.write(f"{feature}\n")
                
        self.logger.info("Feature selection results saved successfully")
        
    def run_feature_selection(self) -> Dict[str, any]:
        """Main method to run comprehensive feature selection pipeline"""
        self.logger.info("Starting comprehensive feature selection...")
        
        try:
            # Load data
            datasets = self.load_engineered_data()
            
            if 'train' not in datasets or 'validation' not in datasets:
                raise ValueError("Training and validation datasets required")
            
            # Identify feature groups
            feature_groups = self.identify_feature_groups(datasets['train'])
            
            # Prepare ML data
            X_train, y_train, X_val, y_val = self.prepare_ml_data(
                datasets['train'], datasets['validation']
            )
            
            # Create visualizations
            self.visualize_feature_distributions(datasets['train'], feature_groups)
            
            # Compute correlation analysis
            correlation_results = self.compute_correlation_analysis(X_train, y_train, feature_groups)
            
            # Compute feature importance using multiple methods
            importance_results = self.compute_feature_importance_methods(X_train, y_train, X_val, y_val)
            
            # Visualize feature importance
            self.visualize_feature_importance(importance_results, feature_groups)
            
            # Create consensus ranking
            consensus_df = self.create_consensus_ranking(importance_results, feature_groups)
            
            # Select optimal features
            selected_features = self.select_optimal_features(consensus_df)
            
            # Save results
            self.save_feature_selection_results(selected_features, consensus_df, 
                                              feature_groups, correlation_results)
            
            self.logger.info("Feature selection completed successfully!")
            
            return {
                'selected_features': selected_features,
                'consensus_ranking': consensus_df,
                'feature_groups': feature_groups,
                'importance_results': importance_results,
                'correlation_results': correlation_results
            }
            
        except Exception as e:
            self.logger.error(f"Error during feature selection: {str(e)}")
            raise


def main():
    """Main function for standalone execution"""
    selector = FeatureSelector()
    results = selector.run_feature_selection()
    
    print(f"\nFeature Selection Completed Successfully!")
    print(f"Selected {len(results['selected_features'])} features out of {len(results['consensus_ranking'])}")
    print(f"Visualizations saved to: {selector.output_dir}")
    
    # Show top 10 selected features
    print(f"\nTop 10 Selected Features:")
    top_features = results['consensus_ranking'].head(10)
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        print(f"  {i}. {row['feature']} (score: {row['mean_importance']:.4f}, group: {row['group']})")
    
    return results


if __name__ == "__main__":
    selection_results = main()