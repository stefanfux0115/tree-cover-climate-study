"""
Comprehensive ML Pipeline for Urban Climate-Energy Analysis - City Level
Trains multiple ML models, generates predictions, and provides interpretability analysis
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import warnings
import json
import joblib
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb
import optuna

# Interpretability
import shap

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(str(Path(__file__).parent.parent))
from config_loader import load_ml_analysis_config

class CityMLPipeline:
    """Comprehensive ML pipeline for urban climate-energy analysis at city level"""
    
    def __init__(self):
        self.config = load_ml_analysis_config()
        self.data_dir = Path(__file__).parent.parent.parent / "data" / "processed" / "ml_analysis_city"
        self.feature_dir = Path(__file__).parent.parent.parent / "output" / "feature_selection" / "city_level"
        self.output_dir = Path(__file__).parent.parent.parent / "output" / "ml_pipeline" / "city_level"
        
        # Create organized subdirectories
        self.models_dir = self.output_dir / "trained_models"
        self.results_dir = self.output_dir / "model_results"
        self.interpretability_dir = self.output_dir / "interpretability"
        self.predictions_dir = self.output_dir / "predictions"
        self.policy_dir = self.output_dir / "policy_analysis"
        
        # Create all directories
        for dir_path in [self.output_dir, self.models_dir, self.results_dir, 
                        self.interpretability_dir, self.predictions_dir, self.policy_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        self.setup_logging()
        
        # Set style for better visualizations
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Initialize containers
        self.models = {}
        self.results = {}
        self.scalers = {}
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(__file__).parent.parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"city_ml_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_data_and_features(self) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """Load engineered data and selected features"""
        self.logger.info("Loading engineered data and selected features...")
        
        # Load engineered datasets
        engineered_dir = self.data_dir / "engineered"
        datasets = {}
        
        for split in ['train', 'validation', 'test']:
            file_path = engineered_dir / f"{split}_engineered.parquet"
            if file_path.exists():
                datasets[split] = pd.read_parquet(file_path)
                self.logger.info(f"Loaded {split}: {datasets[split].shape[0]:,} rows, {datasets[split].shape[1]} features")
            else:
                raise FileNotFoundError(f"Dataset not found: {file_path}")
        
        # Load selected features
        selected_features_file = self.feature_dir / "final_selection" / "selected_features_list.txt"
        if selected_features_file.exists():
            with open(selected_features_file, 'r') as f:
                selected_features = [line.strip() for line in f if line.strip()]
            self.logger.info(f"Loaded {len(selected_features)} selected features")
        else:
            raise FileNotFoundError(f"Selected features file not found: {selected_features_file}")
        
        return datasets, selected_features
        
    def prepare_ml_data(self, datasets: Dict[str, pd.DataFrame], 
                       selected_features: List[str]) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """Prepare data for ML training and evaluation"""
        self.logger.info("Preparing ML data...")
        
        ml_data = {}
        
        # First pass: identify categorical features and create encoders
        from sklearn.preprocessing import LabelEncoder
        encoders = {}
        
        # Use training data to fit encoders
        train_df = datasets['train']
        available_features = [f for f in selected_features if f in train_df.columns]
        missing_features = [f for f in selected_features if f not in train_df.columns]
        
        if missing_features:
            self.logger.warning(f"Missing features: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
        
        # Identify and prepare encoders for categorical features
        for feature in available_features:
            if train_df[feature].dtype == 'object':
                self.logger.info(f"Creating encoder for categorical feature: {feature}")
                encoder = LabelEncoder()
                # Combine all unique values from all splits to ensure consistent encoding
                all_values = []
                for split_df in datasets.values():
                    if feature in split_df.columns:
                        all_values.extend(split_df[feature].dropna().unique())
                encoder.fit(list(set(all_values)))
                encoders[feature] = encoder
        
        # Second pass: process each split with consistent encoding
        for split, df in datasets.items():
            X = df[available_features].copy()
            y = df['ln_elec_kwh'].copy()
            
            # Encode categorical features
            for feature in X.columns:
                if feature in encoders:
                    # Handle missing values before encoding
                    X[feature] = X[feature].fillna('missing')
                    X[feature] = encoders[feature].transform(X[feature])
                elif X[feature].dtype == 'object':
                    # If somehow we missed a categorical feature, convert to numeric
                    self.logger.warning(f"Converting unexpected categorical feature {feature} to numeric")
                    X[feature] = pd.to_numeric(X[feature], errors='coerce')
            
            # Handle missing values for numeric features
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                X[col] = X[col].fillna(X[col].median())
            
            # Final check: ensure all features are numeric
            for col in X.columns:
                if X[col].dtype == 'object':
                    self.logger.error(f"Feature {col} is still categorical after processing")
                    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            
            ml_data[split] = (X, y)
            self.logger.info(f"Prepared {split}: {X.shape[0]:,} samples, {X.shape[1]} features")
        
        # Store encoders for later use
        self.encoders = encoders
        
        return ml_data
        
    def define_models(self) -> Dict[str, Any]:
        """Define ML models with hyperparameter spaces"""
        self.logger.info("Defining ML models...")
        
        models = {
            'ridge': {
                'model': Ridge(random_state=42),
                'param_grid': {
                    'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
                },
                'needs_scaling': True
            },
            'lasso': {
                'model': Lasso(random_state=42, max_iter=2000),
                'param_grid': {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
                },
                'needs_scaling': True
            },
            'random_forest': {
                'model': RandomForestRegressor(random_state=42, n_jobs=-1),
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'needs_scaling': False
            },
            'lightgbm': {
                'model': lgb.LGBMRegressor(random_state=42, verbose=-1),
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7, -1],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [31, 50, 100],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'needs_scaling': False
            },
            'xgboost': {
                'model': xgb.XGBRegressor(random_state=42),
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                },
                'needs_scaling': False
            }
        }
        
        return models
        
    def train_single_model(self, model_name: str, model_config: Dict, 
                          X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Train a single model with hyperparameter optimization"""
        self.logger.info(f"Training {model_name}...")
        
        # Prepare data with scaling if needed
        if model_config['needs_scaling']:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            self.scalers[model_name] = scaler
        else:
            X_train_scaled = X_train.values
            X_val_scaled = X_val.values
            
        # Hyperparameter optimization
        grid_search = GridSearchCV(
            model_config['model'],
            model_config['param_grid'],
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        
        # Make predictions
        y_train_pred = best_model.predict(X_train_scaled)
        y_val_pred = best_model.predict(X_val_scaled)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(best_model, X_train_scaled, y_train, 
                                   cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())
        
        results = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'cv_rmse': cv_rmse,
            'y_train_pred': y_train_pred,
            'y_val_pred': y_val_pred
        }
        
        self.logger.info(f"{model_name} - Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}, CV RMSE: {cv_rmse:.4f}")
        
        return results
        
    def train_all_models(self, ml_data: Dict[str, Tuple[pd.DataFrame, pd.Series]]) -> None:
        """Train all models"""
        self.logger.info("Training all models...")
        
        X_train, y_train = ml_data['train']
        X_val, y_val = ml_data['validation']
        
        models = self.define_models()
        
        for model_name, model_config in models.items():
            try:
                self.results[model_name] = self.train_single_model(
                    model_name, model_config, X_train, y_train, X_val, y_val
                )
                self.models[model_name] = self.results[model_name]['model']
                
                # Save model
                model_file = self.models_dir / f"{model_name}_model.joblib"
                joblib.dump(self.results[model_name]['model'], model_file)
                
                if model_name in self.scalers:
                    scaler_file = self.models_dir / f"{model_name}_scaler.joblib"
                    joblib.dump(self.scalers[model_name], scaler_file)
                    
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {str(e)}")
        
        self.logger.info(f"Successfully trained {len(self.results)} models")
        
    def generate_model_comparison(self) -> pd.DataFrame:
        """Generate comprehensive model comparison"""
        self.logger.info("Generating model comparison...")
        
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Train R²': results['train_r2'],
                'Validation R²': results['val_r2'],
                'Train RMSE': results['train_rmse'],
                'Validation RMSE': results['val_rmse'],
                'Train MAE': results['train_mae'],
                'Validation MAE': results['val_mae'],
                'CV RMSE': results['cv_rmse'],
                'Overfitting': results['train_r2'] - results['val_r2']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Validation R²', ascending=False)
        
        # Save comparison
        comparison_df.to_csv(self.results_dir / 'model_comparison.csv', index=False)
        
        return comparison_df
        
    def visualize_model_results(self, comparison_df: pd.DataFrame, 
                               ml_data: Dict[str, Tuple[pd.DataFrame, pd.Series]]) -> None:
        """Create comprehensive visualizations of model results"""
        self.logger.info("Creating model result visualizations...")
        
        # 1. Model Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # R² comparison
        x_pos = np.arange(len(comparison_df))
        axes[0, 0].bar(x_pos - 0.2, comparison_df['Train R²'], 0.4, label='Train R²', alpha=0.7)
        axes[0, 0].bar(x_pos + 0.2, comparison_df['Validation R²'], 0.4, label='Validation R²', alpha=0.7)
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_title('Model Performance: R² Comparison')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(comparison_df['Model'], rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # RMSE comparison
        axes[0, 1].bar(x_pos - 0.2, comparison_df['Train RMSE'], 0.4, label='Train RMSE', alpha=0.7)
        axes[0, 1].bar(x_pos + 0.2, comparison_df['Validation RMSE'], 0.4, label='Validation RMSE', alpha=0.7)
        axes[0, 1].set_xlabel('Models')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_title('Model Performance: RMSE Comparison')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(comparison_df['Model'], rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Overfitting analysis
        colors = ['red' if x > 0.1 else 'orange' if x > 0.05 else 'green' for x in comparison_df['Overfitting']]
        axes[1, 0].bar(x_pos, comparison_df['Overfitting'], color=colors, alpha=0.7)
        axes[1, 0].set_xlabel('Models')
        axes[1, 0].set_ylabel('Overfitting (Train R² - Val R²)')
        axes[1, 0].set_title('Overfitting Analysis')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(comparison_df['Model'], rotation=45)
        axes[1, 0].axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Warning (0.05)')
        axes[1, 0].axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='High (0.10)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cross-validation RMSE
        axes[1, 1].bar(x_pos, comparison_df['CV RMSE'], color='skyblue', alpha=0.7)
        axes[1, 1].set_xlabel('Models')
        axes[1, 1].set_ylabel('Cross-Validation RMSE')
        axes[1, 1].set_title('Cross-Validation Performance')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(comparison_df['Model'], rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Prediction vs Actual plots for best model
        best_model_name = comparison_df.iloc[0]['Model'].lower().replace(' ', '_')
        if best_model_name in self.results:
            X_train, y_train = ml_data['train']
            X_val, y_val = ml_data['validation']
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Training predictions
            y_train_pred = self.results[best_model_name]['y_train_pred']
            axes[0].scatter(y_train, y_train_pred, alpha=0.5, s=20)
            axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
            axes[0].set_xlabel('Actual Log Electricity Consumption')
            axes[0].set_ylabel('Predicted Log Electricity Consumption')
            axes[0].set_title(f'{best_model_name.title()} - Training Predictions\nR² = {self.results[best_model_name]["train_r2"]:.4f}')
            axes[0].grid(True, alpha=0.3)
            
            # Validation predictions
            y_val_pred = self.results[best_model_name]['y_val_pred']
            axes[1].scatter(y_val, y_val_pred, alpha=0.5, s=20, color='orange')
            axes[1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
            axes[1].set_xlabel('Actual Log Electricity Consumption')
            axes[1].set_ylabel('Predicted Log Electricity Consumption')
            axes[1].set_title(f'{best_model_name.title()} - Validation Predictions\nR² = {self.results[best_model_name]["val_r2"]:.4f}')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.results_dir / f'{best_model_name}_predictions.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info("Model result visualizations saved")
        
    def generate_feature_importance_analysis(self, ml_data: Dict[str, Tuple[pd.DataFrame, pd.Series]]) -> None:
        """Generate feature importance analysis for tree-based models"""
        self.logger.info("Generating feature importance analysis...")
        
        X_train, y_train = ml_data['train']
        feature_names = X_train.columns.tolist()
        
        # Collect feature importance from tree-based models
        tree_models = ['random_forest', 'lightgbm', 'xgboost']
        importance_data = []
        
        for model_name in tree_models:
            if model_name in self.results:
                model = self.results[model_name]['model']
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    for feature, importance in zip(feature_names, importances):
                        importance_data.append({
                            'Model': model_name.replace('_', ' ').title(),
                            'Feature': feature,
                            'Importance': importance
                        })
        
        if importance_data:
            importance_df = pd.DataFrame(importance_data)
            
            # Average importance across models
            avg_importance = importance_df.groupby('Feature')['Importance'].mean().sort_values(ascending=False)
            
            # Visualize top 20 features
            plt.figure(figsize=(12, 10))
            top_features = avg_importance.head(20)
            plt.barh(range(len(top_features)), top_features.values)
            plt.yticks(range(len(top_features)), top_features.index)
            plt.xlabel('Average Feature Importance')
            plt.title('Top 20 Features by Average Importance (Tree Models)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(self.interpretability_dir / 'feature_importance_tree_models.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save importance data
            avg_importance.to_csv(self.interpretability_dir / 'average_feature_importance.csv')
            
        self.logger.info("Feature importance analysis completed")
        
    def save_comprehensive_results(self, comparison_df: pd.DataFrame) -> None:
        """Save comprehensive results and summary"""
        self.logger.info("Saving comprehensive results...")
        
        # Create summary results
        results_summary = {
            'pipeline_info': {
                'run_date': datetime.now().isoformat(),
                'total_models_trained': len(self.results),
                'best_model': comparison_df.iloc[0]['Model'] if len(comparison_df) > 0 else None,
                'best_validation_r2': float(comparison_df.iloc[0]['Validation R²']) if len(comparison_df) > 0 else None,
                'feature_count': len(ml_data['train'][0].columns) if 'ml_data' in locals() else None
            },
            'model_performance': comparison_df.to_dict('records'),
            'hyperparameters': {model_name: results['best_params'] 
                              for model_name, results in self.results.items()}
        }
        
        # Save summary
        with open(self.results_dir / 'pipeline_results_summary.json', 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        
        # Save detailed results for each model
        for model_name, results in self.results.items():
            model_results = {
                'model_name': model_name,
                'hyperparameters': results['best_params'],
                'performance_metrics': {
                    'train_r2': results['train_r2'],
                    'validation_r2': results['val_r2'],
                    'train_rmse': results['train_rmse'],
                    'validation_rmse': results['val_rmse'],
                    'train_mae': results['train_mae'],
                    'validation_mae': results['val_mae'],
                    'cv_rmse': results['cv_rmse']
                }
            }
            
            with open(self.results_dir / f'{model_name}_detailed_results.json', 'w') as f:
                json.dump(model_results, f, indent=2)
        
        self.logger.info("Comprehensive results saved")
        
    def run_ml_pipeline(self) -> Dict[str, Any]:
        """Main method to run the complete ML pipeline"""
        self.logger.info("Starting comprehensive ML pipeline...")
        
        try:
            # Load data and features
            datasets, selected_features = self.load_data_and_features()
            
            # Prepare ML data
            ml_data = self.prepare_ml_data(datasets, selected_features)
            
            # Train all models
            self.train_all_models(ml_data)
            
            # Generate model comparison
            comparison_df = self.generate_model_comparison()
            
            # Create visualizations
            self.visualize_model_results(comparison_df, ml_data)
            
            # Generate feature importance analysis
            self.generate_feature_importance_analysis(ml_data)
            
            # Save comprehensive results
            self.save_comprehensive_results(comparison_df)
            
            self.logger.info("ML pipeline completed successfully!")
            
            return {
                'models': self.models,
                'results': self.results,
                'comparison': comparison_df,
                'ml_data': ml_data
            }
            
        except Exception as e:
            self.logger.error(f"Error in ML pipeline: {str(e)}")
            raise


def main():
    """Main function for standalone execution"""
    pipeline = CityMLPipeline()
    results = pipeline.run_ml_pipeline()
    
    print(f"\nCity ML Pipeline Completed Successfully!")
    print(f"Trained {len(results['models'])} models")
    print(f"Results saved to: {pipeline.output_dir}")
    
    # Show best model results
    if len(results['comparison']) > 0:
        best_model = results['comparison'].iloc[0]
        print(f"\nBest Model: {best_model['Model']}")
        print(f"Validation R²: {best_model['Validation R²']:.4f}")
        print(f"Validation RMSE: {best_model['Validation RMSE']:.4f}")
        print(f"Cross-validation RMSE: {best_model['CV RMSE']:.4f}")
    
    return results


if __name__ == "__main__":
    pipeline_results = main()