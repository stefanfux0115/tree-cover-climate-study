"""
Comprehensive SHAP Analysis for City-Level Urban Climate-Energy Models
Provides interpretability analysis for trained ML models with policy-focused insights
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

# SHAP and visualization imports
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(str(Path(__file__).parent.parent))
from config_loader import load_ml_analysis_config

class CitySHAPAnalyzer:
    """Comprehensive SHAP analysis for city-level urban climate-energy models"""
    
    def __init__(self):
        self.config = load_ml_analysis_config()
        self.data_dir = Path(__file__).parent.parent.parent / "data" / "processed" / "ml_analysis_city"
        self.models_dir = Path(__file__).parent.parent.parent / "output" / "ml_pipeline" / "city_level" / "trained_models"
        self.output_dir = Path(__file__).parent.parent.parent / "output" / "shap_analysis" / "city_level"
        
        # Create organized subdirectories
        self.global_dir = self.output_dir / "global_explanations"
        self.local_dir = self.output_dir / "local_explanations"
        self.policy_dir = self.output_dir / "policy_analysis"
        self.interactions_dir = self.output_dir / "feature_interactions"
        
        # Create all directories
        for dir_path in [self.output_dir, self.global_dir, self.local_dir, 
                        self.policy_dir, self.interactions_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        self.setup_logging()
        
        # Set style for visualizations
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Initialize containers
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(__file__).parent.parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"city_shap_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_trained_models(self) -> Dict[str, Any]:
        """Load trained models and preprocessors"""
        self.logger.info("Loading trained models and preprocessors...")
        
        model_files = {
            'ridge': 'ridge_model.joblib',
            'lasso': 'lasso_model.joblib', 
            'random_forest': 'random_forest_model.joblib',
            'lightgbm': 'lightgbm_model.joblib',
            'xgboost': 'xgboost_model.joblib'
        }
        
        loaded_models = {}
        
        for model_name, filename in model_files.items():
            model_path = self.models_dir / filename
            if model_path.exists():
                try:
                    loaded_models[model_name] = joblib.load(model_path)
                    self.logger.info(f"Loaded {model_name} model")
                    
                    # Load scaler if it exists
                    scaler_path = self.models_dir / f"{model_name}_scaler.joblib"
                    if scaler_path.exists():
                        self.scalers[model_name] = joblib.load(scaler_path)
                        self.logger.info(f"Loaded {model_name} scaler")
                        
                except Exception as e:
                    self.logger.warning(f"Failed to load {model_name}: {str(e)}")
            else:
                self.logger.warning(f"Model file not found: {model_path}")
        
        self.models = loaded_models
        return loaded_models
        
    def load_data(self) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """Load engineered data and feature names"""
        self.logger.info("Loading engineered data...")
        
        # Load datasets
        engineered_dir = self.data_dir / "engineered"
        datasets = {}
        
        for split in ['train', 'validation', 'test']:
            file_path = engineered_dir / f"{split}_engineered.parquet"
            if file_path.exists():
                datasets[split] = pd.read_parquet(file_path)
                self.logger.info(f"Loaded {split}: {datasets[split].shape[0]:,} rows, {datasets[split].shape[1]} features")
        
        # Load selected features
        feature_dir = Path(__file__).parent.parent.parent / "output" / "feature_selection" / "city_level"
        selected_features_file = feature_dir / "final_selection" / "selected_features_list.txt"
        
        if selected_features_file.exists():
            with open(selected_features_file, 'r') as f:
                selected_features = [line.strip() for line in f if line.strip()]
            self.logger.info(f"Loaded {len(selected_features)} selected features")
        else:
            raise FileNotFoundError(f"Selected features file not found: {selected_features_file}")
        
        return datasets, selected_features
        
    def prepare_shap_data(self, datasets: Dict[str, pd.DataFrame], 
                         selected_features: List[str]) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """Prepare data for SHAP analysis with proper encoding"""
        self.logger.info("Preparing data for SHAP analysis...")
        
        # Reproduce the same preprocessing as in ML pipeline
        from sklearn.preprocessing import LabelEncoder
        encoders = {}
        
        # First pass: create encoders from training data
        train_df = datasets['train']
        available_features = [f for f in selected_features if f in train_df.columns]
        
        for feature in available_features:
            if train_df[feature].dtype == 'object':
                encoder = LabelEncoder()
                all_values = []
                for split_df in datasets.values():
                    if feature in split_df.columns:
                        all_values.extend(split_df[feature].dropna().unique())
                encoder.fit(list(set(all_values)))
                encoders[feature] = encoder
        
        self.encoders = encoders
        
        # Second pass: process data
        shap_data = {}
        for split, df in datasets.items():
            X = df[available_features].copy()
            y = df['ln_elec_kwh'].copy()
            
            # Encode categorical features
            for feature in X.columns:
                if feature in encoders:
                    X[feature] = X[feature].fillna('missing')
                    X[feature] = encoders[feature].transform(X[feature])
                elif X[feature].dtype == 'object':
                    X[feature] = pd.to_numeric(X[feature], errors='coerce')
            
            # Handle missing values
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                X[col] = X[col].fillna(X[col].median())
            
            shap_data[split] = (X, y)
            
        return shap_data
        
    def generate_global_shap_explanations(self, shap_data: Dict[str, Tuple[pd.DataFrame, pd.Series]]) -> None:
        """Generate global SHAP explanations for all models"""
        self.logger.info("Generating global SHAP explanations...")
        
        X_train, y_train = shap_data['train']
        feature_names = X_train.columns.tolist()
        
        # Sample data for efficient SHAP computation
        sample_size = min(1000, len(X_train))
        sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
        X_sample = X_train.iloc[sample_idx]
        
        for model_name, model in self.models.items():
            self.logger.info(f"Computing SHAP values for {model_name}...")
            
            try:
                # Prepare data for this model
                if model_name in self.scalers:
                    X_model = self.scalers[model_name].transform(X_sample)
                    explainer = shap.LinearExplainer(model, X_model)
                    shap_values = explainer.shap_values(X_model)
                    X_display = X_sample
                else:
                    # Tree-based models
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample)
                    X_display = X_sample
                
                # 1. SHAP Summary Plot (beeswarm)
                plt.figure(figsize=(12, 10))
                shap.summary_plot(shap_values, X_display, feature_names=feature_names, 
                                show=False, max_display=25)
                plt.title(f'SHAP Summary Plot - {model_name.title()}', fontsize=16, pad=20)
                plt.tight_layout()
                plt.savefig(self.global_dir / f'{model_name}_shap_summary.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                # 2. SHAP Feature Importance (bar plot)
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_display, feature_names=feature_names,
                                plot_type="bar", show=False, max_display=25)
                plt.title(f'SHAP Feature Importance - {model_name.title()}', fontsize=16)
                plt.tight_layout()
                plt.savefig(self.global_dir / f'{model_name}_shap_importance.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                # 3. Save SHAP importance values
                global_importance = np.abs(shap_values).mean(0)
                shap_importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'shap_importance': global_importance,
                    'rank': range(1, len(feature_names) + 1)
                }).sort_values('shap_importance', ascending=False).reset_index(drop=True)
                shap_importance_df['rank'] = range(1, len(shap_importance_df) + 1)
                
                shap_importance_df.to_csv(
                    self.global_dir / f'{model_name}_shap_importance.csv', 
                    index=False
                )
                
                # 4. Generate dependence plots for top features
                self.generate_dependence_plots(shap_values, X_display, feature_names, model_name)
                
            except Exception as e:
                self.logger.error(f"SHAP analysis failed for {model_name}: {str(e)}")
        
    def generate_dependence_plots(self, shap_values: np.ndarray, X_display: pd.DataFrame, 
                                feature_names: List[str], model_name: str) -> None:
        """Generate SHAP dependence plots for top features"""
        self.logger.info(f"Generating dependence plots for {model_name}...")
        
        # Get top 8 most important features
        feature_importance = np.abs(shap_values).mean(0)
        top_features_idx = np.argsort(feature_importance)[::-1][:8]
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, feature_idx in enumerate(top_features_idx):
            feature_name = feature_names[feature_idx]
            
            # Find best interaction feature
            try:
                correlations = np.abs(np.corrcoef(X_display.values.T)[feature_idx])
                correlations[feature_idx] = 0  # Exclude self-correlation
                interaction_idx = np.argmax(correlations)
            except:
                # Fallback to no interaction
                interaction_idx = None
            
            plt.sca(axes[i])
            shap.dependence_plot(feature_idx, shap_values, X_display, 
                               feature_names=feature_names, 
                               interaction_index=interaction_idx,
                               show=False, ax=axes[i])
            axes[i].set_title(f'{feature_name}', fontsize=10)
        
        plt.suptitle(f'SHAP Dependence Plots - {model_name.title()}', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.global_dir / f'{model_name}_dependence_plots.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_policy_focused_analysis(self, shap_data: Dict[str, Tuple[pd.DataFrame, pd.Series]]) -> None:
        """Generate policy-focused SHAP analysis"""
        self.logger.info("Generating policy-focused SHAP analysis...")
        
        X_train, y_train = shap_data['train']
        feature_names = X_train.columns.tolist()
        
        # Define policy-relevant feature categories
        policy_categories = {
            'Tree_Canopy': [f for f in feature_names if 'canopy' in f.lower()],
            'Temperature': [f for f in feature_names if any(x in f.lower() for x in ['temp', 'days_in_bin'])],
            'Socioeconomic': [f for f in feature_names if any(x in f.lower() for x in ['gdp', 'population', 'nightlight'])],
            'Climate': [f for f in feature_names if any(x in f.lower() for x in ['prec', 'humidity', 'rhu', 'wind', 'pres'])],
            'Spatial_Temporal': [f for f in feature_names if any(x in f.lower() for x in ['lat', 'lon', 'year', 'month'])]
        }
        
        sample_size = min(1000, len(X_train))
        sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
        X_sample = X_train.iloc[sample_idx]
        
        policy_results = {}
        
        for model_name, model in self.models.items():
            self.logger.info(f"Policy analysis for {model_name}...")
            
            try:
                # Compute SHAP values
                if model_name in self.scalers:
                    X_model = self.scalers[model_name].transform(X_sample)
                    explainer = shap.LinearExplainer(model, X_model)
                    shap_values = explainer.shap_values(X_model)
                else:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample)
                
                # Calculate category-level importance
                category_importance = {}
                category_effects = {}
                
                for category, features in policy_categories.items():
                    category_indices = [i for i, f in enumerate(feature_names) if f in features]
                    if category_indices:
                        # Average absolute SHAP values for this category
                        category_shap = shap_values[:, category_indices]
                        category_importance[category] = float(np.abs(category_shap).mean())
                        
                        # Mean SHAP effect (direction of impact)
                        category_effects[category] = float(np.mean(category_shap))
                
                policy_results[model_name] = {
                    'category_importance': category_importance,
                    'category_effects': category_effects
                }
                
                # Visualize policy category importance
                if category_importance:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                    
                    # Importance plot
                    categories = list(category_importance.keys())
                    importances = list(category_importance.values())
                    colors = ['green', 'red', 'blue', 'orange', 'purple'][:len(categories)]
                    
                    bars1 = ax1.bar(categories, importances, color=colors, alpha=0.7)
                    ax1.set_ylabel('Average Absolute SHAP Value')
                    ax1.set_title('Policy Category Importance')
                    ax1.tick_params(axis='x', rotation=45)
                    
                    # Add value labels
                    for bar, value in zip(bars1, importances):
                        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                               f'{value:.3f}', ha='center', va='bottom')
                    
                    # Effects plot (positive/negative impact)
                    effects = list(category_effects.values())
                    colors_effects = ['red' if x > 0 else 'green' for x in effects]
                    
                    bars2 = ax2.bar(categories, effects, color=colors_effects, alpha=0.7)
                    ax2.set_ylabel('Average SHAP Value (Direction of Effect)')
                    ax2.set_title('Policy Category Effects\n(Red=Increases Energy, Green=Decreases Energy)')
                    ax2.tick_params(axis='x', rotation=45)
                    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    
                    # Add value labels
                    for bar, value in zip(bars2, effects):
                        ax2.text(bar.get_x() + bar.get_width()/2, 
                               bar.get_height() + (0.001 if value >= 0 else -0.003), 
                               f'{value:.3f}', ha='center', 
                               va='bottom' if value >= 0 else 'top')
                    
                    plt.suptitle(f'Policy Analysis - {model_name.title()}', fontsize=16)
                    plt.tight_layout()
                    plt.savefig(self.policy_dir / f'{model_name}_policy_analysis.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                
                # Detailed canopy analysis
                self.analyze_canopy_effects(shap_values, X_sample, feature_names, model_name)
                
            except Exception as e:
                self.logger.error(f"Policy analysis failed for {model_name}: {str(e)}")
        
        # Save policy results summary
        with open(self.policy_dir / 'policy_analysis_summary.json', 'w') as f:
            json.dump(policy_results, f, indent=2)
            
    def analyze_canopy_effects(self, shap_values: np.ndarray, X_sample: pd.DataFrame, 
                              feature_names: List[str], model_name: str) -> None:
        """Detailed analysis of canopy effects on energy consumption"""
        self.logger.info(f"Analyzing canopy effects for {model_name}...")
        
        canopy_features = [f for f in feature_names if 'canopy' in f.lower()]
        if not canopy_features:
            return
        
        canopy_indices = [i for i, f in enumerate(feature_names) if f in canopy_features]
        canopy_shap = shap_values[:, canopy_indices]
        canopy_feature_names = [feature_names[i] for i in canopy_indices]
        
        # Create canopy effects DataFrame
        canopy_df = pd.DataFrame(canopy_shap, columns=canopy_feature_names)
        
        # Calculate energy savings potential
        energy_savings = {}
        for feature in canopy_feature_names:
            negative_effects = canopy_df[feature][canopy_df[feature] < 0]
            if len(negative_effects) > 0:
                # Convert log-scale SHAP values to percentage energy savings
                avg_log_savings = -negative_effects.mean()
                # Approximate percentage savings (exp(log_savings) - 1)
                pct_savings = (np.exp(avg_log_savings) - 1) * 100
                energy_savings[feature] = {
                    'avg_log_reduction': float(avg_log_savings),
                    'approx_pct_savings': float(pct_savings),
                    'samples_with_savings': int(len(negative_effects)),
                    'total_samples': int(len(canopy_df[feature]))
                }
        
        # Visualize canopy effects
        plt.figure(figsize=(14, 8))
        
        # Box plot of canopy SHAP values
        canopy_df.boxplot()
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('SHAP Value (Log Energy Consumption Impact)')
        plt.title(f'Canopy Coverage Effects on Energy Consumption - {model_name.title()}')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No Effect')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.policy_dir / f'{model_name}_canopy_effects_detailed.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save canopy analysis results
        canopy_results = {
            'model': model_name,
            'energy_savings_potential': energy_savings,
            'summary_statistics': {
                'total_canopy_features': int(len(canopy_features)),
                'avg_effect_across_features': float(canopy_df.mean().mean()),
                'features_with_energy_savings': int(len([f for f in energy_savings.keys()]))
            }
        }
        
        with open(self.policy_dir / f'{model_name}_canopy_analysis.json', 'w') as f:
            json.dump(canopy_results, f, indent=2)
    
    def generate_local_explanations(self, shap_data: Dict[str, Tuple[pd.DataFrame, pd.Series]]) -> None:
        """Generate local explanations for individual predictions"""
        self.logger.info("Generating local explanations...")
        
        X_val, y_val = shap_data['validation']
        feature_names = X_val.columns.tolist()
        
        # Select interesting cases for local explanation
        interesting_cases = {
            'high_consumption': y_val.nlargest(5).index,
            'low_consumption': y_val.nsmallest(5).index,
            'median_consumption': y_val.iloc[y_val.argsort()[len(y_val)//2-2:len(y_val)//2+3]].index
        }
        
        best_model_name = list(self.models.keys())[0]  # Use first available model
        if best_model_name in self.models:
            model = self.models[best_model_name]
            
            try:
                # Prepare data
                if best_model_name in self.scalers:
                    X_model = self.scalers[best_model_name].transform(X_val)
                    explainer = shap.LinearExplainer(model, X_model[:100])  # Background sample
                else:
                    explainer = shap.TreeExplainer(model)
                    X_model = X_val.values
                
                for case_type, indices in interesting_cases.items():
                    self.logger.info(f"Generating explanations for {case_type} cases...")
                    
                    for i, idx in enumerate(indices[:3]):  # Limit to 3 cases per type
                        if best_model_name in self.scalers:
                            instance = X_model[idx:idx+1]
                        else:
                            instance = X_val.iloc[idx:idx+1].values
                        
                        shap_values = explainer.shap_values(instance)
                        
                        # Waterfall plot
                        plt.figure(figsize=(10, 8))
                        if hasattr(shap, 'waterfall_plot'):
                            shap.waterfall_plot(explainer.expected_value, shap_values[0], 
                                              X_val.iloc[idx], feature_names=feature_names,
                                              show=False, max_display=15)
                        else:
                            # Fallback: manual waterfall-style plot
                            feature_importance = shap_values[0]
                            sorted_idx = np.argsort(np.abs(feature_importance))[::-1][:15]
                            
                            plt.barh(range(len(sorted_idx)), 
                                   feature_importance[sorted_idx])
                            plt.yticks(range(len(sorted_idx)), 
                                     [feature_names[i] for i in sorted_idx])
                            plt.xlabel('SHAP Value')
                            plt.title(f'Local Explanation - {case_type.title()} Case {i+1}')
                            plt.gca().invert_yaxis()
                        
                        plt.tight_layout()
                        plt.savefig(self.local_dir / f'{case_type}_case_{i+1}_explanation.png', 
                                   dpi=300, bbox_inches='tight')
                        plt.close()
                        
            except Exception as e:
                self.logger.error(f"Local explanation generation failed: {str(e)}")
    
    def create_comprehensive_summary(self) -> None:
        """Create comprehensive summary of SHAP analysis results"""
        self.logger.info("Creating comprehensive SHAP analysis summary...")
        
        # Collect all importance files
        importance_summary = {}
        for model_name in self.models.keys():
            importance_file = self.global_dir / f'{model_name}_shap_importance.csv'
            if importance_file.exists():
                df = pd.read_csv(importance_file)
                importance_summary[model_name] = df.head(10).to_dict('records')
        
        # Create summary report
        summary = {
            'analysis_info': {
                'run_date': datetime.now().isoformat(),
                'models_analyzed': list(self.models.keys()),
                'total_features_analyzed': len(self.models),
                'analysis_methods': ['Global SHAP', 'Dependence Plots', 'Policy Analysis', 'Local Explanations']
            },
            'top_features_by_model': importance_summary,
            'output_directories': {
                'global_explanations': str(self.global_dir),
                'local_explanations': str(self.local_dir), 
                'policy_analysis': str(self.policy_dir),
                'feature_interactions': str(self.interactions_dir)
            }
        }
        
        with open(self.output_dir / 'shap_analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info("SHAP analysis summary created")
    
    def run_complete_shap_analysis(self) -> Dict[str, Any]:
        """Run complete SHAP analysis pipeline"""
        self.logger.info("Starting comprehensive SHAP analysis...")
        
        try:
            # Load models and data
            models = self.load_trained_models()
            if not models:
                raise ValueError("No trained models found")
            
            datasets, selected_features = self.load_data()
            shap_data = self.prepare_shap_data(datasets, selected_features)
            
            # Generate all analyses
            self.generate_global_shap_explanations(shap_data)
            self.generate_policy_focused_analysis(shap_data)
            self.generate_local_explanations(shap_data)
            self.create_comprehensive_summary()
            
            self.logger.info("SHAP analysis completed successfully!")
            
            return {
                'models_analyzed': list(models.keys()),
                'output_directory': self.output_dir,
                'summary_file': self.output_dir / 'shap_analysis_summary.json'
            }
            
        except Exception as e:
            self.logger.error(f"Error in SHAP analysis: {str(e)}")
            raise


def main():
    """Main function for standalone execution"""
    analyzer = CitySHAPAnalyzer()
    results = analyzer.run_complete_shap_analysis()
    
    print(f"\nCity SHAP Analysis Completed Successfully!")
    print(f"Models analyzed: {', '.join(results['models_analyzed'])}")
    print(f"Results saved to: {results['output_directory']}")
    print(f"Summary file: {results['summary_file']}")
    
    return results


if __name__ == "__main__":
    shap_results = main()