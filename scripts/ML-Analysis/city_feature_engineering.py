"""
City-Level Feature Engineering for Urban Climate-Energy Analysis
Creates policy-relevant features from city-level data
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent))
from config_loader import load_ml_analysis_config, load_data_processing_config

class CityFeatureEngineering:
    """Feature engineering for city-level urban climate-energy analysis"""
    
    def __init__(self):
        self.ml_config = load_ml_analysis_config()
        self.processing_config = load_data_processing_config()
        self.data_dir = Path(__file__).parent.parent.parent / "data" / "processed" / "ml_analysis_city"
        self.output_dir = Path(__file__).parent.parent.parent / "data" / "processed" / "ml_analysis_city" / "engineered"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(__file__).parent.parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"city_feature_engineering_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_data_splits(self) -> Dict[str, pd.DataFrame]:
        """Load the prepared data splits"""
        self.logger.info("Loading data splits...")
        
        datasets = {}
        for split in ['train', 'validation', 'test']:
            file_path = self.data_dir / f"{split}_dataset.parquet"
            if file_path.exists():
                datasets[split] = pd.read_parquet(file_path)
                self.logger.info(f"Loaded {split}: {datasets[split].shape}")
            else:
                self.logger.warning(f"Dataset not found: {file_path}")
        
        return datasets
        
    def create_humidity_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create humidity categorical bins following exact same logic as grid-level"""
        self.logger.info("Creating humidity bins...")
        
        if 'rhu' not in df.columns:
            self.logger.warning("Humidity (rhu) column not found")
            return df
            
        humidity_bins_config = self.processing_config.get('humidity_bins', [
            {'name': 'dry', 'min': 0, 'max': 40, 'label': 'Dry (<40%)'},
            {'name': 'moderate', 'min': 40, 'max': 70, 'label': 'Moderate (40-70%)'},
            {'name': 'humid', 'min': 70, 'max': 100, 'label': 'Humid (>70%)'}
        ])
        
        df = df.copy()
        
        # Create humidity categories
        conditions = []
        labels = []
        
        for i, bin_config in enumerate(humidity_bins_config):
            min_val = bin_config['min']
            max_val = bin_config['max']
            category = bin_config['name']
            
            if i == 0:  # First category
                condition = (df['rhu'] >= min_val) & (df['rhu'] < max_val)
            elif i == len(humidity_bins_config) - 1:  # Last category  
                condition = df['rhu'] >= min_val
            else:  # Middle categories
                condition = (df['rhu'] >= min_val) & (df['rhu'] < max_val)
                
            conditions.append(condition)
            labels.append(category)
            
        df['humidity_category'] = np.select(conditions, labels, default='unknown')
        
        # Create dummy variables for interpretability
        humidity_dummies = pd.get_dummies(df['humidity_category'], prefix='humidity')
        df = pd.concat([df, humidity_dummies], axis=1)
        
        self.logger.info(f"Created humidity bins: {labels}")
        self.logger.info(f"Humidity distribution: {df['humidity_category'].value_counts().to_dict()}")
        
        return df
        
    def create_precipitation_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create precipitation categorical bins following exact same logic as grid-level"""
        self.logger.info("Creating precipitation bins...")
        
        if 'prec' not in df.columns:
            self.logger.warning("Precipitation (prec) column not found")
            return df
            
        precipitation_bins_config = self.processing_config.get('precipitation_bins', [
            {'name': 'dry', 'min': 0, 'max': 25, 'label': 'Dry (<25mm)'},
            {'name': 'light', 'min': 25, 'max': 75, 'label': 'Light (25-75mm)'},
            {'name': 'moderate', 'min': 75, 'max': 150, 'label': 'Moderate (75-150mm)'},
            {'name': 'heavy', 'min': 150, 'max': 1000, 'label': 'Heavy (>150mm)'}
        ])
        
        df = df.copy()
        
        # Create precipitation categories
        conditions = []
        labels = []
        
        for i, bin_config in enumerate(precipitation_bins_config):
            min_val = bin_config['min']
            max_val = bin_config['max']
            category = bin_config['name']
            
            if i == len(precipitation_bins_config) - 1:  # Last category
                condition = df['prec'] >= min_val
            else:
                condition = (df['prec'] >= min_val) & (df['prec'] < max_val)
                
            conditions.append(condition)
            labels.append(category)
            
        df['precipitation_category'] = np.select(conditions, labels, default='unknown')
        
        # Create dummy variables for interpretability
        precip_dummies = pd.get_dummies(df['precipitation_category'], prefix='precipitation')
        df = pd.concat([df, precip_dummies], axis=1)
        
        self.logger.info(f"Created precipitation bins: {labels}")
        self.logger.info(f"Precipitation distribution: {df['precipitation_category'].value_counts().to_dict()}")
        
        return df
        
    def create_socioeconomic_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create socioeconomic categorical bins following exact same logic as grid-level"""
        self.logger.info("Creating socioeconomic bins...")
        
        df = df.copy()
        
        # GDP per capita bins (using quantiles for balanced categories)
        if 'gdp_per_capita' in df.columns:
            gdp_quantiles = df['gdp_per_capita'].quantile([0.33, 0.67]).values
            
            gdp_conditions = [
                df['gdp_per_capita'] <= gdp_quantiles[0],
                (df['gdp_per_capita'] > gdp_quantiles[0]) & (df['gdp_per_capita'] <= gdp_quantiles[1]),
                df['gdp_per_capita'] > gdp_quantiles[1]
            ]
            gdp_labels = ['low_gdp', 'medium_gdp', 'high_gdp']
            
            df['gdp_category'] = np.select(gdp_conditions, gdp_labels, default='unknown')
            
            # Create dummy variables
            gdp_dummies = pd.get_dummies(df['gdp_category'], prefix='gdp')
            df = pd.concat([df, gdp_dummies], axis=1)
            
            self.logger.info(f"GDP categories created with quantiles: {gdp_quantiles}")
            
        # Population density bins (using quantiles)
        if 'population' in df.columns:
            pop_quantiles = df['population'].quantile([0.33, 0.67]).values
            
            pop_conditions = [
                df['population'] <= pop_quantiles[0],
                (df['population'] > pop_quantiles[0]) & (df['population'] <= pop_quantiles[1]),
                df['population'] > pop_quantiles[1]
            ]
            pop_labels = ['low_pop', 'medium_pop', 'high_pop']
            
            df['population_category'] = np.select(pop_conditions, pop_labels, default='unknown')
            
            # Create dummy variables
            pop_dummies = pd.get_dummies(df['population_category'], prefix='population')
            df = pd.concat([df, pop_dummies], axis=1)
            
            self.logger.info(f"Population categories created with quantiles: {pop_quantiles}")
            
        # Nightlight bins (using quantiles)
        if 'nightlight' in df.columns:
            # Handle zero nightlight separately (rural areas)
            zero_nightlight = df['nightlight'] == 0
            nonzero_nightlight = df['nightlight'] > 0
            
            if nonzero_nightlight.sum() > 0:
                nightlight_quantiles = df.loc[nonzero_nightlight, 'nightlight'].quantile([0.5]).values
                
                nightlight_conditions = [
                    zero_nightlight,
                    (nonzero_nightlight) & (df['nightlight'] <= nightlight_quantiles[0]),
                    (nonzero_nightlight) & (df['nightlight'] > nightlight_quantiles[0])
                ]
                nightlight_labels = ['rural', 'urban_low', 'urban_high']
                
                df['nightlight_category'] = np.select(nightlight_conditions, nightlight_labels, default='unknown')
                
                # Create dummy variables
                nightlight_dummies = pd.get_dummies(df['nightlight_category'], prefix='nightlight')
                df = pd.concat([df, nightlight_dummies], axis=1)
                
                self.logger.info(f"Nightlight categories created: {nightlight_labels}")
        
        return df
        
    def create_canopy_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create tree canopy categorical bins following exact same logic as grid-level"""
        self.logger.info("Creating canopy bins...")
        
        if 'canopy_share' not in df.columns:
            self.logger.warning("Canopy share column not found")
            return df
            
        df = df.copy()
        
        # Create canopy categories using natural breaks (same as grid-level)
        canopy_conditions = [
            df['canopy_share'] == 0,  # No canopy
            (df['canopy_share'] > 0) & (df['canopy_share'] <= 0.1),  # Low canopy (0-10%)
            (df['canopy_share'] > 0.1) & (df['canopy_share'] <= 0.3),  # Medium canopy (10-30%)
            df['canopy_share'] > 0.3  # High canopy (>30%)
        ]
        canopy_labels = ['no_canopy', 'low_canopy', 'medium_canopy', 'high_canopy']
        
        df['canopy_category'] = np.select(canopy_conditions, canopy_labels, default='unknown')
        
        # Create dummy variables
        canopy_dummies = pd.get_dummies(df['canopy_category'], prefix='canopy')
        df = pd.concat([df, canopy_dummies], axis=1)
        
        self.logger.info(f"Canopy categories: {canopy_labels}")
        self.logger.info(f"Canopy distribution: {df['canopy_category'].value_counts().to_dict()}")
        
        return df
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features"""
        self.logger.info("Creating temporal features...")
        
        df_enhanced = df.copy()
        
        if 'month' in df.columns:
            # Season categories
            season_map = {12: 'winter', 1: 'winter', 2: 'winter',
                         3: 'spring', 4: 'spring', 5: 'spring',
                         6: 'summer', 7: 'summer', 8: 'summer',
                         9: 'autumn', 10: 'autumn', 11: 'autumn'}
            df_enhanced['season'] = df['month'].map(season_map)
            
            # Summer/winter indicators
            df_enhanced['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
            df_enhanced['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
            
            # Peak energy months (summer cooling, winter heating)
            df_enhanced['peak_energy_month'] = df['month'].isin([1, 2, 7, 8]).astype(int)
        
        if 'year' in df.columns:
            # Year since start of data
            df_enhanced['years_since_start'] = df['year'] - df['year'].min()
            
            # Economic development period (pre/post 2015)
            df_enhanced['post_2015'] = (df['year'] >= 2015).astype(int)
        
        self.logger.info(f"Added temporal features: {df_enhanced.shape[1] - df.shape[1]} new features")
        return df_enhanced
        
    def create_two_way_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interpretable two-way interactions (same pattern as grid-level but adapted for city data)"""
        self.logger.info("Creating two-way interactions...")
        
        df = df.copy()
        
        # 1. Canopy × Climate interactions (key for policy analysis)
        if 'canopy_category' in df.columns:
            # Canopy × Humidity
            if 'humidity_category' in df.columns:
                df['canopy_x_humidity'] = df['canopy_category'] + '_' + df['humidity_category']
            
            # Canopy × Precipitation  
            if 'precipitation_category' in df.columns:
                df['canopy_x_precipitation'] = df['canopy_category'] + '_' + df['precipitation_category']
        
        # 2. Socioeconomic × Climate interactions
        if 'gdp_category' in df.columns:
            # GDP × Humidity
            if 'humidity_category' in df.columns:
                df['gdp_x_humidity'] = df['gdp_category'] + '_' + df['humidity_category']
            
            # GDP × Precipitation
            if 'precipitation_category' in df.columns:
                df['gdp_x_precipitation'] = df['gdp_category'] + '_' + df['precipitation_category']
        
        if 'population_category' in df.columns:
            # Population × Humidity
            if 'humidity_category' in df.columns:
                df['population_x_humidity'] = df['population_category'] + '_' + df['humidity_category']
        
        # 3. Canopy × Socioeconomic interactions (for equity analysis)
        if 'canopy_category' in df.columns:
            # Canopy × GDP
            if 'gdp_category' in df.columns:
                df['canopy_x_gdp'] = df['canopy_category'] + '_' + df['gdp_category']
            
            # Canopy × Population
            if 'population_category' in df.columns:
                df['canopy_x_population'] = df['canopy_category'] + '_' + df['population_category']
            
            # Canopy × Nightlight
            if 'nightlight_category' in df.columns:
                df['canopy_x_nightlight'] = df['canopy_category'] + '_' + df['nightlight_category']
        
        # 4. Create dummy variables for all interaction features
        interaction_cols = [col for col in df.columns if '_x_' in col and df[col].dtype == 'object']
        
        for col in interaction_cols:
            # Create dummy variables for each interaction
            interaction_dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, interaction_dummies], axis=1)
        
        self.logger.info("Created two-way interactions with dummy encoding")
        
        return df
        
    def create_policy_relevant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features specifically relevant for policy analysis"""
        self.logger.info("Creating policy-relevant features...")
        
        df_enhanced = df.copy()
        
        # Energy efficiency proxy (electricity per capita)
        if 'elec_kwh' in df.columns and 'population' in df.columns:
            df_enhanced['elec_per_capita'] = df['elec_kwh'] / (df['population'] + 1)  # +1 to avoid division by zero
            df_enhanced['ln_elec_per_capita'] = np.log1p(df_enhanced['elec_per_capita'])
        
        # Economic energy intensity (electricity per GDP)
        if 'elec_kwh' in df.columns and 'gdp_per_capita' in df.columns and 'population' in df.columns:
            total_gdp = df['gdp_per_capita'] * df['population']
            df_enhanced['elec_per_gdp'] = df['elec_kwh'] / (total_gdp + 1)
            df_enhanced['ln_elec_per_gdp'] = np.log1p(df_enhanced['elec_per_gdp'])
        
        # Canopy effectiveness index (canopy share relative to city type)
        if 'canopy_share' in df.columns and 'population_category' in df.columns:
            # Calculate mean canopy by population category
            canopy_by_pop = df.groupby('population_category')['canopy_share'].transform('mean')
            df_enhanced['canopy_effectiveness'] = df['canopy_share'] / (canopy_by_pop + 1)
        
        # Climate vulnerability index
        if all(col in df.columns for col in ['humidity_category', 'precipitation_category', 'pressure_category']):
            # Simple scoring: dry+low_precip = high vulnerability
            vulnerability_score = 0
            vulnerability_score += (df['humidity_category'] == 'dry').astype(int) * 3
            vulnerability_score += (df['precipitation_category'] == 'low').astype(int) * 2
            vulnerability_score += (df['pressure_category'] == 'high').astype(int) * 1
            df_enhanced['climate_vulnerability'] = vulnerability_score
        
        self.logger.info(f"Added policy features: {df_enhanced.shape[1] - df.shape[1]} new features")
        return df_enhanced
        
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables including log transformation"""
        self.logger.info("Creating target variables...")
        
        df = df.copy()
        
        # Create log-transformed target variable (add 1 to avoid log(0))
        if 'elec_kwh' in df.columns:
            df['ln_elec_kwh'] = np.log1p(df['elec_kwh'])
            self.logger.info("Created ln_elec_kwh target variable")
        
        return df
        
    def clean_and_validate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate engineered features"""
        self.logger.info("Cleaning and validating features...")
        
        df_clean = df.copy()
        
        # Handle missing values in categorical features
        categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            df_clean[col] = df_clean[col].fillna('missing')
        
        # Handle missing values in numerical features
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col not in ['elec_kwh', 'ln_elec_kwh']:  # Don't fill target variables
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Remove infinite values
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        
        # Fill any remaining NaN values
        for col in df_clean.columns:
            if col not in ['elec_kwh', 'ln_elec_kwh']:
                if df_clean[col].dtype in ['object', 'category']:
                    df_clean[col] = df_clean[col].fillna('missing')
                else:
                    df_clean[col] = df_clean[col].fillna(0)
        
        self.logger.info(f"Feature cleaning completed: {df_clean.shape}")
        return df_clean
        
    def engineer_features_for_split(self, df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        """Engineer features for a single data split following exact same pattern as grid-level"""
        self.logger.info(f"Engineering features for {split_name} split...")
        
        # Apply all feature engineering steps in same order as grid-level
        df_enhanced = self.create_target_variables(df)
        df_enhanced = self.create_humidity_bins(df_enhanced)
        df_enhanced = self.create_precipitation_bins(df_enhanced)
        df_enhanced = self.create_socioeconomic_bins(df_enhanced)
        df_enhanced = self.create_canopy_bins(df_enhanced)
        # Note: No temperature features for city-level since we don't have temperature bins
        df_enhanced = self.create_two_way_interactions(df_enhanced)
        df_enhanced = self.clean_and_validate_features(df_enhanced)
        
        self.logger.info(f"{split_name} feature engineering completed: {df.shape} → {df_enhanced.shape}")
        return df_enhanced
        
    def engineer_all_features(self) -> Dict[str, pd.DataFrame]:
        """Engineer features for all data splits"""
        self.logger.info("Starting comprehensive feature engineering...")
        
        # Load data splits
        datasets = self.load_data_splits()
        
        # Engineer features for each split
        engineered_datasets = {}
        for split_name, df in datasets.items():
            engineered_df = self.engineer_features_for_split(df, split_name)
            engineered_datasets[split_name] = engineered_df
            
            # Save engineered dataset
            output_file = self.output_dir / f"{split_name}_engineered.parquet"
            engineered_df.to_parquet(output_file, index=False)
            self.logger.info(f"Saved {split_name} engineered dataset: {output_file}")
        
        # Create feature summary
        self.create_feature_summary(engineered_datasets)
        
        self.logger.info("Feature engineering completed for all splits!")
        return engineered_datasets
        
    def create_feature_summary(self, engineered_datasets: Dict[str, pd.DataFrame]) -> None:
        """Create a summary of engineered features"""
        self.logger.info("Creating feature summary...")
        
        # Use training data for feature analysis
        train_df = engineered_datasets['train']
        
        # Categorize features
        feature_categories = {
            'target_variables': [col for col in train_df.columns if 'elec' in col.lower()],
            'climate_features': [col for col in train_df.columns if any(term in col.lower() for term in ['humidity', 'precipitation', 'pressure', 'climate'])],
            'socioeconomic_features': [col for col in train_df.columns if any(term in col.lower() for term in ['gdp', 'population', 'nightlight', 'socioeconomic'])],
            'canopy_features': [col for col in train_df.columns if 'canopy' in col.lower()],
            'temporal_features': [col for col in train_df.columns if any(term in col.lower() for term in ['year', 'month', 'season', 'summer', 'winter'])],
            'interaction_features': [col for col in train_df.columns if '_x_' in col],
            'policy_features': [col for col in train_df.columns if any(term in col.lower() for term in ['per_capita', 'per_gdp', 'effectiveness', 'vulnerability'])],
            'spatial_features': [col for col in train_df.columns if any(term in col.lower() for term in ['city', 'province', 'lat', 'lon'])],
            'other_features': []
        }
        
        # Identify other features
        all_categorized = set()
        for category_features in feature_categories.values():
            all_categorized.update(category_features)
        
        feature_categories['other_features'] = [col for col in train_df.columns if col not in all_categorized]
        
        # Create summary
        summary = {
            'total_features': len(train_df.columns),
            'feature_categories': {
                category: {
                    'count': len(features),
                    'features': features
                }
                for category, features in feature_categories.items()
            },
            'data_shapes': {
                split_name: {
                    'rows': len(df),
                    'columns': len(df.columns)
                }
                for split_name, df in engineered_datasets.items()
            }
        }
        
        # Save summary
        import json
        summary_file = self.output_dir / "feature_engineering_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Feature summary saved: {summary_file}")
        
        # Log summary
        self.logger.info("Feature Engineering Summary:")
        self.logger.info(f"  Total features: {summary['total_features']}")
        for category, info in summary['feature_categories'].items():
            if info['count'] > 0:
                self.logger.info(f"  {category}: {info['count']} features")


def main():
    """Main function for standalone execution"""
    engineer = CityFeatureEngineering()
    engineered_datasets = engineer.engineer_all_features()
    
    print(f"\nCity-Level Feature Engineering Completed!")
    print(f"Engineered datasets:")
    for split_name, df in engineered_datasets.items():
        print(f"  {split_name}: {df.shape[0]:,} rows × {df.shape[1]} features")
    
    return engineered_datasets


if __name__ == "__main__":
    main()