"""
Enhanced Feature Engineering for ML Analysis
Creates interpretable features with humidity/precipitation binning, 
socioeconomic categorization, and two-way interactions only
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))
from config_loader import load_ml_analysis_config, load_data_processing_config

class FeatureEngineer:
    """Enhanced feature engineering for urban climate-energy ML analysis"""
    
    def __init__(self):
        self.ml_config = load_ml_analysis_config()
        self.processing_config = load_data_processing_config()
        self.data_dir = Path(__file__).parent.parent.parent / "data" / "processed"
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(__file__).parent.parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"feature_engineering_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_ml_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load the ML-ready datasets"""
        self.logger.info("Loading ML datasets...")
        
        ml_dir = self.data_dir / "ml_analysis_grid"
        datasets = {}
        
        for split in ['train', 'validation', 'test', 'full']:
            file_path = ml_dir / f"{split}_dataset.parquet"
            if file_path.exists():
                datasets[split] = pd.read_parquet(file_path)
                self.logger.info(f"Loaded {split}: {datasets[split].shape[0]:,} rows")
            else:
                self.logger.warning(f"Dataset not found: {file_path}")
                
        return datasets
        
    def create_humidity_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create humidity categorical bins"""
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
        """Create precipitation categorical bins"""
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
        """Create socioeconomic categorical bins"""
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
        """Create tree canopy categorical bins"""
        self.logger.info("Creating canopy bins...")
        
        if 'canopy_share' not in df.columns:
            self.logger.warning("Canopy share column not found")
            return df
            
        df = df.copy()
        
        # Create canopy categories using natural breaks
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
        
    def create_temperature_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced temperature features from bins"""
        self.logger.info("Creating enhanced temperature features...")
        
        df = df.copy()
        
        # Get temperature bin columns
        temp_bin_cols = [col for col in df.columns if 'days_in_bin_' in col and '_x_canopy' not in col]
        
        if not temp_bin_cols:
            self.logger.warning("No temperature bin columns found")
            return df
            
        # Create temperature aggregates
        df['total_days'] = df[temp_bin_cols].sum(axis=1)
        
        # Extreme temperature days
        cold_bins = [col for col in temp_bin_cols if any(x in col for x in ['below_neg10', 'neg10_neg5', 'neg5_0'])]
        hot_bins = [col for col in temp_bin_cols if any(x in col for x in ['30_35', '35_40', '40_plus'])]
        moderate_bins = [col for col in temp_bin_cols if any(x in col for x in ['15_20', '20_25', '25_30'])]
        
        if cold_bins:
            df['extreme_cold_days'] = df[cold_bins].sum(axis=1)
            df['extreme_cold_share'] = df['extreme_cold_days'] / (df['total_days'] + 1e-6)
            
        if hot_bins:
            df['extreme_hot_days'] = df[hot_bins].sum(axis=1)
            df['extreme_hot_share'] = df['extreme_hot_days'] / (df['total_days'] + 1e-6)
            
        if moderate_bins:
            df['moderate_temp_days'] = df[moderate_bins].sum(axis=1)
            df['moderate_temp_share'] = df['moderate_temp_days'] / (df['total_days'] + 1e-6)
        
        # Temperature variability (entropy-like measure)
        temp_shares = df[temp_bin_cols].div(df['total_days'] + 1e-6, axis=0)
        temp_shares = temp_shares.replace(0, 1e-6)  # Avoid log(0)
        df['temperature_entropy'] = -(temp_shares * np.log(temp_shares)).sum(axis=1)
        
        self.logger.info("Created enhanced temperature features")
        
        return df
        
    def create_two_way_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interpretable two-way interactions (avoiding 3-4 way interactions)"""
        self.logger.info("Creating two-way interactions...")
        
        df = df.copy()
        
        # 1. Temperature × Socioeconomic interactions
        if 'extreme_hot_share' in df.columns:
            # Hot days × GDP level
            if 'gdp_high' in df.columns:
                df['hot_days_x_high_gdp'] = df['extreme_hot_share'] * df['gdp_high']
            if 'gdp_low' in df.columns:
                df['hot_days_x_low_gdp'] = df['extreme_hot_share'] * df['gdp_low']
                
            # Hot days × Population density
            if 'population_high' in df.columns:
                df['hot_days_x_high_pop'] = df['extreme_hot_share'] * df['population_high']
            if 'population_low' in df.columns:
                df['hot_days_x_low_pop'] = df['extreme_hot_share'] * df['population_low']
                
            # Hot days × Urban type
            if 'nightlight_urban_high' in df.columns:
                df['hot_days_x_urban_high'] = df['extreme_hot_share'] * df['nightlight_urban_high']
            if 'nightlight_rural' in df.columns:
                df['hot_days_x_rural'] = df['extreme_hot_share'] * df['nightlight_rural']
        
        if 'extreme_cold_share' in df.columns:
            # Cold days × GDP level
            if 'gdp_high' in df.columns:
                df['cold_days_x_high_gdp'] = df['extreme_cold_share'] * df['gdp_high']
            if 'gdp_low' in df.columns:
                df['cold_days_x_low_gdp'] = df['extreme_cold_share'] * df['gdp_low']
        
        # 2. Climate × Socioeconomic interactions
        if 'humidity_humid' in df.columns and 'gdp_high' in df.columns:
            df['humid_x_high_gdp'] = df['humidity_humid'] * df['gdp_high']
        if 'humidity_dry' in df.columns and 'gdp_low' in df.columns:
            df['dry_x_low_gdp'] = df['humidity_dry'] * df['gdp_low']
            
        if 'precipitation_heavy' in df.columns and 'population_high' in df.columns:
            df['heavy_rain_x_high_pop'] = df['precipitation_heavy'] * df['population_high']
        if 'precipitation_dry' in df.columns and 'nightlight_rural' in df.columns:
            df['dry_rain_x_rural'] = df['precipitation_dry'] * df['nightlight_rural']
        
        # 3. Socioeconomic × Socioeconomic interactions
        if 'gdp_high' in df.columns and 'population_high' in df.columns:
            df['high_gdp_x_high_pop'] = df['gdp_high'] * df['population_high']
        if 'gdp_low' in df.columns and 'population_low' in df.columns:
            df['low_gdp_x_low_pop'] = df['gdp_low'] * df['population_low']
            
        if 'nightlight_urban_high' in df.columns and 'gdp_high' in df.columns:
            df['urban_high_x_gdp_high'] = df['nightlight_urban_high'] * df['gdp_high']
        if 'nightlight_rural' in df.columns and 'gdp_low' in df.columns:
            df['rural_x_gdp_low'] = df['nightlight_rural'] * df['gdp_low']
        
        # 4. Canopy × Climate interactions (already exist as temperature×canopy)
        # Enhanced canopy interactions with new climate features
        if 'canopy_high' in df.columns:
            if 'humidity_humid' in df.columns:
                df['high_canopy_x_humid'] = df['canopy_high'] * df['humidity_humid']
            if 'precipitation_heavy' in df.columns:
                df['high_canopy_x_heavy_rain'] = df['canopy_high'] * df['precipitation_heavy']
                
        if 'canopy_low' in df.columns:
            if 'humidity_dry' in df.columns:
                df['low_canopy_x_dry'] = df['canopy_low'] * df['humidity_dry']
            if 'extreme_hot_share' in df.columns:
                df['low_canopy_x_hot_days'] = df['canopy_low'] * df['extreme_hot_share']
        
        # Count new interaction features
        interaction_cols = [col for col in df.columns if '_x_' in col and col not in df.columns[:47]]
        self.logger.info(f"Created {len(interaction_cols)} two-way interaction features")
        
        return df
        
    def create_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create seasonal features"""
        self.logger.info("Creating seasonal features...")
        
        if 'month' not in df.columns:
            self.logger.warning("Month column not found")
            return df
            
        df = df.copy()
        
        # Create seasonal categories
        season_map = {
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring', 
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'autumn', 10: 'autumn', 11: 'autumn'
        }
        
        df['season'] = df['month'].map(season_map)
        
        # Create seasonal dummy variables
        season_dummies = pd.get_dummies(df['season'], prefix='season')
        df = pd.concat([df, season_dummies], axis=1)
        
        # Create cooling/heating season indicators
        df['cooling_season'] = (df['month'].isin([6, 7, 8, 9])).astype(int)  # Jun-Sep
        df['heating_season'] = (df['month'].isin([12, 1, 2, 3])).astype(int)  # Dec-Mar
        
        self.logger.info("Created seasonal features")
        
        return df
        
    def remove_redundant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove redundant categorical columns after creating dummies"""
        self.logger.info("Removing redundant categorical columns...")
        
        categorical_cols_to_remove = [
            'humidity_category', 'precipitation_category', 'gdp_category',
            'population_category', 'nightlight_category', 'canopy_category', 'season'
        ]
        
        cols_to_remove = [col for col in categorical_cols_to_remove if col in df.columns]
        df = df.drop(columns=cols_to_remove)
        
        self.logger.info(f"Removed {len(cols_to_remove)} categorical columns")
        
        return df
        
    def validate_engineered_features(self, df: pd.DataFrame) -> Dict[str, any]:
        """Validate the engineered features"""
        self.logger.info("Validating engineered features...")
        
        validation = {
            'total_features': len(df.columns),
            'feature_groups': {},
            'missing_data': {},
            'feature_ranges': {},
            'correlation_warnings': []
        }
        
        # Categorize features
        feature_groups = {
            'original': [col for col in df.columns if not any(x in col for x in ['_x_', 'extreme_', 'moderate_', 'entropy', 'season_', 'heating_', 'cooling_'])],
            'temperature_derived': [col for col in df.columns if any(x in col for x in ['extreme_', 'moderate_', 'entropy'])],
            'categorical_dummies': [col for col in df.columns if any(x in col for x in ['humidity_', 'precipitation_', 'gdp_', 'population_', 'nightlight_', 'canopy_', 'season_'])],
            'interactions': [col for col in df.columns if '_x_' in col],
            'seasonal': [col for col in df.columns if any(x in col for x in ['season_', 'heating_', 'cooling_'])]
        }
        
        for group, cols in feature_groups.items():
            validation['feature_groups'][group] = len(cols)
            
        # Check missing data
        missing_counts = df.isnull().sum()
        validation['missing_data'] = {
            col: int(count) for col, count in missing_counts.items() if count > 0
        }
        
        # Check feature ranges for new features
        new_features = [col for col in df.columns if any(x in col for x in ['extreme_', 'moderate_', 'entropy', '_x_'])]
        numeric_new_features = df[new_features].select_dtypes(include=[np.number]).columns
        
        for col in numeric_new_features:
            validation['feature_ranges'][col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'std': float(df[col].std())
            }
        
        # Log validation summary
        self.logger.info("Feature Engineering Validation:")
        self.logger.info(f"  Total features: {validation['total_features']}")
        for group, count in validation['feature_groups'].items():
            self.logger.info(f"  {group}: {count} features")
        self.logger.info(f"  Missing data columns: {len(validation['missing_data'])}")
        
        return validation
        
    def engineer_features_for_split(self, df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        """Apply feature engineering to a single data split"""
        self.logger.info(f"Engineering features for {split_name} split...")
        
        # Apply all feature engineering steps
        df = self.create_humidity_bins(df)
        df = self.create_precipitation_bins(df)
        df = self.create_socioeconomic_bins(df)
        df = self.create_canopy_bins(df)
        df = self.create_temperature_features(df)
        df = self.create_seasonal_features(df)
        df = self.create_two_way_interactions(df)
        df = self.remove_redundant_features(df)
        
        self.logger.info(f"Feature engineering completed for {split_name}: {df.shape[1]} features")
        
        return df
        
    def save_engineered_datasets(self, datasets: Dict[str, pd.DataFrame], validation: Dict) -> None:
        """Save the feature-engineered datasets"""
        output_dir = self.data_dir / "ml_analysis_grid" / "engineered"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save engineered datasets
        for split_name, df in datasets.items():
            output_file = output_dir / f"{split_name}_engineered.parquet"
            df.to_parquet(output_file, index=False)
            self.logger.info(f"Saved {split_name} engineered dataset: {output_file}")
        
        # Save feature engineering metadata
        import json
        metadata = {
            'feature_engineering_info': {
                'total_features': validation['total_features'],
                'feature_groups': validation['feature_groups'],
                'engineering_date': datetime.now().isoformat()
            },
            'feature_validation': validation,
            'dataset_sizes': {
                split_name: len(df) for split_name, df in datasets.items()
            }
        }
        
        metadata_file = output_dir / "feature_engineering_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved feature engineering metadata: {metadata_file}")
        
        # Save feature list for reference (use train dataset as representative)
        representative_dataset = datasets['train']
        feature_list = {
            'all_features': list(representative_dataset.columns),
            'feature_groups': {
                group: [col for col in representative_dataset.columns 
                       if group == 'original' or any(keyword in col for keyword in {
                           'temperature_derived': ['extreme_', 'moderate_', 'entropy'],
                           'categorical_dummies': ['humidity_', 'precipitation_', 'gdp_', 'population_', 'nightlight_', 'canopy_'],
                           'interactions': ['_x_'],
                           'seasonal': ['season_', 'heating_', 'cooling_']
                       }.get(group, []))]
                for group in validation['feature_groups'].keys()
            }
        }
        
        feature_file = output_dir / "feature_list.json"
        with open(feature_file, 'w', encoding='utf-8') as f:
            json.dump(feature_list, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved feature list: {feature_file}")
        
    def run_feature_engineering(self) -> Dict[str, pd.DataFrame]:
        """Main method to run complete feature engineering pipeline"""
        self.logger.info("Starting comprehensive feature engineering...")
        
        try:
            # Load ML datasets (skip full dataset to save memory)
            datasets = self.load_ml_datasets()
            if 'full' in datasets:
                del datasets['full']  # Remove full dataset to save memory
            
            # Apply feature engineering to each split
            engineered_datasets = {}
            split_names = list(datasets.keys())  # Create a copy of keys to avoid iteration issues
            for split_name in split_names:
                self.logger.info(f"Processing {split_name} split...")
                df = datasets[split_name]
                engineered_datasets[split_name] = self.engineer_features_for_split(df, split_name)
                # Clear original dataset from memory
                del datasets[split_name]
            
            # Validate engineered features (using train dataset as representative)
            validation = self.validate_engineered_features(engineered_datasets['train'])
            
            # Save engineered datasets
            self.save_engineered_datasets(engineered_datasets, validation)
            
            self.logger.info("Feature engineering completed successfully!")
            return engineered_datasets
            
        except Exception as e:
            self.logger.error(f"Error during feature engineering: {str(e)}")
            raise


def main():
    """Main function for standalone execution"""
    engineer = FeatureEngineer()
    engineered_datasets = engineer.run_feature_engineering()
    
    print(f"\nFeature-engineered datasets created:")
    for split_name, df in engineered_datasets.items():
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        print(f"  {split_name}: {df.shape[0]:,} rows × {df.shape[1]} features ({memory_mb:.1f} MB)")
    
    print("\nDatasets ready for ML model training!")
    return engineered_datasets


if __name__ == "__main__":
    engineered_data = main()