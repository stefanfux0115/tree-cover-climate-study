"""
Grid-Level ML Data Loader - Load the already integrated grid_month_panel dataset
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))
from config_loader import load_ml_analysis_config

class GridMLDataLoader:
    """Load the integrated grid-level dataset for ML analysis"""
    
    def __init__(self):
        self.config = load_ml_analysis_config()
        self.data_dir = Path(__file__).parent.parent.parent / "data" / "processed"
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(__file__).parent.parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"grid_ml_data_loader_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_integrated_grid_data(self) -> pd.DataFrame:
        """Load the already integrated grid_month_panel dataset"""
        self.logger.info("Loading integrated grid-level dataset...")
        
        # Load the pre-integrated dataset
        integrated_file = self.data_dir / "grid_level" / "integrated" / "final" / "grid_month_panel.parquet"
        
        if not integrated_file.exists():
            raise FileNotFoundError(f"Integrated grid data not found: {integrated_file}")
        
        df = pd.read_parquet(integrated_file)
        
        self.logger.info(f"Loaded integrated grid dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
        return df
        
    def analyze_dataset_structure(self, df: pd.DataFrame) -> Dict[str, any]:
        """Analyze the structure of the integrated dataset"""
        self.logger.info("Analyzing dataset structure...")
        
        # Identify column types
        temperature_bins = [col for col in df.columns if 'days_in_bin_' in col and '_x_canopy' not in col]
        temperature_interactions = [col for col in df.columns if 'days_in_bin_' in col and '_x_canopy' in col]
        temperature_derived = [col for col in df.columns if any(x in col for x in ['extreme_', 'moderate_', 'entropy', 'total_days'])]
        climate_vars = [col for col in df.columns if col in ['prec', 'pres', 'rhu', 'wind']]
        socioeconomic_vars = [col for col in df.columns if col in ['gdp_per_capita', 'population', 'nightlight']]
        spatial_vars = [col for col in df.columns if col in ['grid_row', 'grid_col', 'x_albers', 'y_albers', 'lon', 'lat', 'grid_id', 'city_id', 'city_name', 'province']]
        temporal_vars = [col for col in df.columns if col in ['year', 'month', 'year_month']]
        target_vars = [col for col in df.columns if col in ['elec_kwh', 'ln_elec_kwh']]
        canopy_vars = [col for col in df.columns if 'canopy' in col and 'x_canopy' not in col]
        
        structure = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'column_groups': {
                'temperature_bins': temperature_bins,
                'temperature_interactions': temperature_interactions,
                'temperature_derived': temperature_derived,
                'climate_variables': climate_vars,
                'socioeconomic_variables': socioeconomic_vars,
                'canopy_variables': canopy_vars,
                'spatial_variables': spatial_vars,
                'temporal_variables': temporal_vars,
                'target_variables': target_vars
            },
            'column_counts': {
                'temperature_bins': len(temperature_bins),
                'temperature_interactions': len(temperature_interactions),
                'temperature_derived': len(temperature_derived),
                'climate_variables': len(climate_vars),
                'socioeconomic_variables': len(socioeconomic_vars),
                'canopy_variables': len(canopy_vars),
                'spatial_variables': len(spatial_vars),
                'temporal_variables': len(temporal_vars),
                'target_variables': len(target_vars)
            }
        }
        
        # Log structure summary
        self.logger.info("Dataset Structure:")
        self.logger.info(f"  Total: {structure['total_rows']:,} rows × {structure['total_columns']} columns")
        self.logger.info(f"  Temperature bins: {structure['column_counts']['temperature_bins']}")
        self.logger.info(f"  Temperature interactions: {structure['column_counts']['temperature_interactions']}")
        self.logger.info(f"  Temperature derived: {structure['column_counts']['temperature_derived']}")
        self.logger.info(f"  Climate variables: {structure['column_counts']['climate_variables']}")
        self.logger.info(f"  Socioeconomic variables: {structure['column_counts']['socioeconomic_variables']}")
        self.logger.info(f"  Canopy variables: {structure['column_counts']['canopy_variables']}")
        
        return structure
        
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """Validate data quality of the integrated dataset"""
        self.logger.info("Validating data quality...")
        
        validation = {
            'missing_data': {},
            'data_ranges': {},
            'temporal_coverage': {},
            'spatial_coverage': {},
            'data_quality_issues': []
        }
        
        # Check missing data
        missing_counts = df.isnull().sum()
        validation['missing_data'] = {
            col: {
                'count': int(count),
                'percentage': float(count / len(df) * 100)
            }
            for col, count in missing_counts.items() if count > 0
        }
        
        # Check data ranges for key variables
        key_numeric_vars = ['elec_kwh', 'ln_elec_kwh', 'canopy_share', 'gdp_per_capita', 
                           'population', 'nightlight', 'prec', 'pres', 'rhu', 'wind']
        
        for var in key_numeric_vars:
            if var in df.columns:
                validation['data_ranges'][var] = {
                    'min': float(df[var].min()),
                    'max': float(df[var].max()),
                    'mean': float(df[var].mean()),
                    'std': float(df[var].std()),
                    'q25': float(df[var].quantile(0.25)),
                    'q75': float(df[var].quantile(0.75))
                }
        
        # Check temporal coverage
        if 'year' in df.columns and 'month' in df.columns:
            validation['temporal_coverage'] = {
                'year_range': [int(df['year'].min()), int(df['year'].max())],
                'months': sorted(df['month'].unique().tolist()),
                'total_year_months': len(df[['year', 'month']].drop_duplicates()),
                'observations_per_year': df.groupby('year').size().to_dict()
            }
            
        # Check spatial coverage
        if 'grid_id' in df.columns:
            validation['spatial_coverage'] = {
                'unique_grids': int(df['grid_id'].nunique()),
                'unique_cities': int(df['city_id'].nunique()) if 'city_id' in df.columns else None,
                'unique_provinces': int(df['province'].nunique()) if 'province' in df.columns else None,
                'observations_per_grid': {
                    'mean': float(df.groupby('grid_id').size().mean()),
                    'std': float(df.groupby('grid_id').size().std())
                }
            }
        
        # Check for data quality issues
        # Negative electricity consumption
        if 'elec_kwh' in df.columns:
            negative_elec = (df['elec_kwh'] < 0).sum()
            if negative_elec > 0:
                validation['data_quality_issues'].append(f"Negative electricity values: {negative_elec}")
        
        # Zero electricity with high GDP/population
        if all(col in df.columns for col in ['elec_kwh', 'gdp_per_capita', 'population']):
            zero_elec_high_gdp = ((df['elec_kwh'] == 0) & (df['gdp_per_capita'] > df['gdp_per_capita'].quantile(0.75))).sum()
            if zero_elec_high_gdp > 0:
                validation['data_quality_issues'].append(f"Zero electricity with high GDP: {zero_elec_high_gdp}")
        
        # Temperature bins sum consistency (should sum to ~30 days per month)
        temp_bins = [col for col in df.columns if 'days_in_bin_' in col and '_x_canopy' not in col]
        if temp_bins:
            temp_sum = df[temp_bins].sum(axis=1)
            unusual_temp_sums = ((temp_sum < 20) | (temp_sum > 40)).sum()
            if unusual_temp_sums > 0:
                validation['data_quality_issues'].append(f"Unusual temperature bin sums: {unusual_temp_sums}")
        
        # Log validation summary
        self.logger.info("Data Quality Validation:")
        self.logger.info(f"  Missing data columns: {len(validation['missing_data'])}")
        self.logger.info(f"  Temporal coverage: {validation['temporal_coverage']['year_range']}")
        self.logger.info(f"  Spatial coverage: {validation['spatial_coverage']['unique_grids']:,} grids")
        self.logger.info(f"  Data quality issues: {len(validation['data_quality_issues'])}")
        
        return validation
        
    def prepare_for_ml(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Prepare data splits for ML analysis based on config"""
        self.logger.info("Preparing data splits for ML analysis...")
        
        data_splits_config = self.config.get('data_splits', {})
        train_years = data_splits_config.get('train_years', [2012, 2013, 2014, 2015, 2016, 2017])
        val_years = data_splits_config.get('validation_years', [2018])
        test_years = data_splits_config.get('test_years', [2019])
        
        # Create temporal splits
        train_df = df[df['year'].isin(train_years)].copy()
        val_df = df[df['year'].isin(val_years)].copy()
        test_df = df[df['year'].isin(test_years)].copy()
        
        splits = {
            'full': df,
            'train': train_df,
            'validation': val_df,
            'test': test_df
        }
        
        # Log split sizes
        self.logger.info("Data splits created:")
        for split_name, split_df in splits.items():
            memory_mb = split_df.memory_usage(deep=True).sum() / 1024**2
            self.logger.info(f"  {split_name}: {len(split_df):,} rows ({memory_mb:.1f} MB)")
            
        return splits
        
    def save_ml_ready_data(self, data_splits: Dict[str, pd.DataFrame], 
                          structure: Dict, validation: Dict) -> None:
        """Save ML-ready datasets and metadata"""
        output_dir = Path(__file__).parent.parent.parent / "data" / "processed" / "ml_analysis_grid"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save data splits
        for split_name, split_df in data_splits.items():
            output_file = output_dir / f"{split_name}_dataset.parquet"
            split_df.to_parquet(output_file, index=False)
            self.logger.info(f"Saved {split_name} dataset: {output_file}")
        
        # Save metadata
        import json
        metadata = {
            'dataset_info': {
                'total_observations': len(data_splits['full']),
                'temporal_range': validation['temporal_coverage']['year_range'],
                'spatial_coverage': validation['spatial_coverage']['unique_grids'],
                'features_count': len(data_splits['full'].columns) - 2,  # Excluding target variables
                'created_at': datetime.now().isoformat(),
                'level': 'grid'
            },
            'data_structure': structure,
            'data_validation': validation,
            'data_splits': {
                split_name: len(split_df) for split_name, split_df in data_splits.items()
            }
        }
        
        metadata_file = output_dir / "grid_ml_dataset_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved metadata: {metadata_file}")
        
    def load_and_prepare_ml_data(self) -> Dict[str, pd.DataFrame]:
        """Main method to load and prepare data for ML analysis"""
        self.logger.info("Starting grid-level ML data preparation...")
        
        try:
            # Load grid-level integrated dataset
            df = self.load_integrated_grid_data()
            
            # Analyze structure
            structure = self.analyze_dataset_structure(df)
            
            # Validate data quality
            validation = self.validate_data_quality(df)
            
            # Prepare data splits
            data_splits = self.prepare_for_ml(df)
            
            # Save ML-ready data
            self.save_ml_ready_data(data_splits, structure, validation)
            
            self.logger.info("Grid-level ML data preparation completed successfully!")
            return data_splits
            
        except Exception as e:
            self.logger.error(f"Error during grid-level ML data preparation: {str(e)}")
            raise


def main():
    """Main function for standalone execution"""
    loader = GridMLDataLoader()
    data_splits = loader.load_and_prepare_ml_data()
    
    print(f"\nGrid-Level ML-ready datasets prepared:")
    for split_name, split_df in data_splits.items():
        memory_mb = split_df.memory_usage(deep=True).sum() / 1024**2
        print(f"  {split_name}: {split_df.shape[0]:,} rows × {split_df.shape[1]} cols ({memory_mb:.1f} MB)")
    
    print("\nGrid-level datasets ready for ML analysis!")
    return data_splits


if __name__ == "__main__":
    grid_ml_datasets = main()