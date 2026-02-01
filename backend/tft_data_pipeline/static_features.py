"""
Static Features Module

Extracts static (time-invariant) features from final fire perimeters
and other metadata sources.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List


def load_final_perimeters(perimeter_file: Path) -> gpd.GeoDataFrame:
    """
    Load final fire perimeter data.
    
    Expected columns:
        - fireID: Unique fire identifier
        - year: Year of fire
        - farea: Final fire area (km²)
        - clat, clon: Centroid coordinates
        - tst, ted: Start and end timestamps
        - geometry: Final perimeter polygon
    
    Args:
        perimeter_file: Path to GeoPackage file
        
    Returns:
        GeoDataFrame with final perimeter information
    """
    perimeter_file = Path(perimeter_file)
    
    if not perimeter_file.exists():
        raise FileNotFoundError(f"Perimeter file not found: {perimeter_file}")
    
    gdf = gpd.read_file(perimeter_file)
    print(f"Loaded {len(gdf)} final fire perimeters")
    
    # Standardize column names
    if 'fireID' in gdf.columns:
        gdf['fire_uid'] = gdf['fireID'].astype(str)
    
    # Ensure CRS
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    
    return gdf


def extract_static_features(
    perimeters_gdf: gpd.GeoDataFrame,
    fire_timeseries: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Extract static (time-invariant) features for each fire.
    
    Args:
        perimeters_gdf: GeoDataFrame with final perimeters
        fire_timeseries: Optional fire time series to join
        
    Returns:
        DataFrame with static features per fire_uid
    """
    static_features = []
    
    for _, row in perimeters_gdf.iterrows():
        fire_uid = str(row.get('fireID', row.get('fire_uid', '')))
        
        features = {
            'fire_uid': fire_uid,
        }
        
        # Year and temporal features
        if 'year' in row and pd.notna(row['year']):
            features['start_year'] = int(row['year'])
        
        # Parse start date for month
        if 'tst' in row and pd.notna(row['tst']):
            try:
                start_str = str(row['tst'])[:8]  # YYYYMMDD
                features['start_month'] = int(start_str[4:6])
                features['start_day'] = int(start_str[6:8])
            except:
                pass
        
        # Final fire area
        if 'farea' in row and pd.notna(row['farea']):
            features['final_area_km2'] = float(row['farea'])
        elif row['geometry'] is not None:
            # Compute from geometry
            gdf_temp = gpd.GeoDataFrame([row], crs="EPSG:4326").to_crs("ESRI:102003")
            features['final_area_km2'] = gdf_temp.geometry.area.iloc[0] / 1e6
        
        # Centroid coordinates
        if 'clat' in row and pd.notna(row['clat']):
            features['fire_centroid_lat'] = float(row['clat'])
        if 'clon' in row and pd.notna(row['clon']):
            features['fire_centroid_lon'] = float(row['clon'])
        
        # Fire duration (if start and end available)
        if 'tst' in row and 'ted' in row:
            try:
                from datetime import datetime
                start = datetime.strptime(str(row['tst'])[:8], '%Y%m%d')
                end = datetime.strptime(str(row['ted'])[:8], '%Y%m%d')
                features['fire_duration_days'] = (end - start).days + 1
            except:
                pass
        
        # Compute additional geometry features from final perimeter
        if row['geometry'] is not None:
            geom = row['geometry']
            
            # Final perimeter length
            gdf_proj = gpd.GeoDataFrame(
                [{'geometry': geom}], crs="EPSG:4326"
            ).to_crs("ESRI:102003")
            features['final_perimeter_km'] = gdf_proj.geometry.length.iloc[0] / 1e3
            
            # Shape complexity
            if features.get('final_area_km2', 0) > 0:
                features['final_compactness'] = (
                    4 * np.pi * features['final_area_km2']
                ) / (features['final_perimeter_km'] ** 2)
                features['final_compactness'] = min(features['final_compactness'], 1.0)
            
            # Bounding box aspect ratio
            minx, miny, maxx, maxy = geom.bounds
            width = maxx - minx
            height = maxy - miny
            if width > 0 and height > 0:
                features['bbox_aspect_ratio'] = max(width, height) / min(width, height)
        
        static_features.append(features)
    
    df = pd.DataFrame(static_features)
    print(f"Extracted static features for {len(df)} fires")
    
    return df


def compute_fire_size_category(final_area_km2: pd.Series) -> pd.Series:
    """
    Categorize fires by size class.
    
    Size classes based on NWCG standards:
        A: < 0.1 km² (< 0.25 acres)
        B: 0.1-0.4 km²
        C: 0.4-4 km²
        D: 4-40 km²
        E: 40-120 km²
        F: 120-2000 km²
        G: > 2000 km²
    
    Args:
        final_area_km2: Series of fire areas
        
    Returns:
        Categorical series with size classes
    """
    bins = [0, 0.1, 0.4, 4, 40, 120, 2000, np.inf]
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    
    return pd.cut(final_area_km2, bins=bins, labels=labels, include_lowest=True)


def compute_fire_region(lat: pd.Series, lon: pd.Series) -> pd.Series:
    """
    Assign fires to California regions based on coordinates.
    
    Simplified regions:
        - Northern CA: lat > 39
        - Central CA: 36 < lat <= 39
        - Southern CA: lat <= 36
    
    Args:
        lat: Latitude series
        lon: Longitude series
        
    Returns:
        Categorical series with region codes
    """
    regions = []
    for la in lat:
        if pd.isna(la):
            regions.append('Unknown')
        elif la > 39:
            regions.append('Northern')
        elif la > 36:
            regions.append('Central')
        else:
            regions.append('Southern')
    
    return pd.Series(regions)


def merge_static_to_timeseries(
    timeseries_df: pd.DataFrame,
    static_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge static features to time series data.
    
    Args:
        timeseries_df: Time series DataFrame with fire_uid
        static_df: Static features DataFrame with fire_uid
        
    Returns:
        Merged DataFrame
    """
    # Ensure fire_uid is string in both
    timeseries_df['fire_uid'] = timeseries_df['fire_uid'].astype(str)
    static_df['fire_uid'] = static_df['fire_uid'].astype(str)
    
    # Merge
    merged = timeseries_df.merge(static_df, on='fire_uid', how='left')
    
    # Report merge success
    n_matched = merged[static_df.columns.drop('fire_uid')[0]].notna().sum() if len(static_df.columns) > 1 else 0
    n_total = len(timeseries_df)
    print(f"Merged static features: {n_matched}/{n_total} records matched")
    
    return merged
