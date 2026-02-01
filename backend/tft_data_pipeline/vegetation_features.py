"""
Vegetation Feature Processing Module

Extracts static vegetation/fuel features from FIA (Forest Inventory and Analysis)
plot-based data and assigns them to fires via spatial joins.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from scipy.spatial import cKDTree
from shapely.geometry import Point


def load_vegetation_data(vegetation_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load all vegetation CSV files from the FIA dataset.
    
    Files expected:
        - CA_PLOT.csv: Plot locations and basic info
        - CA_COND.csv: Stand conditions (forest type, age, etc.)
        - CA_GRND_CVR.csv: Ground cover percentages
        - CA_SEEDLING.csv: Seedling counts
        - CA_P2VEG_SUBPLOT_SPP.csv: Vegetation species
    
    Args:
        vegetation_dir: Directory containing vegetation CSV files
        
    Returns:
        Dictionary mapping filename stems to DataFrames
    """
    vegetation_dir = Path(vegetation_dir)
    datasets = {}
    
    expected_files = [
        'CA_PLOT.csv',
        'CA_COND.csv', 
        'CA_GRND_CVR.csv',
        'CA_SEEDLING.csv',
        'CA_P2VEG_SUBPLOT_SPP.csv'
    ]
    
    for filename in expected_files:
        filepath = vegetation_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath, low_memory=False)
            key = filepath.stem  # e.g., "CA_PLOT"
            datasets[key] = df
            print(f"Loaded {key}: {len(df)} records")
        else:
            print(f"Warning: {filename} not found")
    
    return datasets


def create_vegetation_lookup(
    vegetation_data: Dict[str, pd.DataFrame],
    max_year: int = 2020
) -> gpd.GeoDataFrame:
    """
    Create a spatial vegetation lookup table from FIA data.
    
    This function:
    1. Joins plot locations with condition data
    2. Aggregates ground cover by plot
    3. Creates a GeoDataFrame with plot points and features
    
    Args:
        vegetation_data: Dictionary of vegetation DataFrames
        max_year: Maximum inventory year to include (use most recent)
        
    Returns:
        GeoDataFrame with plot locations and vegetation features
    """
    # Start with plot locations
    if 'CA_PLOT' not in vegetation_data:
        raise ValueError("CA_PLOT data required for vegetation lookup")
    
    plots = vegetation_data['CA_PLOT'].copy()
    
    # Filter to plots with valid coordinates
    plots = plots[(plots['LAT'].notna()) & (plots['LON'].notna())]
    plots = plots[(plots['LAT'] != 0) & (plots['LON'] != 0)]
    
    # Use most recent inventory per plot
    plots = plots.sort_values('INVYR', ascending=False)
    plots = plots.drop_duplicates(subset=['LAT', 'LON'], keep='first')
    
    # Filter to reasonable years
    plots = plots[plots['INVYR'] <= max_year]
    
    print(f"Using {len(plots)} unique plot locations")
    
    # Create GeoDataFrame
    geometry = [Point(lon, lat) for lon, lat in zip(plots['LON'], plots['LAT'])]
    plots_gdf = gpd.GeoDataFrame(plots, geometry=geometry, crs="EPSG:4326")
    
    # Join with condition data for forest type
    if 'CA_COND' in vegetation_data:
        cond = vegetation_data['CA_COND'].copy()
        
        # Get primary condition per plot (CONDID=1)
        cond_primary = cond[cond['CONDID'] == 1].copy()
        
        # Select relevant columns
        cond_cols = ['PLT_CN', 'FORTYPCD', 'STDAGE', 'SITECLCD', 'SLOPE', 
                     'BALIVE', 'DSTRBCD1', 'CARBON_LITTER', 'LIVE_CANOPY_CVR_PCT']
        cond_cols = [c for c in cond_cols if c in cond_primary.columns]
        cond_subset = cond_primary[cond_cols].drop_duplicates(subset=['PLT_CN'])
        
        # Join to plots
        plots_gdf = plots_gdf.merge(cond_subset, left_on='CN', right_on='PLT_CN', how='left')
    
    # Join with ground cover data
    if 'CA_GRND_CVR' in vegetation_data:
        grnd = vegetation_data['CA_GRND_CVR'].copy()
        
        # Aggregate ground cover by type per plot
        grnd_pivot = grnd.pivot_table(
            index='PLT_CN',
            columns='GRND_CVR_TYP',
            values='CVR_PCT',
            aggfunc='mean'
        ).reset_index()
        
        # Rename columns to be descriptive
        cover_cols = {
            'LITT': 'litter_cover_pct',
            'BARE': 'bare_cover_pct', 
            'WOOD': 'wood_cover_pct',
            'ROCK': 'rock_cover_pct',
            'MOSS': 'moss_cover_pct'
        }
        grnd_pivot = grnd_pivot.rename(columns=cover_cols)
        
        # Join to plots
        plots_gdf = plots_gdf.merge(grnd_pivot, left_on='CN', right_on='PLT_CN', 
                                     how='left', suffixes=('', '_grnd'))
    
    # Compute vegetation density score (simplified)
    plots_gdf['vegetation_density'] = compute_vegetation_density(plots_gdf)
    
    # Clean up and select final columns
    final_cols = ['CN', 'LAT', 'LON', 'ELEV', 'INVYR', 'geometry',
                  'FORTYPCD', 'STDAGE', 'SLOPE', 'BALIVE', 
                  'litter_cover_pct', 'bare_cover_pct', 'wood_cover_pct',
                  'vegetation_density', 'LIVE_CANOPY_CVR_PCT']
    
    available_cols = [c for c in final_cols if c in plots_gdf.columns]
    plots_gdf = plots_gdf[available_cols].copy()
    
    # Rename for clarity
    rename_map = {
        'LAT': 'plot_lat',
        'LON': 'plot_lon',
        'ELEV': 'elevation_ft',
        'FORTYPCD': 'forest_type_code',
        'STDAGE': 'stand_age',
        'SLOPE': 'slope_pct',
        'BALIVE': 'basal_area_live',
        'LIVE_CANOPY_CVR_PCT': 'canopy_cover_pct'
    }
    plots_gdf = plots_gdf.rename(columns=rename_map)
    
    print(f"Created vegetation lookup with {len(plots_gdf)} plots")
    return plots_gdf


def compute_vegetation_density(df: pd.DataFrame) -> pd.Series:
    """
    Compute a vegetation density score from available metrics.
    
    Score combines:
    - Litter cover (fuel load proxy)
    - Basal area (tree density)
    - Canopy cover
    
    Normalized to 0-100 scale.
    
    Args:
        df: DataFrame with vegetation columns
        
    Returns:
        Series with vegetation density scores
    """
    density = np.zeros(len(df))
    n_components = 0
    
    # Litter cover contribution (0-100%)
    if 'litter_cover_pct' in df.columns:
        litter = df['litter_cover_pct'].fillna(0).clip(0, 100)
        density += litter * 0.4  # 40% weight
        n_components += 1
    
    # Basal area contribution (typically 0-300 sq ft/acre)
    if 'BALIVE' in df.columns:
        basal = df['BALIVE'].fillna(0).clip(0, 300)
        basal_norm = basal / 300 * 100  # Normalize to 0-100
        density += basal_norm * 0.4  # 40% weight
        n_components += 1
    
    # Canopy cover contribution
    if 'LIVE_CANOPY_CVR_PCT' in df.columns:
        canopy = df['LIVE_CANOPY_CVR_PCT'].fillna(0).clip(0, 100)
        density += canopy * 0.2  # 20% weight
        n_components += 1
    
    # Normalize if we have components
    if n_components > 0:
        # Already weighted to sum to ~100 max
        density = density.clip(0, 100)
    
    return pd.Series(density, index=df.index)


def assign_vegetation_features(
    fire_df: pd.DataFrame,
    veg_lookup: gpd.GeoDataFrame,
    search_radius_km: float = 10.0,
    n_nearest: int = 5
) -> pd.DataFrame:
    """
    Assign vegetation features to fires via spatial join.
    
    For each fire, finds nearby vegetation plots and aggregates their features.
    Uses K-nearest neighbors within a search radius.
    
    Args:
        fire_df: Fire DataFrame with centroid_lat, centroid_lon
        veg_lookup: Vegetation GeoDataFrame with plot locations
        search_radius_km: Maximum distance to search for plots (km)
        n_nearest: Number of nearest plots to average
        
    Returns:
        Fire DataFrame with added vegetation columns
    """
    fire_df = fire_df.copy()
    
    # Build KD-tree for vegetation plots
    veg_coords = np.column_stack([
        veg_lookup['plot_lon'].values,
        veg_lookup['plot_lat'].values
    ])
    
    # Convert search radius from km to degrees (approximate)
    search_radius_deg = search_radius_km / 111.0  # ~111 km per degree
    
    tree = cKDTree(veg_coords)
    
    # Get unique fire centroids
    fire_centroids = fire_df[['fire_uid', 'centroid_lon', 'centroid_lat']].drop_duplicates()
    
    # Vegetation features to aggregate
    veg_features = ['vegetation_density', 'litter_cover_pct', 'bare_cover_pct',
                    'slope_pct', 'elevation_ft', 'forest_type_code', 
                    'stand_age', 'basal_area_live', 'canopy_cover_pct']
    veg_features = [f for f in veg_features if f in veg_lookup.columns]
    
    # Find nearest plots for each fire
    veg_assignments = []
    
    for _, row in fire_centroids.iterrows():
        fire_coord = np.array([[row['centroid_lon'], row['centroid_lat']]])
        
        # Query K nearest within radius
        distances, indices = tree.query(fire_coord, k=min(n_nearest, len(veg_lookup)))
        
        # Filter by radius
        mask = distances[0] < search_radius_deg
        valid_indices = indices[0][mask]
        
        if len(valid_indices) > 0:
            # Get vegetation values from nearest plots
            nearby_veg = veg_lookup.iloc[valid_indices]
            
            # Aggregate (mean for numeric, mode for categorical)
            veg_values = {'fire_uid': row['fire_uid']}
            
            for feat in veg_features:
                if feat in nearby_veg.columns:
                    vals = nearby_veg[feat].dropna()
                    if len(vals) > 0:
                        if feat == 'forest_type_code':
                            # Mode for categorical
                            veg_values[feat] = vals.mode().iloc[0] if len(vals.mode()) > 0 else vals.iloc[0]
                        else:
                            # Mean for numeric
                            veg_values[feat] = vals.mean()
            
            veg_values['n_veg_plots'] = len(valid_indices)
            veg_assignments.append(veg_values)
        else:
            # No nearby plots - use defaults
            veg_values = {'fire_uid': row['fire_uid'], 'n_veg_plots': 0}
            veg_assignments.append(veg_values)
    
    # Create assignment DataFrame
    veg_df = pd.DataFrame(veg_assignments)
    
    # Merge with fire data
    fire_df = fire_df.merge(veg_df, on='fire_uid', how='left')
    
    # Fill missing vegetation with defaults
    fill_defaults = {
        'vegetation_density': 50.0,  # Medium density
        'litter_cover_pct': 30.0,
        'bare_cover_pct': 10.0,
        'slope_pct': 15.0,
        'elevation_ft': 3000.0,
        'forest_type_code': 999,  # Unknown
        'n_veg_plots': 0
    }
    
    for col, default in fill_defaults.items():
        if col in fire_df.columns:
            fire_df[col] = fire_df[col].fillna(default)
    
    n_matched = (fire_df['n_veg_plots'] > 0).sum() if 'n_veg_plots' in fire_df.columns else 0
    print(f"Assigned vegetation features: {n_matched}/{len(fire_centroids)} fires matched")
    
    return fire_df


def get_forest_type_categories() -> Dict[int, str]:
    """
    Get forest type code to name mapping (FIA codes).
    
    Returns:
        Dictionary mapping forest type codes to descriptions
    """
    # Common California forest types
    return {
        101: 'Jack pine',
        102: 'Red pine',
        103: 'Eastern white pine',
        104: 'Eastern white pine / hemlock',
        105: 'Eastern hemlock',
        121: 'Ponderosa pine',
        122: 'Jeffrey pine',
        123: 'Sugar pine',
        124: 'Western white pine',
        141: 'Douglas-fir',
        161: 'Western juniper',
        162: 'Rocky Mountain juniper',
        181: 'Pinyon / juniper woodland',
        182: 'Pinyon pine',
        201: 'Douglas-fir',
        221: 'Engelmann spruce',
        224: 'Blue spruce',
        241: 'Western redcedar',
        261: 'White fir',
        262: 'Red fir',
        263: 'Noble fir',
        264: 'Pacific silver fir',
        265: 'Grand fir',
        266: 'Subalpine fir',
        281: 'Incense-cedar',
        301: 'Western larch',
        321: 'Lodgepole pine',
        341: 'Redwood',
        361: 'Port-Orford-cedar',
        371: 'California mixed conifer',
        381: 'Sitka spruce',
        501: 'Post oak / blackjack oak',
        506: 'Blue oak',
        507: 'Coast live oak',
        508: 'Canyon live oak / interior live oak',
        509: 'Gray pine',
        701: 'Black ash / American elm / red maple',
        921: 'California laurel',
        922: 'Giant chinkapin',
        923: 'Pacific madrone',
        924: 'Tanoak',
        931: 'California oakwoods',
        932: 'California oak woodland',
        933: 'California oak / grass',
        934: 'Mixed California hardwood',
        999: 'Unknown / Nonstocked'
    }
