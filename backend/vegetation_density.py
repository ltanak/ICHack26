"""
Vegetation Density Calculator for FIA (Forest Inventory and Analysis) Data

This script calculates a composite vegetation density metric using:
- Live canopy cover percentage
- Basal area of live trees (BALIVE)
- Tree density (trees per acre)
- Understory vegetation cover

The vegetation density score is normalized to 0-100 scale.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_datasets(data_dir: Path, year_min: int = 2017, year_max: int = 2020) -> dict:
    """Load all required FIA datasets, filtered to specified year range."""
    datasets = {}
    
    print(f"Loading datasets (filtering to {year_min}-{year_max})...")
    
    # Plot data - contains coordinates
    datasets['plot'] = pd.read_csv(
        data_dir / 'CA_PLOT.csv',
        usecols=['CN', 'INVYR', 'STATECD', 'COUNTYCD', 'PLOT', 'LAT', 'LON', 'ELEV'],
        low_memory=False
    )
    # Filter to year range early
    datasets['plot'] = datasets['plot'][
        (datasets['plot']['INVYR'] >= year_min) & (datasets['plot']['INVYR'] <= year_max)
    ]
    print(f"  Loaded CA_PLOT.csv: {len(datasets['plot'])} records")
    
    # Condition data - contains canopy cover, basal area
    datasets['cond'] = pd.read_csv(
        data_dir / 'CA_COND.csv',
        usecols=['CN', 'PLT_CN', 'INVYR', 'CONDID', 'COND_STATUS_CD', 
                 'LIVE_CANOPY_CVR_PCT', 'BALIVE', 'CONDPROP_UNADJ',
                 'FORTYPCD', 'STDAGE', 'STDSZCD', 'CARBON_UNDERSTORY_AG'],
        low_memory=False
    )
    datasets['cond'] = datasets['cond'][
        (datasets['cond']['INVYR'] >= year_min) & (datasets['cond']['INVYR'] <= year_max)
    ]
    print(f"  Loaded CA_COND.csv: {len(datasets['cond'])} records")
    
    # Vegetation subplot species - contains cover percentages by layer
    datasets['veg_spp'] = pd.read_csv(
        data_dir / 'CA_P2VEG_SUBPLOT_SPP.csv',
        usecols=['PLT_CN', 'INVYR', 'SUBP', 'CONDID', 'LAYER', 'COVER_PCT', 'GROWTH_HABIT_CD'],
        low_memory=False
    )
    datasets['veg_spp'] = datasets['veg_spp'][
        (datasets['veg_spp']['INVYR'] >= year_min) & (datasets['veg_spp']['INVYR'] <= year_max)
    ]
    print(f"  Loaded CA_P2VEG_SUBPLOT_SPP.csv: {len(datasets['veg_spp'])} records")
    
    # Seedling data - tree regeneration density
    datasets['seedling'] = pd.read_csv(
        data_dir / 'CA_SEEDLING.csv',
        usecols=['PLT_CN', 'INVYR', 'SUBP', 'CONDID', 'TPA_UNADJ', 'STOCKING'],
        low_memory=False
    )
    datasets['seedling'] = datasets['seedling'][
        (datasets['seedling']['INVYR'] >= year_min) & (datasets['seedling']['INVYR'] <= year_max)
    ]
    print(f"  Loaded CA_SEEDLING.csv: {len(datasets['seedling'])} records")
    
    # Ground cover data
    datasets['grnd_cvr'] = pd.read_csv(
        data_dir / 'CA_GRND_CVR.csv',
        usecols=['PLT_CN', 'INVYR', 'SUBP', 'CVR_PCT', 'GRND_CVR_TYP'],
        low_memory=False
    )
    datasets['grnd_cvr'] = datasets['grnd_cvr'][
        (datasets['grnd_cvr']['INVYR'] >= year_min) & (datasets['grnd_cvr']['INVYR'] <= year_max)
    ]
    print(f"  Loaded CA_GRND_CVR.csv: {len(datasets['grnd_cvr'])} records")
    
    return datasets


def calculate_canopy_density(cond_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate canopy density metrics from condition data.
    
    Uses:
    - LIVE_CANOPY_CVR_PCT: Percentage of plot covered by live tree canopy
    - BALIVE: Basal area of live trees (sq ft/acre)
    """
    # Filter to forested conditions (COND_STATUS_CD = 1 means accessible forest land)
    forest_cond = cond_df[cond_df['COND_STATUS_CD'] == 1].copy()
    
    # Aggregate by plot
    canopy_metrics = forest_cond.groupby('PLT_CN').agg({
        'LIVE_CANOPY_CVR_PCT': 'mean',
        'BALIVE': 'sum',  # Total basal area across conditions
        'CONDPROP_UNADJ': 'sum',  # Proportion of plot that's forested
        'CARBON_UNDERSTORY_AG': 'sum'
    }).reset_index()
    
    canopy_metrics.columns = ['PLT_CN', 'canopy_cover_pct', 'basal_area', 
                              'forest_proportion', 'understory_carbon']
    
    return canopy_metrics


def calculate_understory_density(veg_spp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate understory vegetation density from P2VEG data.
    
    Layers in FIA:
    1 = 0 to 2 feet (ground layer)
    2 = 2 to 6 feet (low shrub)
    3 = 6 to 16 feet (tall shrub)
    4 = 16+ feet (subcanopy/canopy)
    """
    # Calculate mean cover by layer
    layer_cover = veg_spp_df.groupby(['PLT_CN', 'LAYER'])['COVER_PCT'].mean().unstack(fill_value=0)
    layer_cover = layer_cover.reset_index()
    
    # Rename columns
    layer_names = {1: 'ground_layer_cover', 2: 'low_shrub_cover', 
                   3: 'tall_shrub_cover', 4: 'subcanopy_cover'}
    layer_cover = layer_cover.rename(columns=layer_names)
    
    # Fill missing layers with 0
    for col in layer_names.values():
        if col not in layer_cover.columns:
            layer_cover[col] = 0
    
    # Calculate total understory cover (layers 1-3)
    understory_cols = ['ground_layer_cover', 'low_shrub_cover', 'tall_shrub_cover']
    existing_cols = [c for c in understory_cols if c in layer_cover.columns]
    layer_cover['total_understory_cover'] = layer_cover[existing_cols].sum(axis=1)
    
    return layer_cover[['PLT_CN'] + existing_cols + ['total_understory_cover']]


def calculate_regeneration_density(seedling_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate tree regeneration density from seedling data.
    
    TPA_UNADJ: Trees per acre (unadjusted)
    STOCKING: Stocking percentage
    """
    regen_metrics = seedling_df.groupby('PLT_CN').agg({
        'TPA_UNADJ': 'sum',  # Total seedlings per acre
        'STOCKING': 'mean'   # Average stocking
    }).reset_index()
    
    regen_metrics.columns = ['PLT_CN', 'seedling_tpa', 'stocking_pct']
    
    return regen_metrics


def calculate_ground_vegetation(grnd_cvr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate ground vegetation cover from ground cover transect data.
    
    GRND_CVR_TYP categories:
    - LITT: Litter
    - MOSS: Moss
    - LICH: Lichen
    - FORB: Forbs/herbs
    - GRAS: Grass
    - SHRB: Shrubs
    - ROCK: Rock
    - BARE: Bare soil
    - WATE: Water
    """
    # Filter to vegetation types (exclude rock, bare, water)
    veg_types = ['LITT', 'MOSS', 'LICH', 'FORB', 'GRAS', 'SHRB']
    veg_cover = grnd_cvr_df[grnd_cvr_df['GRND_CVR_TYP'].isin(veg_types)]
    
    # Calculate total vegetation ground cover by plot
    ground_veg = veg_cover.groupby('PLT_CN')['CVR_PCT'].sum().reset_index()
    ground_veg.columns = ['PLT_CN', 'ground_veg_cover']
    
    return ground_veg


def normalize_score(series: pd.Series, min_val: float = 0, max_val: float = None) -> pd.Series:
    """Normalize a series to 0-100 scale using percentile-based normalization."""
    if max_val is None:
        max_val = series.quantile(0.95)  # Use 95th percentile to reduce outlier impact
    
    normalized = ((series - min_val) / (max_val - min_val)) * 100
    return normalized.clip(0, 100)


def calculate_vegetation_density(data_dir: str | Path, output_path: str | Path = None,
                                  year_min: int = 2017, year_max: int = 2020) -> pd.DataFrame:
    """
    Calculate comprehensive vegetation density score for each plot.
    
    The vegetation density score combines:
    - Canopy cover (30%)
    - Basal area (25%)
    - Understory cover (20%)
    - Seedling/regeneration density (15%)
    - Ground vegetation cover (10%)
    
    Args:
        data_dir: Path to vegetation datasets
        output_path: Optional path to save results CSV
        year_min: Minimum year to include (default 2017)
        year_max: Maximum year to include (default 2020)
    
    Returns a DataFrame with plot coordinates and vegetation density score.
    """
    data_dir = Path(data_dir)
    
    # Load all datasets (filtered to year range)
    datasets = load_datasets(data_dir, year_min=year_min, year_max=year_max)
    
    print("\nCalculating density metrics...")
    
    # Calculate component metrics
    canopy_metrics = calculate_canopy_density(datasets['cond'])
    print(f"  Canopy metrics: {len(canopy_metrics)} plots")
    
    understory_metrics = calculate_understory_density(datasets['veg_spp'])
    print(f"  Understory metrics: {len(understory_metrics)} plots")
    
    regen_metrics = calculate_regeneration_density(datasets['seedling'])
    print(f"  Regeneration metrics: {len(regen_metrics)} plots")
    
    ground_metrics = calculate_ground_vegetation(datasets['grnd_cvr'])
    print(f"  Ground cover metrics: {len(ground_metrics)} plots")
    
    # Start with plot data (has coordinates)
    result = datasets['plot'][['CN', 'INVYR', 'LAT', 'LON', 'ELEV', 'COUNTYCD']].copy()
    result = result.rename(columns={'CN': 'PLT_CN'})
    
    # Merge all metrics
    result = result.merge(canopy_metrics, on='PLT_CN', how='left')
    result = result.merge(understory_metrics, on='PLT_CN', how='left')
    result = result.merge(regen_metrics, on='PLT_CN', how='left')
    result = result.merge(ground_metrics, on='PLT_CN', how='left')
    
    # Fill NaN with 0 for plots without certain data
    fill_cols = ['canopy_cover_pct', 'basal_area', 'forest_proportion', 
                 'total_understory_cover', 'seedling_tpa', 'stocking_pct', 
                 'ground_veg_cover']
    for col in fill_cols:
        if col in result.columns:
            result[col] = result[col].fillna(0)
    
    print("\nNormalizing scores...")
    
    # Normalize each component to 0-100 scale
    result['canopy_score'] = normalize_score(result['canopy_cover_pct'], max_val=100)
    result['basal_score'] = normalize_score(result['basal_area'], max_val=400)  # 400 sq ft/acre is very dense
    result['understory_score'] = normalize_score(result['total_understory_cover'], max_val=150)
    result['regen_score'] = normalize_score(result['seedling_tpa'], max_val=500)
    result['ground_score'] = normalize_score(result['ground_veg_cover'], max_val=100)
    
    # Calculate composite vegetation density score (weighted average)
    result['vegetation_density'] = (
        result['canopy_score'] * 0.30 +
        result['basal_score'] * 0.25 +
        result['understory_score'] * 0.20 +
        result['regen_score'] * 0.15 +
        result['ground_score'] * 0.10
    )
    
    # Round scores
    score_cols = ['canopy_score', 'basal_score', 'understory_score', 
                  'regen_score', 'ground_score', 'vegetation_density']
    result[score_cols] = result[score_cols].round(2)
    
    # Filter to plots with valid coordinates
    result = result.dropna(subset=['LAT', 'LON'])
    
    # Keep all years (don't deduplicate) for time-series analysis
    result = result.sort_values(['LAT', 'LON', 'INVYR'])
    
    print(f"\nFinal dataset: {len(result)} plot-year records")
    print(f"Year range: {result['INVYR'].min()} - {result['INVYR'].max()}")
    print(f"Unique plot locations: {result.drop_duplicates(subset=['LAT', 'LON']).shape[0]}")
    print(f"Vegetation density range: {result['vegetation_density'].min():.1f} - {result['vegetation_density'].max():.1f}")
    print(f"Mean vegetation density: {result['vegetation_density'].mean():.1f}")
    
    # Select output columns
    output_cols = ['PLT_CN', 'INVYR', 'LAT', 'LON', 'ELEV', 'COUNTYCD',
                   'canopy_cover_pct', 'basal_area', 'total_understory_cover',
                   'seedling_tpa', 'ground_veg_cover', 'vegetation_density']
    result = result[output_cols]
    
    # Save to file if path provided
    if output_path:
        output_path = Path(output_path)
        result.to_csv(output_path, index=False)
        print(f"\nSaved results to {output_path}")
    
    return result


def get_high_density_areas(density_df: pd.DataFrame, year: int = None, 
                           threshold: float = 70.0, 
                           percentile: float = None) -> np.ndarray:
    """
    Identify highly dense vegetation areas for a given year.
    
    Args:
        density_df: DataFrame from calculate_vegetation_density()
        year: Specific year to filter (None = all years in dataset)
        threshold: Minimum vegetation density score to be considered "high density" (default 70)
        percentile: Alternative to threshold - use top X percentile (e.g., 90 for top 10%)
                   If provided, overrides threshold
    
    Returns:
        numpy array of shape (N, 2) containing [latitude, longitude] coordinates
        of high density vegetation areas
    """
    # Filter by year if specified
    if year is not None:
        data = density_df[density_df['INVYR'] == year].copy()
        if len(data) == 0:
            print(f"Warning: No data found for year {year}")
            return np.array([]).reshape(0, 2)
    else:
        data = density_df.copy()
    
    # Determine threshold
    if percentile is not None:
        threshold = data['vegetation_density'].quantile(percentile / 100)
        print(f"Using {percentile}th percentile threshold: {threshold:.1f}")
    
    # Filter to high density areas
    high_density = data[data['vegetation_density'] >= threshold]
    
    # Extract coordinates as numpy array
    coords = high_density[['LAT', 'LON']].values
    
    print(f"Found {len(coords)} high density areas (threshold >= {threshold:.1f})")
    if len(coords) > 0:
        print(f"  Density range: {high_density['vegetation_density'].min():.1f} - {high_density['vegetation_density'].max():.1f}")
        print(f"  Lat range: {coords[:, 0].min():.4f} to {coords[:, 0].max():.4f}")
        print(f"  Lon range: {coords[:, 1].min():.4f} to {coords[:, 1].max():.4f}")

        print(f"the coords: {coords}")

    return coords


def get_high_density_areas_detailed(density_df: pd.DataFrame, year: int = None,
                                    threshold: float = 70.0,
                                    percentile: float = None) -> pd.DataFrame:
    """
    Get detailed info for highly dense vegetation areas.
    
    Same as get_high_density_areas but returns full DataFrame with all columns.
    
    Args:
        density_df: DataFrame from calculate_vegetation_density()
        year: Specific year to filter (None = all years in dataset)
        threshold: Minimum vegetation density score (default 70)
        percentile: Alternative - use top X percentile (overrides threshold if provided)
    
    Returns:
        DataFrame with all columns for high density areas, sorted by density descending
    """
    # Filter by year if specified
    if year is not None:
        data = density_df[density_df['INVYR'] == year].copy()
        if len(data) == 0:
            print(f"Warning: No data found for year {year}")
            return pd.DataFrame()
    else:
        data = density_df.copy()
    
    # Determine threshold
    if percentile is not None:
        threshold = data['vegetation_density'].quantile(percentile / 100)
    
    # Filter and sort
    high_density = data[data['vegetation_density'] >= threshold].copy()
    high_density = high_density.sort_values('vegetation_density', ascending=False)
    
    return high_density


def calculate_min_separation(coords: np.ndarray, sample_size: int = None) -> dict:
    """
    Calculate the minimum separation distance between any two vegetation points.
    
    Uses haversine formula for accurate distance on Earth's surface.
    
    Args:
        coords: numpy array of shape (N, 2) containing [latitude, longitude]
        sample_size: Optional - if provided, randomly sample this many points
                    for faster computation on large datasets
    
    Returns:
        Dictionary with:
        - min_distance_km: Minimum distance between any two points in kilometers
        - min_distance_deg: Approximate minimum distance in degrees (for map scaling)
        - avg_distance_km: Average distance to nearest neighbor
        - point1: [lat, lon] of first point in closest pair
        - point2: [lat, lon] of second point in closest pair
    """
    if len(coords) < 2:
        return {
            'min_distance_km': None,
            'min_distance_deg': None,
            'avg_distance_km': None,
            'point1': None,
            'point2': None
        }
    
    # Sample if dataset is large
    if sample_size and len(coords) > sample_size:
        indices = np.random.choice(len(coords), sample_size, replace=False)
        coords = coords[indices]
        print(f"Sampled {sample_size} points for distance calculation")
    
    def haversine(lat1, lon1, lat2, lon2):
        """Calculate haversine distance in kilometers."""
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    n = len(coords)
    min_dist = float('inf')
    min_pair = (None, None)
    nearest_distances = []
    
    # Calculate pairwise distances
    for i in range(n):
        nearest_to_i = float('inf')
        for j in range(n):
            if i != j:
                dist = haversine(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])
                if dist < nearest_to_i:
                    nearest_to_i = dist
                if dist < min_dist:
                    min_dist = dist
                    min_pair = (coords[i].tolist(), coords[j].tolist())
        nearest_distances.append(nearest_to_i)
    
    avg_nearest = np.mean(nearest_distances)
    
    # Convert km to approximate degrees (rough: 1 degree ≈ 111 km at equator)
    min_dist_deg = min_dist / 111.0
    
    result = {
        'min_distance_km': round(min_dist, 3),
        'min_distance_deg': round(min_dist_deg, 6),
        'avg_distance_km': round(avg_nearest, 3),
        'point1': min_pair[0],
        'point2': min_pair[1],
        'n_points': n
    }
    
    print(f"Minimum separation: {min_dist:.3f} km ({min_dist_deg:.6f} degrees)")
    print(f"Average nearest neighbor distance: {avg_nearest:.3f} km")
    
    return result


def save_dense_areas_npz(density_df: pd.DataFrame, year: int, threshold: float = 10.0, 
                         output_dir: str | Path = None) -> Path:
    """
    Save dense vegetation areas as .npz file with coordinates and radius.
    
    Args:
        density_df: DataFrame from calculate_vegetation_density()
        year: Year to filter for
        threshold: Minimum density threshold
        output_dir: Directory to save .npz file (default: same as script)
    
    Returns:
        Path to saved .npz file
        
    Saves:
        - 'coordinates': numpy array of shape (N, 2) with [lat, lon] for each area
        - 'radius_km': minimum separation distance in kilometers
        - 'radius_deg': minimum separation distance in degrees
        - 'metadata': dict with year, threshold, n_points info
    """
    # Get high density areas for the year
    coords = get_high_density_areas(density_df, year=year, threshold=threshold)
    
    if len(coords) < 2:
        print(f"Warning: Only {len(coords)} dense areas found, cannot calculate meaningful radius")
        radius_km = 1.0  # Default 1km radius
        radius_deg = radius_km / 111.0
    else:
        # Calculate minimum separation to use as radius
        sep_info = calculate_min_separation(coords, sample_size=500)
        radius_km = sep_info['min_distance_km'] / 2  # Half the min separation
        radius_deg = sep_info['min_distance_deg'] / 2
    
    # Prepare output directory
    if output_dir is None:
        output_dir = Path(__file__).parent / 'output' / 'vegetation_data'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output filename
    output_file = output_dir / f'dense_areas_{year}_t{threshold:.0f}.npz'
    
    # Metadata
    metadata = {
        'year': year,
        'threshold': threshold,
        'n_points': len(coords),
        'radius_km': radius_km,
        'radius_deg': radius_deg,
        'description': f'Dense vegetation areas for {year} with density >= {threshold}'
    }
    
    # Save to .npz file
    np.savez(
        output_file,
        coordinates=coords,
        radius_km=np.array([radius_km]),
        radius_deg=np.array([radius_deg]),
        metadata=np.array([metadata], dtype=object)
    )
    
    print(f"Saved {len(coords)} dense areas to {output_file}")
    print(f"  Radius: {radius_km:.3f} km ({radius_deg:.6f} degrees)")
    
    return output_file


def get_density_by_location(density_df: pd.DataFrame, lat: float, lon: float, 
                            radius_deg: float = 0.1) -> dict:
    """
    Get vegetation density for a specific location (average of nearby plots).
    
    Args:
        density_df: DataFrame from calculate_vegetation_density()
        lat: Latitude
        lon: Longitude
        radius_deg: Search radius in degrees (~11km per 0.1 degree)
    
    Returns:
        Dictionary with location stats
    """
    # Find plots within radius
    mask = (
        (density_df['LAT'] >= lat - radius_deg) &
        (density_df['LAT'] <= lat + radius_deg) &
        (density_df['LON'] >= lon - radius_deg) &
        (density_df['LON'] <= lon + radius_deg)
    )
    nearby = density_df[mask]
    
    if len(nearby) == 0:
        return {'n_plots': 0, 'vegetation_density': None}
    
    return {
        'n_plots': len(nearby),
        'vegetation_density': nearby['vegetation_density'].mean(),
        'min_density': nearby['vegetation_density'].min(),
        'max_density': nearby['vegetation_density'].max(),
        'avg_canopy_cover': nearby['canopy_cover_pct'].mean(),
        'avg_basal_area': nearby['basal_area'].mean()
    }


def get_density_timeseries(density_df: pd.DataFrame, lat: float, lon: float,
                           radius_deg: float = 0.1, location_name: str = None) -> pd.DataFrame:
    """
    Get vegetation density time-series for a specific location.
    
    Args:
        density_df: DataFrame from calculate_vegetation_density()
        lat: Latitude
        lon: Longitude
        radius_deg: Search radius in degrees (~11km per 0.1 degree)
        location_name: Optional name for the location
    
    Returns:
        DataFrame with yearly vegetation density stats
    """
    # Find plots within radius
    mask = (
        (density_df['LAT'] >= lat - radius_deg) &
        (density_df['LAT'] <= lat + radius_deg) &
        (density_df['LON'] >= lon - radius_deg) &
        (density_df['LON'] <= lon + radius_deg)
    )
    nearby = density_df[mask]
    
    if len(nearby) == 0:
        print(f"No plots found within {radius_deg} degrees of ({lat}, {lon})")
        return pd.DataFrame()
    
    # Aggregate by year
    yearly_stats = nearby.groupby('INVYR').agg({
        'vegetation_density': ['mean', 'std', 'min', 'max', 'count'],
        'canopy_cover_pct': 'mean',
        'basal_area': 'mean',
        'total_understory_cover': 'mean',
        'seedling_tpa': 'mean',
        'ground_veg_cover': 'mean'
    }).reset_index()
    
    # Flatten column names
    yearly_stats.columns = [
        'year', 'density_mean', 'density_std', 'density_min', 'density_max', 'n_plots',
        'canopy_cover_pct', 'basal_area', 'understory_cover', 'seedling_tpa', 'ground_veg_cover'
    ]
    
    # Round values
    for col in yearly_stats.columns:
        if col not in ['year', 'n_plots']:
            yearly_stats[col] = yearly_stats[col].round(2)
    
    if location_name:
        yearly_stats['location'] = location_name
    
    return yearly_stats


def print_timeseries_summary(timeseries_df: pd.DataFrame, location_name: str = "Location"):
    """Print a formatted summary of vegetation density over time."""
    if timeseries_df.empty:
        print(f"No data available for {location_name}")
        return
    
    print(f"\n{'='*70}")
    print(f"VEGETATION DENSITY TIME-SERIES: {location_name}")
    print(f"{'='*70}")
    print(f"{'Year':<8} {'Density':<12} {'Std Dev':<10} {'Range':<16} {'Plots':<8} {'Canopy%':<10}")
    print(f"{'-'*70}")
    
    for _, row in timeseries_df.iterrows():
        density_range = f"{row['density_min']:.1f}-{row['density_max']:.1f}"
        std_str = f"±{row['density_std']:.1f}" if pd.notna(row['density_std']) else "N/A"
        print(f"{int(row['year']):<8} {row['density_mean']:<12.1f} {std_str:<10} {density_range:<16} {int(row['n_plots']):<8} {row['canopy_cover_pct']:<10.1f}")
    
    print(f"{'-'*70}")
    
    # Calculate trend
    if len(timeseries_df) >= 2:
        first_year = timeseries_df.iloc[0]
        last_year = timeseries_df.iloc[-1]
        change = last_year['density_mean'] - first_year['density_mean']
        pct_change = (change / first_year['density_mean']) * 100 if first_year['density_mean'] > 0 else 0
        years_span = last_year['year'] - first_year['year']
        
        trend = "↑ INCREASING" if change > 0 else "↓ DECREASING" if change < 0 else "→ STABLE"
        print(f"\nTrend ({int(first_year['year'])}-{int(last_year['year'])}): {trend}")
        print(f"Change: {change:+.1f} points ({pct_change:+.1f}%) over {int(years_span)} years")
        if years_span > 0:
            print(f"Annual rate: {change/years_span:+.2f} points/year")


if __name__ == '__main__':
    # Run the vegetation density calculation
    data_dir = Path(__file__).parent / 'Datasets' / 'vegetation'
    output_path = Path(__file__).parent / 'output' / 'vegetation_data' / 'vegetation_density.csv'
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate vegetation density
    density_df = calculate_vegetation_density(data_dir, output_path)
    
    # Example: Get density for a specific location (Los Angeles area)
    la_stats = get_density_by_location(density_df, lat=34.05, lon=-118.25, radius_deg=0.5)
    print(f"\nLos Angeles area vegetation stats: {la_stats}")
    
    # Example: Get density for Yosemite area
    yosemite_stats = get_density_by_location(density_df, lat=37.75, lon=-119.58, radius_deg=0.3)
    print(f"Yosemite area vegetation stats: {yosemite_stats}")
    
    # ============================================================
    # TIME-SERIES ANALYSIS
    # ============================================================
    
    # Los Angeles area time-series (larger radius to capture more plots)
    la_timeseries = get_density_timeseries(
        density_df, lat=34.05, lon=-118.25, radius_deg=0.5, location_name="Los Angeles"
    )
    print_timeseries_summary(la_timeseries, "Los Angeles Area")
    
    # Save LA time-series to CSV
    if not la_timeseries.empty:
        la_output = Path(__file__).parent / 'output' / 'vegetation_data' / 'la_vegetation_timeseries.csv'
        la_timeseries.to_csv(la_output, index=False)
        print(f"\nSaved LA time-series to {la_output}")
    
    # San Francisco Bay Area
    sf_timeseries = get_density_timeseries(
        density_df, lat=37.77, lon=-122.42, radius_deg=0.5, location_name="San Francisco"
    )
    print_timeseries_summary(sf_timeseries, "San Francisco Bay Area")
    
    # Yosemite/Sierra Nevada
    yosemite_timeseries = get_density_timeseries(
        density_df, lat=37.75, lon=-119.58, radius_deg=0.3, location_name="Yosemite"
    )
    print_timeseries_summary(yosemite_timeseries, "Yosemite / Sierra Nevada")
    
    # San Diego area
    sd_timeseries = get_density_timeseries(
        density_df, lat=32.72, lon=-117.16, radius_deg=0.5, location_name="San Diego"
    )
    print_timeseries_summary(sd_timeseries, "San Diego Area")
    
    # ============================================================
    # SAVE .NPZ FILES FOR DENSE AREAS
    # ============================================================
    print("\n" + "="*70)
    print("SAVING DENSE AREAS AS .NPZ FILES")
    print("="*70)
    
    for year in [2017, 2018, 2019, 2020]:
        print(f"\n--- Creating .npz for {year} ---")
        npz_file = save_dense_areas_npz(density_df, year=year, threshold=10.0)
        
        # Load and verify the saved file
        data = np.load(npz_file, allow_pickle=True)
        print(f"  Verification: {len(data['coordinates'])} coordinates, radius={data['radius_km'][0]:.3f} km")
        data.close()
    
    print(f"\nAll .npz files saved in {Path(__file__).parent / 'output' / 'vegetation_data'}")
    
    # ============================================================
    # HIGH DENSITY VEGETATION AREAS
    # ============================================================
    print("\n" + "="*70)
    print("HIGH DENSITY VEGETATION AREAS BY YEAR")
    print("="*70)
    
    for year in [2017, 2018, 2019, 2020]:
        print(f"\n--- Year {year} ---")
        # Get top 10% most densely vegetated areas
        high_density_coords = get_high_density_areas(density_df, year=year, percentile=90)
        
        # Also save to file
        if len(high_density_coords) > 0:
            np.save(
                Path(__file__).parent / 'output' / 'vegetation_data' / f'high_density_coords_{year}.npy',
                high_density_coords
            )
            print(f"  Saved to output/vegetation_data/high_density_coords_{year}.npy")
