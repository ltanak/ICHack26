import requests

def get_satellite_image(
    min_lon, min_lat, max_lon, max_lat,
    target_width=1024,
    out_file="satellite.png"
):

    width_deg = max_lon - min_lon
    height_deg = max_lat - min_lat

    if width_deg <= 0 or height_deg <= 0:
        raise ValueError("Invalid bounding box")

    aspect_ratio = width_deg / height_deg

    # Compute height from width
    width_px = target_width
    height_px = int(width_px / aspect_ratio)

    url = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"

    params = {
        "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "bboxSR": 4326,
        "imageSR": 4326,
        "size": f"{width_px},{height_px}",
        "format": "png",
        "f": "image"
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    with open(out_file, "wb") as f:
        f.write(response.content)

    return out_file, width_px, height_px

# Example for California
'''
get_satellite_image(
    min_lon=-123.54624440876643,
    min_lat=32.59507847453237,
    max_lon=-114.5110288020599,
    max_lat=41.992327628379186,
    out_file="sf1_satellite.png"
)   
'''
