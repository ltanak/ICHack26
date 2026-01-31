"use client"

import { ComposableMap, Geographies, Geography, Marker, ZoomableGroup } from "react-simple-maps";
import { useState, useEffect } from "react";
import { geoMercator } from "d3-geo";

export default function MapChart() {
    const [centre, setCentre] = useState([0, 0]);
    const [zoom, setZoom] = useState(1);
    const [markers, setMarkers] = useState([]);
    const [hoveredMarker, setHoveredMarker] = useState(null);
    
    useEffect(() => {
        const fetchPoints = async () => {
            try {
            const res = await fetch("http://127.0.0.1:5000/points");
            const data = await res.json();
            setMarkers(data);
            } catch (err) {
            console.error("Error loading points:", err);
            }
        };
        fetchPoints();
    }, []);
    
    // Map configuration
    const width = 800; // your SVG width
    const height = 600; // your SVG height
    const projectionScale = 130; // same as in your ComposableMap

    // Create a projection instance
    const getProjection = (scale, center) =>
        geoMercator()
            .scale(scale)
            .center(center)
            .translate([width / 2, height / 2]);

    const clampCentre = (center, zoom) => {
        const scale = initialScale * zoom;
        const projection = getProjection(scale, center);

        // World bounds in Mercator coordinates
        const worldLeft = -180;
        const worldRight = 180;
        const worldTop = 90;
        const worldBottom = -90;

        // Compute how far the center can move before empty space is visible
        const lngPadding = (width / 2) / scale * 360; // approximate in degrees
        const latPadding = (height / 2) / scale * 180; // approximate in degrees

        const clampedLng = Math.max(worldLeft + lngPadding, Math.min(worldRight - lngPadding, center[0]));
        const clampedLat = Math.max(worldBottom + latPadding, Math.min(worldTop - latPadding, center[1]));

        return [clampedLng, clampedLat];
        };

    return (
        <div className="flex items-center justify-center overflow-hidden">
            <section className="w-full h-screen overflow-hidden bg-sky-800">
            {/* <div className="absolute inset-0 bg-gradient-to-b from-transparent to-black/40 pointer-events-none" /> */}
                <ComposableMap
                projection="geoMercator"
                projectionConfig={{
                    scale: projectionScale,
                }}
                // className="block"
                style={{ display: "block", width: "100%", height: "130%" }}
                >
                    <ZoomableGroup
                        center={centre}
                        minZoom={1}
                        maxZoom={4}             // adjust max zoom
                        onMove={({ coordinates, zoom }) => {
                            if (!coordinates) return;
                            setZoom(zoom);
                            const clamped = clampCentre(coordinates, zoom);
                            setCentre(clamped);
                        }}
                    >
                        <Geographies geography="/countries-110m.json">
                        {({ geographies }) =>
                            geographies.map((geo) => (
                            <Geography
                                key={geo.rsmKey}
                                geography={geo}
                                fill="var(--map-fill)"
                                stroke="var(--map-stroke)"
                                    style={{
                                    default: { outline: "none" },
                                    hover: { outline: "none" },
                                    pressed: { outline: "none" },
                                }}
                            />
                            ))
                        }
                        </Geographies>
                        {markers.map(({ name, coordinates, markerOffset }) => {
                            const scale = 1 / zoom;
                            return (
                                <Marker key={name} coordinates={coordinates}>
                                    <circle 
                                        r={5 * scale} 
                                        fill="#f43f5e" 
                                        stroke="#fff" 
                                        strokeWidth={2 * scale} 
                                        onMouseEnter={() => setHoveredMarker(name)} 
                                        onMouseLeave={() => setHoveredMarker(null)} 
                                        className="cursor-pointer"
                                    />
                                    <text 
                                        textAnchor="middle" 
                                        y={markerOffset * scale}
                                        className={`font-semibold text-black transition-all duration-300 ${hoveredMarker === name ? "opacity-100" : "opacity-0"}`}
                                        style={{ fontSize: 12 * scale }}
                                    >
                                    {name}
                                    </text>
                                </Marker>
                            )
                        })}
                    </ZoomableGroup>
                </ComposableMap>
            </section>
        </div>
    )
}