"use client"

import { ComposableMap, Geographies, Geography, Marker, ZoomableGroup } from "react-simple-maps";
import { useState, useEffect } from "react";
import { geoMercator } from "d3-geo";
import { useGetPointsQuery } from "../store/api";
import { useDispatch } from "react-redux";
import { setSelectedPoint } from "@/store/features/points/pointsSlice";

export default function MapChart() {
    const dispatch = useDispatch();
    const [centre, setCentre] = useState([0, 0]);
    const [zoom, setZoom] = useState(1);
    const { data: markers, error, isLoading } = useGetPointsQuery();
    const [hoveredMarker, setHoveredMarker] = useState(null);
    
    // Map configuration
    const width = 1000; // your SVG width
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

    if (isLoading) return <div>Loading map...</div>;
    if (error) return <div>Error loading map data</div>;

    return (
        <div className="w-full bg-sky-800">
            <div className="flex items-center justify-center overflow-hidden">
                <section className="h-[40rem] overflow-hidden bg-sky-800">
                {/* <div className="absolute inset-0 bg-gradient-to-b from-transparent to-black/40 pointer-events-none" /> */}
                    <ComposableMap
                    projection="geoMercator"
                    projectionConfig={{
                        scale: projectionScale,
                        rotate: [-260, -7, 0]
                    }}
                    width={width}
                    height={height}
                    // className="block"
                    style={{ display: "block", width: "100%", height: "120%" }}
                    >
                        <ZoomableGroup
                            center={centre}
                            minZoom={1}
                            maxZoom={4}
                            translateExtent={[
                                [0, 0],       // top-left corner
                                [width, height],   // bottom-right corner (SVG size)
                            ]}
                            onMove={({ coordinates, zoom }) => {
                                setZoom(zoom);
                                if (!coordinates) return;
                                const clamped = clampCentre(coordinates, zoom);
                                setCentre(clamped);
                            }}
                        >
                            <Geographies geography="/countries-filtered.json">
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
                            {markers.map(({ name, coordinates, markerOffset, year }) => {
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
                                            onClick={() => dispatch(setSelectedPoint({ name, year, coordinates }))}
                                            className="cursor-pointer"
                                        />
                                        <text 
                                            textAnchor="middle" 
                                            y={markerOffset * scale}
                                            className={`font-semibold text-black bg-stone-100 rounded-md transition-all duration-300 ${hoveredMarker === name ? "opacity-100" : "opacity-0"}`}
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
        </div>
    )
}