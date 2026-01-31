"use client"

import { ComposableMap, Geographies, Geography, Marker, ZoomableGroup } from "react-simple-maps";
import { useState, useMemo } from "react";

export default function Home() {
  const [position, setPosition] = useState({ coordinates: [0, 0], zoom: 1 });

  const handleMove = (newPosition) => {
    setPosition(newPosition);
  };

  const markers = [
    {
      markerOffset: -15,
      name: "Buenos Aires",
      coordinates: [-58.3816, -34.6037]
    },
    { markerOffset: -15, name: "La Paz", coordinates: [-68.1193, -16.4897] },
    { markerOffset: 25, name: "Brasilia", coordinates: [-47.8825, -15.7942] },
    { markerOffset: 25, name: "Santiago", coordinates: [-70.6693, -33.4489] },
    { markerOffset: 25, name: "Bogota", coordinates: [-74.0721, 4.711] },
    { markerOffset: 25, name: "Quito", coordinates: [-78.4678, -0.1807] },
    { markerOffset: -15, name: "Georgetown", coordinates: [-58.1551, 6.8013] },
    { markerOffset: -15, name: "Asuncion", coordinates: [-57.5759, -25.2637] },
    { markerOffset: 25, name: "Paramaribo", coordinates: [-55.2038, 5.852] },
    { markerOffset: 25, name: "Montevideo", coordinates: [-56.1645, -34.9011] },
    { markerOffset: -15, name: "Caracas", coordinates: [-66.9036, 10.4806] },
    { markerOffset: -15, name: "Lima", coordinates: [-77.0428, -12.0464] }
  ];

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-stone-900">
      <div className="w-full flex items-center justify-center">
        <section className="w-fit h-[420px] sm:h-[500px] md:h-[600px] overflow-hidden bg-sky-300 dark:bg-sky-900 rounded-md">
          {/* <div className="absolute inset-0 bg-gradient-to-b from-transparent to-black/40 pointer-events-none" /> */}
          <ComposableMap
          projection="geoMercator"
          projectionConfig={{
            scale: 130,
          }}
          // className="block"
          style={{ display: "block", width: "100%", height: "130%" }}
        >
          <ZoomableGroup
            zoom={position.zoom}
            center={position.coordinates}
            minZoom={1}
            maxZoom={4}             // adjust max zoom
            onMoveEnd={handleMove}  // updates position state
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
                      hover: { fill: "#3b82f6", outline: "none" },
                      pressed: { outline: "none" },
                    }}
                  />
                ))
              }
            </Geographies>
            {markers.map(({ name, coordinates, markerOffset }) => (
              <Marker key={name} coordinates={coordinates}>
                <circle r={5} fill="#f43f5e" stroke="#fff" strokeWidth={2} />
                <text textAnchor="middle" y={markerOffset} style={{ fontFamily: "system-ui", fill: "var(--foreground)", fontSize: "10px" }}>
                  {name}
                </text>
              </Marker>
            ))}
          </ZoomableGroup>
        </ComposableMap>
        </section>
      </div>
    </div>
  );
}
