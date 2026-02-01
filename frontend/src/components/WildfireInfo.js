import { Divider } from "antd";
import { useSelector } from "react-redux";
import WildfireSummary from "./WildfireSummary";
import WildfireSimulation from "./WildfireSimulation";
import WildfireMitigation from "./WildfireMitigation";
import { Button, Flex, Splitter, Switch, Typography } from 'antd';
import { useState, useMemo } from "react";

const yearImageMap = {
  "2017": "/2017.png",
  "2018": "/2018.png",
  "2019": "/2019.png",
  "2020": "/2020.png",
};

export default function WildfireInfo() {
    const selectedPoint = useSelector((state) => state.points.selectedPoint);
    const [sizes, setSizes] = useState([50, 50]);

    const satelliteSrc = useMemo(() => {
        const ts = Date.now(); // new timestamp ONLY when year changes
        return `http://127.0.0.1:5001/satellite/?year=${selectedPoint?.year}&t=${ts}`;
    }, [selectedPoint?.year]);

    if (!selectedPoint) {
        return (
            <div className="bg-slate-200 p-4 text-center flex-1">
                <h2 className="text-3xl font-bold mb-2">No Wildfire Selected</h2>
                <p className="text-xl">Please select a wildfire on the map to see more information.</p>
            </div>
        );
    }

    return (
        <div className="bg-slate-200 p-4 text-center flex-1">
            <h2 className="text-3xl font-bold mb-2">{selectedPoint.name}</h2>
            <Divider />
            <WildfireSummary />
            <Divider />
            <div className="grid grid-cols-2">
                <div>
                    <h2 className="text-2xl text-bold text-center items-center mr-16">Overlay of True Wildfire Spread</h2>
                    <div className="mt-6" style={{ 
                    position: 'relative', 
                    width: '800px', 
                    height: '600px',
                    userSelect: 'none'
                }}>
                    {/* Base image (right side) - always visible */}
                    <div style={{ 
                        position: 'absolute',
                        left: 0,
                        top: 0,
                        width: '100%',
                        height: '100%',
                        pointerEvents: 'none'
                    }}>
                        <img
                            src={yearImageMap[selectedPoint.year] || "/2020.png"}
                            alt="Fire overlay"
                            style={{
                                width: '100%',
                                height: '100%',
                                objectFit: 'cover',
                                display: 'block'
                            }}
                        />
                    </div>
                    
                    {/* Overlay image (left side) - clipped by width */}
                    <div style={{ 
                        position: 'absolute',
                        left: 0,
                        top: 0,
                        width: `${sizes[0]}%`,
                        height: '100%',
                        overflow: 'hidden',
                        pointerEvents: 'none'
                    }}>
                        <img
                            src={satelliteSrc}
                            alt="Satellite view"
                            style={{
                                position: 'absolute',
                                left: 0,
                                top: 0,
                                width: '800px',
                                minWidth: '800px',
                                maxWidth: '800px',
                                height: '600px',
                                minHeight: '600px',
                                maxHeight: '600px',
                                objectFit: 'cover',
                                display: 'block'
                            }}
                            onLoad={() => console.log(`Image loaded for year: ${selectedPoint.year}`)}
                            onError={(e) => console.error(`Failed to load image for year: ${selectedPoint.year}`, e)}
                        />
                    </div>
                    
                    {/* Draggable divider */}
                    <div 
                        style={{ 
                            position: 'absolute',
                            left: `${sizes[0]}%`,
                            top: 0,
                            width: '4px',
                            height: '100%',
                            backgroundColor: '#fff',
                            cursor: 'ew-resize',
                            boxShadow: '0 0 10px rgba(0, 0, 0, 0.3)',
                            zIndex: 10,
                            transform: 'translateX(-2px)'
                        }}
                        onMouseDown={(e) => {
                            e.preventDefault();
                            const container = e.currentTarget.parentElement;
                            const startX = e.clientX;
                            const startSize = sizes[0];
                            
                            document.body.style.userSelect = 'none';
                            document.body.style.cursor = 'ew-resize';
                            
                            const handleMouseMove = (moveEvent) => {
                                const containerRect = container.getBoundingClientRect();
                                const deltaX = moveEvent.clientX - startX;
                                const deltaPercent = (deltaX / containerRect.width) * 100;
                                const newSize = Math.max(0, Math.min(100, startSize + deltaPercent));
                                setSizes([newSize, 100 - newSize]);
                            };
                            
                            const handleMouseUp = () => {
                                document.body.style.userSelect = '';
                                document.body.style.cursor = '';
                                document.removeEventListener('mousemove', handleMouseMove);
                                document.removeEventListener('mouseup', handleMouseUp);
                            };
                            
                            document.addEventListener('mousemove', handleMouseMove);
                            document.addEventListener('mouseup', handleMouseUp);
                        }}
                    />
                </div>
                </div>
        
                <div>
                    <h2 className="text-2xl text-bold items-center text-center">Percolation Theory-based Prediction</h2>
                <WildfireSimulation gridMode={true}/>
                </div>
            </div>
            <h2 className="text-2xl text-bold items-center text-center items-center mt-4">Wildfire Mitigation Resource Allocator</h2>
            <WildfireMitigation />
        </div>
    );
}