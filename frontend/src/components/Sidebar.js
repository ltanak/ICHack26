import { MenuFoldOutlined, MenuUnfoldOutlined } from "@ant-design/icons";
import { Button } from "antd";
import { useState, useEffect } from "react";
import WildfireSimulation from "./WildfireSimulation";

export default function Sidebar({ collapsed, setCollapsed }) {
    const [snapshots, setSnapshots] = useState([]);
    const [currentIndex, setCurrentIndex] = useState(0);
    const year = 2017;
    const API_BASE = 'http://127.0.0.1:5001';

    useEffect(() => {
        // Fetch available snapshots
        fetch(`${API_BASE}/snapshots?year=${year}`)
            .then(res => res.json())
            .then(data => setSnapshots(data.snapshots))
            .catch(err => console.error('Error fetching snapshots:', err));
    }, []);

    useEffect(() => {
        if (snapshots.length === 0) return;

        // Loop through snapshots every 800ms
        const interval = setInterval(() => {
            setCurrentIndex((prev) => (prev + 1) % snapshots.length);
        }, 800);

        return () => clearInterval(interval);
    }, [snapshots]);

    const currentSnapshot = snapshots[currentIndex];
    const imageUrl = currentSnapshot 
        ? `${API_BASE}/satellite/${year}?snapshot=${currentSnapshot}`
        : `${API_BASE}/satellite/${year}`;

    const [simulationData, setSimulationData] = useState(null);

    useEffect(() => {
        fetch(`${process.env.API_URL}/simulation/frame`)
            .then(res => res.json())
            .then(data => setSimulationData(data))
            .catch(err => console.error('Error fetching simulation data:', err));
    }, []);
    return (
        <div className={`flex flex-col items-end px-4 py-6 transition-all duration-300 mr-2`}>
            <Button 
                type="text" 
                    icon={
                    <span style={{ fontSize: 24 }}>
                    {collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
                    </span>
                }
                onClick={() => setCollapsed(!collapsed)} 
            />
            <div className={`px-2 gap-y-6 flex flex-col items-start ${collapsed ? "opacity-0" : "opacity-100"} transition-all duration-300 mt-4 w-full`}>
                <h1 className="text-3xl"><strong>Wildfire Tracker</strong></h1>
                <div>
                    <h2 className="text-2xl"><strong>Information</strong></h2>
                    <ul className="">
                        <li className="text-lg">Wildfire Name: to add</li>
                        <li className="text-lg">Location: to add</li>
                        <li className="text-lg">Size: to add</li>
                        <li className="text-lg">Containment: to add</li>
                        <li className="text-lg">Start Date: to add</li>
                        <li className="text-lg">Cause: to add</li>
                    </ul>
                </div>
                <div>
                    <h2 className="text-2xl"><strong>Summary</strong></h2>
                    <p className="text-lg">Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aenean porttitor magna eget tempus tristique. Suspendisse commodo arcu lacus, id iaculis elit aliquam non. Sed interdum dignissim turpis, vitae cursus neque volutpat eget. Aliquam vitae efficitur mauris. Integer ullamcorper, diam vitae pharetra tristique, lorem lorem lacinia diam, a cursus libero metus a felis. Sed luctus venenatis pellentesque. Donec sed nunc eget nunc placerat lobortis. Donec fringilla leo dolor, quis faucibus enim imperdiet sed. In fermentum libero ipsum, eget convallis eros gravida et. Aliquam ex massa, bibendum non rutrum vel, posuere mollis ante. Nam congue laoreet dictum. Nulla quis neque imperdiet, fringilla lacus ac, iaculis mi. Nunc laoreet lacinia feugiat. Aliquam nec rhoncus nisi, a malesuada velit.</p>
                    <div className="relative">
                        <img
                            key={imageUrl}
                            src={imageUrl}
                            alt="Satellite view"
                            className="w-full h-auto"
                            style={{ imageRendering: 'auto' }}
                            onError={(e) => console.error('Image failed to load:', imageUrl)}
                        />
                        {snapshots.length > 0 && (
                            <div className="text-sm mt-2 text-gray-600">
                                Frame {currentIndex + 1} / {snapshots.length} - {currentSnapshot}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}