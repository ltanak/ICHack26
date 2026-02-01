import { Divider } from "antd";
import { useSelector } from "react-redux";
import WildfireSummary from "./WildfireSummary";
import WildfireSimulation from "./WildfireSimulation";

export default function WildfireInfo() {
    const selectedPoint = useSelector((state) => state.points.selectedPoint);

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
                <img
                    src={`http://127.0.0.1:5001/satellite`}
                    alt="Satellite view"
                    className="w-full h-auto"
                />
                <WildfireSimulation gridMode={true}/>
            </div>
        </div>
    );
}