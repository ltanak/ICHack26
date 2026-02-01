import { Divider } from "antd";
import { useSelector } from "react-redux";
import { useGetPointSummaryQuery } from "../store/api";
import { useEffect, useState } from "react";
import { useAnimatedText } from "@/app/hooks/useAnimatedText";

export default function WildfireSummary() {
    const selectedPoint = useSelector((state) => state.points.selectedPoint);
    const { data: pointSummary, error, isFetching } = useGetPointSummaryQuery(encodeURIComponent(selectedPoint?.name), {
        skip: !selectedPoint,
    });
    
    // Dynamic loading messages
    const loadingMessages = [
        "Generating summary...",
        "Processing fire details...", 
        "Analyzing satellite data...",
        "Calculating risk factors...",
        "Compiling environmental data...",
        "Preparing comprehensive report..."
    ];
    
    const [currentMessageIndex, setCurrentMessageIndex] = useState(0);
    
    // Cycle through loading messages
    useEffect(() => {
        if (!isFetching) return;
        
        const interval = setInterval(() => {
            setCurrentMessageIndex((prevIndex) => 
                (prevIndex + 1) % loadingMessages.length
            );
        }, 2000); // Change message every 2 seconds
        
        return () => clearInterval(interval);
    }, [isFetching, loadingMessages.length]);
    
    // Reset message index when starting to fetch
    useEffect(() => {
        if (isFetching) {
            setCurrentMessageIndex(0);
        }
    }, [isFetching]);
    
    const animatedText = useAnimatedText(pointSummary || "");

    if (!selectedPoint) {
        return (
            <div className="bg-slate-200 p-4 text-center flex-1">
                <h2 className="text-3xl font-bold mb-2">No Wildfire Selected</h2>
                <p className="text-xl">Please select a wildfire on the map to see more information.</p>
            </div>
        );
    }

    return (
        <>
            {isFetching && (
                <div className="flex flex-col items-center justify-center mt-8 animate-fade-in">
                    <div className="w-8 h-8 border-4 border-slate-300 border-t-indigo-500 rounded-full animate-spin mb-3"></div>
                    <p className="text-slate-700 text-lg font-medium transition-all duration-500 ease-in-out">
                        {loadingMessages[currentMessageIndex]}
                    </p>
                    <div className="flex space-x-1 mt-2">
                        {loadingMessages.map((_, index) => (
                            <div
                                key={index}
                                className={`w-2 h-2 rounded-full transition-all duration-300 ${
                                    index === currentMessageIndex 
                                        ? 'bg-indigo-500 scale-125' 
                                        : 'bg-slate-300'
                                }`}
                            />
                        ))}
                    </div>
                </div>
            )}
            {error && (
                <p className="text-red-600 mt-6 text-center">
                    Failed to generate summary.
                </p>
            )}
            {!isFetching && pointSummary&& (
                <div className="px-32">
                    <h2 className="text-2xl font-semibold text-center pb-2">Summary</h2>
                    <div className="text-left text-lg">
                        {animatedText}
                        <span className="inline-block w-2 animate-pulse">▌</span>
                    </div>
                </div>
            )}
        </>
    );
}