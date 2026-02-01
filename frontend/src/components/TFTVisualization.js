"use client";

import { useState, useEffect } from "react";
import { Button, Spin, message, Space } from "antd";
import { useSelector } from "react-redux";

export default function TFTVisualization() {
  const selectedPoint = useSelector((state) => state.points.selectedPoint);
  const [sessionId, setSessionId] = useState(null);
  const [imageData, setImageData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [isMonteCarloMode, setIsMonteCarloMode] = useState(false);

  // Initialize TFT simulation when component mounts or point changes
  useEffect(() => {
    if (selectedPoint) {
      initializeTFT();
    }
  }, [selectedPoint?.year]); // Reinitialize when year changes

  const initializeTFT = async () => {
    try {
      setLoading(true);
      const response = await fetch("http://127.0.0.1:5001/tft/init", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          temperature: 50,
          humidity: 5,
          wind_speed: 20,
          wind_direction: 270,
        }),
      });

      const data = await response.json();
      setSessionId(data.session_id);
      setImageData(data.image);
      setStats(data.stats);
      setIsRunning(true);
      setIsMonteCarloMode(false);
    } catch (error) {
      console.error("Failed to initialize TFT simulation:", error);
      message.error("Failed to initialize TFT simulation");
    } finally {
      setLoading(false);
    }
  };

  const stepSimulation = async () => {
    if (!sessionId || !isRunning) return;

    try {
      const response = await fetch(`http://127.0.0.1:5001/tft/step/${sessionId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });

      const data = await response.json();
      setImageData(data.image);
      setStats(data.stats);
      setIsRunning(data.stats.is_running);
    } catch (error) {
      console.error("Failed to step simulation:", error);
      message.error("Failed to advance simulation");
    }
  };

  const resetSimulation = async () => {
    if (!sessionId) return;

    try {
      setLoading(true);
      const response = await fetch(`http://127.0.0.1:5001/tft/reset/${sessionId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });

      const data = await response.json();
      setImageData(data.image);
      setStats(data.stats);
      setIsRunning(true);
      setIsMonteCarloMode(false);
    } catch (error) {
      console.error("Failed to reset simulation:", error);
      message.error("Failed to reset simulation");
    } finally {
      setLoading(false);
    }
  };

  const runMonteCarloAnalysis = async () => {
    if (!sessionId) return;

    try {
      setLoading(true);
      const response = await fetch(`http://127.0.0.1:5001/tft/monte-carlo/${sessionId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ n_runs: 20 }),
      });

      const data = await response.json();
      setImageData(data.image);
      setStats(data.stats);
      setIsMonteCarloMode(true);
      setIsRunning(false);
    } catch (error) {
      console.error("Failed to run Monte Carlo analysis:", error);
      message.error("Failed to run Monte Carlo analysis");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center">
      <Spin spinning={loading} size="large">
        <div
          style={{
            width: "800px",
            height: "600px",
            backgroundColor: "#f0f0f0",
            borderRadius: "4px",
            overflow: "hidden",
            marginBottom: "16px",
          }}
        >
          {imageData ? (
            <img
              src={imageData}
              alt="TFT Simulation"
              style={{
                width: "100%",
                height: "100%",
                objectFit: "contain",
              }}
            />
          ) : (
            <div className="flex items-center justify-center w-full h-full">
              <p className="text-gray-400">No simulation data</p>
            </div>
          )}
        </div>
      </Spin>

      {stats && (
        <div className="mb-4 p-3 bg-white rounded border">
          <p className="text-sm">
            <strong>Step:</strong> {stats.frame_count} | <strong>Burning:</strong>{" "}
            {stats.burning} | <strong>Burnt:</strong> {stats.burnt}
          </p>
          {isMonteCarloMode && (
            <p className="text-sm text-orange-600">
              <strong>Mode:</strong> Monte Carlo Analysis (20 runs)
            </p>
          )}
        </div>
      )}

      <Space>
        <Button
          type="primary"
          onClick={stepSimulation}
          disabled={!isRunning || isMonteCarloMode || loading}
        >
          Step
        </Button>
        <Button onClick={resetSimulation} disabled={loading}>
          Reset
        </Button>
        <Button
          type="default"
          onClick={runMonteCarloAnalysis}
          disabled={loading || isMonteCarloMode}
        >
          Monte Carlo (20)
        </Button>
      </Space>
    </div>
  );
}
