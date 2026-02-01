import React, { useState, useEffect, useRef, useCallback } from 'react';
import './WildfireSimulation.css';

const API_URL = 'http://localhost:5000/api';

// Cell state constants
const EMPTY = 0;
const TREE = 1;
const BURNING = 2;
const BURNT = 3;

// Cell colors
const COLORS = {
  [EMPTY]: '#D3D3D3',    // lightgrey
  [TREE]: '#228B22',      // green
  [BURNING]: '#FF0000',   // red
  [BURNT]: '#CD853F',     // peru/brown
  OUTSIDE: '#FFFFFF'      // white for outside California
};

const WildfireSimulation = () => {
  // Simulation state
  const [sessionId, setSessionId] = useState(null);
  const [grid, setGrid] = useState(null);
  const [mask, setMask] = useState(null);
  const [gridShape, setGridShape] = useState([0, 0]);
  const [isRunning, setIsRunning] = useState(false);
  const [hasBurning, setHasBurning] = useState(false);
  
  // Monte Carlo state
  const [monteCarloMode, setMonteCarloMode] = useState(false);
  const [monteCarloData, setMonteCarloData] = useState(null);
  const [isLoadingMC, setIsLoadingMC] = useState(false);
  
  // Parameters
  const [pTree, setPTree] = useState(0.6);
  const [ignitionProb, setIgnitionProb] = useState(0.7);
  const [windStrength, setWindStrength] = useState(1.0);
  const [windDir, setWindDir] = useState('None');
  const [mode, setMode] = useState('historic'); // 'historic' or 'custom'
  
  // Custom fire points
  const [customFires, setCustomFires] = useState([]);
  
  // Canvas ref
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  
  // Cell size calculation
  const canvasWidth = 800;
  const canvasHeight = 800;
  
  const cellSize = gridShape[0] > 0 ? Math.floor(canvasWidth / gridShape[0]) : 1;

  // Initialize simulation
  const initializeSimulation = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/init`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          p_tree: pTree,
          mode: mode,
          custom_fires: customFires
        })
      });
      
      const data = await response.json();
      setSessionId(data.session_id);
      setGrid(data.grid);
      setMask(data.mask);
      setGridShape(data.shape);
      setMonteCarloMode(false);
      setMonteCarloData(null);
      
      // Check if there are burning cells
      const burning = data.grid.some(row => row.includes(BURNING));
      setHasBurning(burning);
      
      if (mode === 'historic' && burning) {
        setIsRunning(false); // Don't auto-start, wait for user
      } else {
        setIsRunning(false);
      }
    } catch (error) {
      console.error('Failed to initialize simulation:', error);
    }
  }, [pTree, mode, customFires]);

  const hasBurningRef = useRef(hasBurning);

  useEffect(() => {
    hasBurningRef.current = hasBurning;
  }, [hasBurning]);


  // Step simulation
  const stepSimulation = useCallback(async () => {
    if (!sessionId) return;
    
    try {
      const response = await fetch(`${API_URL}/step`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          ignition_prob: ignitionProb,
          wind_dir: windDir,
          wind_strength: windStrength
        })
      });
      
      const data = await response.json();
      setGrid(data.grid);
      setHasBurning(data.has_burning);
      
      // Stop if no more burning cells
      if (!data.has_burning) {
        setIsRunning(false);
      }
    } catch (error) {
      console.error('Failed to step simulation:', error);
      setIsRunning(false);
    }
  }, [sessionId, ignitionProb, windDir, windStrength]);

  // Add fire point (custom mode)
  const addFire = useCallback(async (y, x) => {
    // if (!sessionId || mode !== 'custom') return;
    
    try {
      const response = await fetch(`${API_URL}/add-fire`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          y: y,
          x: x
        })
      });
      
      const data = await response.json();
      if (data.success) {
        setGrid(data.grid);
        setCustomFires([...customFires, [y, x]]);
        setHasBurning(true);
      }
    } catch (error) {
      console.error('Failed to add fire:', error);
    }
  }, [sessionId, mode, customFires]);

  // Run Monte Carlo
  const runMonteCarlo = useCallback(async () => {
    setIsLoadingMC(true);
    setIsRunning(false);
    
    try {
      const response = await fetch(`${API_URL}/monte-carlo`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          n_runs: 100,
          p_tree: pTree,
          ignition_prob: ignitionProb,
          wind_dir: windDir,
          wind_strength: windStrength,
          mode: mode,
          custom_fires: customFires
        })
      });
      
      const data = await response.json();
      setMonteCarloData(data.probability);
      setMonteCarloMode(true);
      setMask(data.mask);
    } catch (error) {
      console.error('Failed to run Monte Carlo:', error);
    } finally {
      setIsLoadingMC(false);
    }
  }, [pTree, ignitionProb, windDir, windStrength, mode, customFires]);

  // Draw grid on canvas
  const drawGrid = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !grid || !mask) return;
    
    const ctx = canvas.getContext('2d');
    const N = gridShape[0];
    
    // Clear canvas
    ctx.fillStyle = COLORS.OUTSIDE;
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);
    
    if (monteCarloMode && monteCarloData) {
      // Draw Monte Carlo probability heatmap
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          if (mask[i][j] === 1) {
            const prob = monteCarloData[i][j];
            // Hot colormap: black -> red -> orange -> yellow -> white
            const intensity = Math.floor(prob * 255);
            ctx.fillStyle = `rgb(${intensity}, ${Math.floor(intensity * 0.5)}, 0)`;
            // ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);

            const x = j * cellSize;
            const y = (N - 1 - i) * cellSize;


            ctx.fillRect(x, y, cellSize, cellSize);
          }
        }
      }
    } else {
      // Draw regular grid
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          if (mask[i][j] === 1) {
            const cellState = grid[i][j];
            ctx.fillStyle = COLORS[cellState];

            const x = j * cellSize;
            const y = (N - 1 - i) * cellSize;


            ctx.fillRect(x, y, cellSize, cellSize);
            // ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
          }
        }
      }
    }
    
    // Draw grid lines (optional, for small grids)
    if (N <= 100) {
      ctx.strokeStyle = '#00000010';
      ctx.lineWidth = 0.5;
      for (let i = 0; i <= N; i++) {
        ctx.beginPath();
        ctx.moveTo(0, i * cellSize);
        ctx.lineTo(canvasWidth, i * cellSize);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(i * cellSize, 0);
        ctx.lineTo(i * cellSize, canvasHeight);
        ctx.stroke();
      }
    }
  }, [grid, mask, gridShape, cellSize, monteCarloMode, monteCarloData]);

  // Handle canvas click
  const handleCanvasClick = useCallback((e) => {
    // if (mode !== 'custom' || isRunning || monteCarloMode) return;
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / cellSize);
    const y = Math.floor((e.clientY - rect.top) / cellSize);
    
    addFire(y, x);
  }, [mode, isRunning, monteCarloMode, cellSize, addFire]);


  const runSimulation = useCallback(async () => {
    if (!isRunning || !hasBurningRef.current) return;

    await stepSimulation();

    animationRef.current = setTimeout(runSimulation, 100);
  }, [isRunning, stepSimulation]);



  // Animation loop
  useEffect(() => {
    if (isRunning) {
      runSimulation();
    } else {
      if (animationRef.current) clearTimeout(animationRef.current);
    }

    return () => {
      if (animationRef.current) clearTimeout(animationRef.current);
    };
  }, [isRunning, runSimulation]);


  // Draw grid whenever it changes
  useEffect(() => {
    drawGrid();
  }, [drawGrid]);

  // Initialize on mount
  useEffect(() => {
    initializeSimulation();
  }, []);

  // Handle parameter changes - reset simulation
  const handleReset = () => {
    setCustomFires([]);
    setIsRunning(false);
    initializeSimulation();
  };

  // Handle mode change
  const handleModeChange = (newMode) => {
    setMode(newMode);
    setCustomFires([]);
    setIsRunning(false);
  };

  return (
    <div className="wildfire-simulation">
      <h1>California Wildfire Simulation</h1>

      {/* <div className="simulation-container flex justify-center">
        <div className="w-full max-w-[800px] aspect-square">
          <canvas
            ref={canvasRef}
            width={canvasWidth}
            height={canvasHeight}
            onClick={handleCanvasClick}
            className="w-full h-full border-2 border-zinc-800"
            style={{
              cursor:
                mode === "custom" && !isRunning && !monteCarloMode
                  ? "crosshair"
                  : "default",
            }}
          />
        </div>
      </div> */}

      
      <div className="simulation-container">

        <div className="canvas-container">
          <canvas
            ref={canvasRef}
            width={canvasWidth}
            height={canvasHeight}
            onClick={handleCanvasClick}
            style={{ 
              cursor: mode === 'custom' && !isRunning && !monteCarloMode ? 'crosshair' : 'default',
              border: '2px solid #333'
            }}
          />
          

          <div className="status">
            {monteCarloMode && (
              <p><strong>Monte Carlo Mode</strong> - Showing burn probability (20 runs)</p>
            )}
            {mode === 'custom' && !isRunning && !monteCarloMode && (
              <p>Click on California to place fires, then click <strong>Start</strong></p>
            )}
            {isLoadingMC && <p>Running Monte Carlo simulation...</p>}
          </div>
        </div>
        

        <div className="controls">
          <h3>Parameters</h3>
          

          <div className="control-group">
            <label>
              Vegetation Density: {pTree.toFixed(2)}
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.02"
                value={pTree}
                onChange={(e) => setPTree(parseFloat(e.target.value))}
                disabled={isRunning || isLoadingMC}
              />
            </label>
          </div>
          

          <div className="control-group">
            <label>
              Dryness (Ignition Prob): {ignitionProb.toFixed(2)}
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.02"
                value={ignitionProb}
                onChange={(e) => setIgnitionProb(parseFloat(e.target.value))}
                disabled={isRunning || isLoadingMC}
              />
            </label>
          </div>
          

          <div className="control-group">
            <label>
              Wind Strength: {windStrength.toFixed(1)}
              <input
                type="range"
                min="1.0"
                max="9.0"
                step="0.1"
                value={windStrength}
                onChange={(e) => setWindStrength(parseFloat(e.target.value))}
                disabled={isRunning || isLoadingMC}
              />
            </label>
          </div>
          
          <h3>Mode</h3>
          <div className="control-group">
            <label>
              <input
                type="radio"
                value="historic"
                checked={mode === 'historic'}
                onChange={(e) => handleModeChange(e.target.value)}
                disabled={isRunning || isLoadingMC}
              />
              Historic Run
            </label>
            <label>
              <input
                type="radio"
                value="custom"
                checked={mode === 'custom'}
                onChange={(e) => handleModeChange(e.target.value)}
                disabled={isRunning || isLoadingMC}
              />
              Custom Run
            </label>
          </div>
          
          <h3>Wind Direction</h3>
          <div className="control-group wind-direction">
            {['None', 'Up', 'Down', 'Left', 'Right'].map((dir) => (
              <label key={dir}>
                <input
                  type="radio"
                  value={dir}
                  checked={windDir === dir}
                  onChange={(e) => setWindDir(e.target.value)}
                  disabled={isRunning || isLoadingMC}
                />
                {dir}
              </label>
            ))}
          </div>
          
          <h3>Actions</h3>
          <div className="button-group">
            <button
              onClick={() => setIsRunning(!isRunning)}
              disabled={!hasBurning || monteCarloMode || isLoadingMC}
            >
              {isRunning ? 'Pause' : 'Start'}
            </button>
            
            <button
              onClick={handleReset}
              disabled={isLoadingMC}
            >
              Reset
            </button>
            
            <button
              onClick={runMonteCarlo}
              disabled={isRunning || isLoadingMC}
            >
              {isLoadingMC ? 'Running...' : 'Monte Carlo'}
            </button>
          </div>
          
          <div className="legend">
            <h3>Legend</h3>
            <div className="legend-item">
              <div className="legend-color" style={{ backgroundColor: COLORS[EMPTY] }}></div>
              <span>Empty</span>
            </div>
            <div className="legend-item">
              <div className="legend-color" style={{ backgroundColor: COLORS[TREE] }}></div>
              <span>Tree</span>
            </div>
            <div className="legend-item">
              <div className="legend-color" style={{ backgroundColor: COLORS[BURNING] }}></div>
              <span>Burning</span>
            </div>
            <div className="legend-item">
              <div className="legend-color" style={{ backgroundColor: COLORS[BURNT] }}></div>
              <span>Burnt</span>
            </div>
          </div>
        </div>
      </div>
      
      <div className="instructions">
        <h3>Instructions</h3>
        <ul>
          <li><strong>Historic Run:</strong> Uses real fire start locations from data</li>
          <li><strong>Custom Run:</strong> Click on California to place your own fire points</li>
          <li><strong>Monte Carlo:</strong> Runs 20 simulations and shows burn probability heatmap</li>
          <li>Adjust parameters to see how vegetation density, dryness, and wind affect fire spread</li>
        </ul>
      </div>
    </div>
  );
};

export default WildfireSimulation;