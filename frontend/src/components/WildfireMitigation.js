import React, { useState, useEffect, useRef, useCallback } from 'react';
import './WildfireSimulation.css';



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
  OUTSIDE: '#FFFFFF',     // white for outside California
  RETARDANT: '#00FFFF',   // cyan for retardant zones
  CLEARED: '#FFA500'      // orange for cleared zones
};

// Mitigation zone defaults (matching Python visualize.py)
const RETARDANT_RADIUS = 10;
const CLEAR_WIDTH = 18;
const CLEAR_HEIGHT = 9;

const WildfireSimulation = ({ gridMode = false }) => {
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
  const [pTree, setPTree] = useState(0.8);
  const [ignitionProb, setIgnitionProb] = useState(0.8);
  const [windStrength, setWindStrength] = useState(1.0);
  const [windDir, setWindDir] = useState('None');
  const [mode, setMode] = useState('historic'); // 'historic' or 'custom'
  
  // Custom fire points
  const [customFires, setCustomFires] = useState([]);
  
  // Mitigation zones state
  const [mitigationMode, setMitigationMode] = useState('None'); // 'None', 'Retardant', 'Clear'
  const [retardantZones, setRetardantZones] = useState([]); // [{y, x, radius}, ...]
  const [clearedZones, setClearedZones] = useState([]); // [{y, x, width, height}, ...]
  
  // Canvas ref
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  
  // Cell size calculation
  const baseWidth = 800;
  const baseHeight = 800;
  const canvasWidth = gridMode ? baseWidth : baseWidth;
  const canvasHeight = gridMode ? baseHeight  : baseHeight;
  
  const cellSize = gridShape[0] > 0 ? Math.floor(canvasWidth / gridShape[0]) : 1;

  // Initialize simulation
  const initializeSimulation = useCallback(async () => {
    try {
      const response = await fetch(`http://127.0.0.1:5001/api/init`, {
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
      const response = await fetch(`http://127.0.0.1:5001/api/step`, {
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
      const response = await fetch(`http://127.0.0.1:5001/api/add-fire`, {
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
      // Convert mitigation zones to backend format
      const retardantZonesData = retardantZones.map(z => [z.y, z.x, z.radius]);
      const clearedZonesData = clearedZones.map(z => [z.y, z.x, z.width, z.height]);
      
      const response = await fetch(`http://127.0.0.1:5001/api/monte-carlo`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          n_runs: 20,
          p_tree: pTree,
          ignition_prob: ignitionProb,
          wind_dir: windDir,
          wind_strength: windStrength,
          mode: mode,
          custom_fires: customFires,
          retardant_zones: retardantZonesData,
          cleared_zones: clearedZonesData
        })
      });
      
      const data = await response.json();
      console.log('Monte Carlo response:', data);
      console.log('Probability data:', data.probability);
      console.log('Probability shape:', data.probability ? [data.probability.length, data.probability[0]?.length] : 'null');
      
      // Check for non-zero values
      let nonZeroCount = 0;
      let maxProb = 0;
      let sampleNonZero = [];
      if (data.probability) {
        for (let i = 0; i < data.probability.length; i++) {
          for (let j = 0; j < data.probability[i].length; j++) {
            const val = data.probability[i][j];
            if (val > 0) {
              nonZeroCount++;
              maxProb = Math.max(maxProb, val);
              if (sampleNonZero.length < 5) {
                sampleNonZero.push({i, j, val});
              }
            }
          }
        }
      }
      console.log('Non-zero probability cells:', nonZeroCount);
      console.log('Max probability:', maxProb);
      console.log('Sample non-zero values:', sampleNonZero);
      
      console.log('Mask shape:', data.mask ? [data.mask.length, data.mask[0]?.length] : 'null');
      console.log('Mask sample (first few cells):', data.mask ? data.mask.slice(0, 3).map(row => row.slice(0, 5)) : 'null');
      console.log('Mask actual values in first row:', data.mask ? data.mask[0].slice(0, 10) : 'null');
      console.log('Mask value types:', data.mask && data.mask[0] ? typeof data.mask[0][0] : 'null');
      setMonteCarloData(data.probability);
      setMonteCarloMode(true);
      setMask(data.mask);
    } catch (error) {
      console.error('Failed to run Monte Carlo:', error);
    } finally {
      setIsLoadingMC(false);
    }
  }, [pTree, ignitionProb, windDir, windStrength, mode, customFires, retardantZones, clearedZones]);

  // Draw grid on canvas
  const drawGrid = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !mask) return;
    
    // In Monte Carlo mode, we don't need grid
    if (!monteCarloMode && !grid) return;
    
    const ctx = canvas.getContext('2d');
    const N = gridShape[0];
    
    console.log('Drawing - MonteCarloMode:', monteCarloMode, 'Has data:', !!monteCarloData);
    
    // Clear canvas
    ctx.fillStyle = COLORS.OUTSIDE;
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);
    
    if (monteCarloMode && monteCarloData) {
      console.log('Drawing Monte Carlo heatmap, N:', N, 'cellSize:', cellSize);
      console.log('Mask in drawGrid:', mask ? [mask.length, mask[0]?.length] : 'null');
      console.log('Mask sample at draw time:', mask ? mask.slice(0, 3).map(row => row.slice(0, 5)) : 'null');
      let cellsDrawn = 0;
      let maskOnes = 0;
      // Draw Monte Carlo probability heatmap
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          if (mask[i] && mask[i][j] === 1) maskOnes++;
          if (mask[i] && mask[i][j] === 1) {
            const prob = monteCarloData[i][j];
            
            // Enhanced colormap for better visibility:
            // 0.0 -> white/light yellow (low risk)
            // 0.5 -> orange (medium risk)  
            // 1.0 -> red (high risk)
            let r, g, b;
            if (prob === 0) {
              // No burn probability - use background
              r = 240; g = 240; b = 220;
            } else if (prob < 0.5) {
              // Low to medium: white -> yellow -> orange
              const t = prob * 2; // 0 to 1
              r = 255;
              g = Math.floor(255 - t * 100); // 255 -> 155
              b = Math.floor(50 * (1 - t)); // 50 -> 0
            } else {
              // Medium to high: orange -> red
              const t = (prob - 0.5) * 2; // 0 to 1
              r = 255;
              g = Math.floor(155 * (1 - t)); // 155 -> 0
              b = 0;
            }
            
            ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;

            const x = j * cellSize;
            const y = (N - 1 - i) * cellSize;

            ctx.fillRect(x, y, cellSize, cellSize);
            cellsDrawn++;
          }
        }
      }
      console.log('Cells drawn:', cellsDrawn, 'Mask ones found:', maskOnes);
      
      // Draw mitigation zones on Monte Carlo heatmap (inline)
      // Draw retardant zones (cyan dashed circles)
      ctx.strokeStyle = COLORS.RETARDANT;
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      
      for (const zone of retardantZones) {
        const zoneCanvasX = zone.x * cellSize;
        const zoneCanvasY = (N - 1 - zone.y) * cellSize;
        const zoneCanvasRadius = zone.radius * cellSize;
        
        ctx.beginPath();
        ctx.arc(zoneCanvasX, zoneCanvasY, zoneCanvasRadius, 0, 2 * Math.PI);
        ctx.stroke();
      }
      
      // Draw cleared zones (orange dashed ellipses)
      ctx.strokeStyle = COLORS.CLEARED;
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      
      for (const zone of clearedZones) {
        const zoneCanvasX = zone.x * cellSize;
        const zoneCanvasY = (N - 1 - zone.y) * cellSize;
        const zoneWidth = zone.width * cellSize;
        const zoneHeight = zone.height * cellSize;
        
        ctx.beginPath();
        ctx.ellipse(zoneCanvasX, zoneCanvasY, zoneWidth / 2, zoneHeight / 2, 0, 0, 2 * Math.PI);
        ctx.stroke();
      }
      
      // Reset line dash
      ctx.setLineDash([]);
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
      
      // Draw mitigation zones on regular grid (inline)
      // Draw retardant zones (cyan dashed circles)
      ctx.strokeStyle = COLORS.RETARDANT;
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      
      for (const zone of retardantZones) {
        const zoneCanvasX = zone.x * cellSize;
        const zoneCanvasY = (N - 1 - zone.y) * cellSize;
        const zoneCanvasRadius = zone.radius * cellSize;
        
        ctx.beginPath();
        ctx.arc(zoneCanvasX, zoneCanvasY, zoneCanvasRadius, 0, 2 * Math.PI);
        ctx.stroke();
      }
      
      // Draw cleared zones (orange dashed ellipses)
      ctx.strokeStyle = COLORS.CLEARED;
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      
      for (const zone of clearedZones) {
        const zoneCanvasX = zone.x * cellSize;
        const zoneCanvasY = (N - 1 - zone.y) * cellSize;
        const zoneWidth = zone.width * cellSize;
        const zoneHeight = zone.height * cellSize;
        
        ctx.beginPath();
        ctx.ellipse(zoneCanvasX, zoneCanvasY, zoneWidth / 2, zoneHeight / 2, 0, 0, 2 * Math.PI);
        ctx.stroke();
      }
      
      // Reset line dash
      ctx.setLineDash([]);
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
  }, [grid, mask, gridShape, cellSize, monteCarloMode, monteCarloData, retardantZones, clearedZones]);

  // Handle canvas click
  const handleCanvasClick = useCallback((e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const canvasX = (e.clientX - rect.left) * scaleX;
    const canvasY = (e.clientY - rect.top) * scaleY;
    const x = Math.floor(canvasX / cellSize);
    const N = gridShape[0];
    // Convert canvas Y to grid Y (flip coordinate system)
    const y = N - 1 - Math.floor(canvasY / cellSize);
    
    // Handle based on mitigation mode
    if (mitigationMode === 'Retardant') {
      // Add retardant zone
      setRetardantZones([...retardantZones, { y, x, radius: RETARDANT_RADIUS }]);
      console.log(`Retardant applied at (${y}, ${x}) with radius ${RETARDANT_RADIUS}`);
    } else if (mitigationMode === 'Clear') {
      // Add cleared zone
      setClearedZones([...clearedZones, { y, x, width: CLEAR_WIDTH, height: CLEAR_HEIGHT }]);
      console.log(`Trees cleared at (${y}, ${x}) with size ${CLEAR_WIDTH}x${CLEAR_HEIGHT}`);
    } else {
      // Default: add fire
      addFire(y, x);
    }
  }, [mode, isRunning, monteCarloMode, cellSize, addFire, mitigationMode, retardantZones, clearedZones, gridShape]);


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
  }, [initializeSimulation]);

  // Handle parameter changes - reset simulation
  const handleReset = () => {
    setCustomFires([]);
    setRetardantZones([]);
    setClearedZones([]);
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
      
      <div className="simulation-container ">

        <div className="canvas-container">
          <canvas
            ref={canvasRef}
            width={canvasWidth}
            height={canvasHeight}
            onClick={handleCanvasClick}
            style={{ 
              cursor: mitigationMode !== 'None' ? 'crosshair' : (mode === 'custom' && !isRunning && !monteCarloMode ? 'crosshair' : 'default'),
              border: '2px solid #333'
            }}
          />
          

          <div className={`status ${!monteCarloMode && 'hidden'}`}>
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
          
          <h3>Mitigation Strategy</h3>
          <div className="control-group">
            {['None', 'Retardant', 'Clear'].map((mitMode) => (
              <label key={mitMode}>
                <input
                  type="radio"
                  value={mitMode}
                  checked={mitigationMode === mitMode}
                  onChange={(e) => setMitigationMode(e.target.value)}
                  disabled={isRunning || isLoadingMC}
                />
                {mitMode === 'Clear' ? 'Clear Trees' : mitMode}
              </label>
            ))}
          </div>
          {mitigationMode !== 'None' && (
            <div className="mitigation-info" style={{ fontSize: '0.85em', color: '#666', marginTop: '5px' }}>
              {mitigationMode === 'Retardant' && (
                <p>Click to apply fire retardant (cyan circles). Reduces fire spread by 80%.</p>
              )}
              {mitigationMode === 'Clear' && (
                <p>Click to clear vegetation (orange ellipses). Creates firebreaks.</p>
              )}
            </div>
          )}
          {(retardantZones.length > 0 || clearedZones.length > 0) && (
            <div className="mitigation-status" style={{ fontSize: '0.85em', marginTop: '5px' }}>
              <p>Retardant zones: {retardantZones.length} | Cleared zones: {clearedZones.length}</p>
            </div>
          )}
          
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
          <li><strong>Retardant:</strong> Click to apply fire retardant (reduces spread by 80%)</li>
          <li><strong>Clear Trees:</strong> Click to clear vegetation and create firebreaks</li>
          <li>Adjust parameters to see how vegetation density, dryness, and wind affect fire spread</li>
        </ul>
      </div>
    </div>
  );
};

export default WildfireSimulation;