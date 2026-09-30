import "./App.css";
import { useState } from "react";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const sleepData = [
  { day: "Mon", hours: 7.2 },
  { day: "Tue", hours: 6.5 },
  { day: "Wed", hours: 8.1 },
  { day: "Thu", hours: 7.0 },
  { day: "Fri", hours: 6.8 },
  { day: "Sat", hours: 8.3 },
  { day: "Sun", hours: 7.7 },
];

function App() {
  const [showAnalysis, setShowAnalysis] = useState(false);
  const [submittedData, setSubmittedData] = useState(null);

  const [formData, setFormData] = useState({
    sleepDuration: "",
    sleepQuality: "",
    movementLevel: "",
    sleepStages: [],
    bedtime: "",
    wakeupTime: "",
  });

  const handleChange = (event) => {
    const { name, value } = event.target;

    setFormData((previous) => ({
      ...previous,
      [name]: value,
    }));
  };

  const handleStageChange = (event) => {
    const { value, checked } = event.target;

    setFormData((previous) => ({
      ...previous,
      sleepStages: checked
        ? [...previous.sleepStages, value]
        : previous.sleepStages.filter((stage) => stage !== value),
    }));
  };

  const handleSubmit = (event) => {
    event.preventDefault();

    setSubmittedData(formData);
  };

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>Sleep Cycle Monitor</h1>
          <p>Sleep anomaly detection dashboard</p>
        </div>

        <div className="status">
          <span className="status-dot"></span>
          System Ready
        </div>
      </header>

      <main className="dashboard">

        {/* SUMMARY */}
        <section className="summary-grid">
          <div className="card">
            <p className="card-label">Sleep Duration</p>
            <h2>7h 42m</h2>
            <span className="card-info">Last recorded session</span>
          </div>

          <div className="card">
            <p className="card-label">Sleep Quality</p>
            <h2>Good</h2>
            <span className="card-info">Based on recorded movement</span>
          </div>

          <div className="card">
            <p className="card-label">Anomalies</p>
            <h2>3</h2>
            <span className="card-info">Detected this session</span>
          </div>

          <div className="card">
            <p className="card-label">Model Status</p>
            <h2>Ready</h2>
            <span className="card-info">Anomaly detector available</span>
          </div>
        </section>

        {/* USER INPUT */}
        <section className="panel input-panel">
          <div className="input-heading">
            <div>
              <h2>Enter Sleep Data</h2>
              <p>
                Enter your sleep information based on your experience.
              </p>
            </div>

            <span className="input-badge">Manual Input</span>
          </div>

          <form onSubmit={handleSubmit}>

            <div className="form-grid">

              {/* Sleep Duration */}
              <div className="form-group">
                <label htmlFor="sleepDuration">
                  🕐 Sleep Duration
                </label>

                <div className="input-with-unit">
                  <input
                    id="sleepDuration"
                    type="number"
                    name="sleepDuration"
                    min="0"
                    max="24"
                    step="0.1"
                    placeholder="Example: 7.5"
                    value={formData.sleepDuration}
                    onChange={handleChange}
                    required
                  />
                  <span>hours</span>
                </div>
              </div>

              {/* Sleep Quality */}
              <div className="form-group">
                <label htmlFor="sleepQuality">
                  🛌 Sleep Quality
                </label>

                <select
                  id="sleepQuality"
                  name="sleepQuality"
                  value={formData.sleepQuality}
                  onChange={handleChange}
                  required
                >
                  <option value="">Select quality</option>
                  <option value="Poor">Poor</option>
                  <option value="Fair">Fair</option>
                  <option value="Good">Good</option>
                  <option value="Excellent">Excellent</option>
                </select>
              </div>

              {/* Movement */}
              <div className="form-group">
                <label htmlFor="movementLevel">
                  🏃 Movement / Activity Level
                </label>

                <select
                  id="movementLevel"
                  name="movementLevel"
                  value={formData.movementLevel}
                  onChange={handleChange}
                  required
                >
                  <option value="">Select activity level</option>
                  <option value="Low">Low</option>
                  <option value="Moderate">Moderate</option>
                  <option value="High">High</option>
                </select>
              </div>

              {/* Bedtime */}
              <div className="form-group">
                <label htmlFor="bedtime">
                  🌙 Bedtime
                </label>

                <input
                  id="bedtime"
                  type="time"
                  name="bedtime"
                  value={formData.bedtime}
                  onChange={handleChange}
                  required
                />
              </div>

              {/* Wake-up */}
              <div className="form-group">
                <label htmlFor="wakeupTime">
                  ⏰ Wake-up Time
                </label>

                <input
                  id="wakeupTime"
                  type="time"
                  name="wakeupTime"
                  value={formData.wakeupTime}
                  onChange={handleChange}
                  required
                />
              </div>

            </div>

            {/* Sleep Stages */}
            <div className="form-group sleep-stage-group">
              <label>🌙 Sleep Stage</label>

              <p className="field-description">
                Select all stages that you experienced.
              </p>

              <div className="stage-options">

                <label className="stage-option">
                  <input
                    type="checkbox"
                    value="Light"
                    checked={formData.sleepStages.includes("Light")}
                    onChange={handleStageChange}
                  />
                  <span>Light</span>
                </label>

                <label className="stage-option">
                  <input
                    type="checkbox"
                    value="Deep"
                    checked={formData.sleepStages.includes("Deep")}
                    onChange={handleStageChange}
                  />
                  <span>Deep</span>
                </label>

                <label className="stage-option">
                  <input
                    type="checkbox"
                    value="REM"
                    checked={formData.sleepStages.includes("REM")}
                    onChange={handleStageChange}
                  />
                  <span>REM</span>
                </label>

                <label className="stage-option">
                  <input
                    type="checkbox"
                    value="Awake"
                    checked={formData.sleepStages.includes("Awake")}
                    onChange={handleStageChange}
                  />
                  <span>Awake</span>
                </label>

              </div>
            </div>

            <button type="submit" className="analyze-button">
              Analyze Sleep
            </button>

          </form>
        </section>

        {/* USER INPUT RESULT */}
        {submittedData && (
          <section className="panel result-panel">

            <div className="result-header">
              <div>
                <h2>Sleep Data Summary</h2>
                <p>Your entered information is shown below.</p>
              </div>

              <span className="result-badge">Input Received</span>
            </div>

            <div className="result-grid">

              <div className="result-card">
                <span>Sleep Duration</span>
                <strong>{submittedData.sleepDuration} hours</strong>
              </div>

              <div className="result-card">
                <span>Sleep Quality</span>
                <strong>{submittedData.sleepQuality}</strong>
              </div>

              <div className="result-card">
                <span>Movement / Activity</span>
                <strong>{submittedData.movementLevel}</strong>
              </div>

              <div className="result-card">
                <span>Bedtime</span>
                <strong>{submittedData.bedtime}</strong>
              </div>

              <div className="result-card">
                <span>Wake-up Time</span>
                <strong>{submittedData.wakeupTime}</strong>
              </div>

              <div className="result-card">
                <span>Sleep Stage</span>
                <strong>
                  {submittedData.sleepStages.length > 0
                    ? submittedData.sleepStages.join(", ")
                    : "Not selected"}
                </strong>
              </div>

            </div>

            <div className="backend-note">
              <strong>Analysis Status</strong>
              <p>
                Your sleep data has been recorded successfully.
                Advanced anomaly analysis will be available when the
                backend model is connected.
              </p>
            </div>

          </section>
        )}

        {/* CONTENT */}
        <section className="content-grid">

          {/* GRAPH */}
          <div className="panel">
            <h2>Sleep Overview</h2>

            <p>
              Sleep duration recorded over the last 7 days.
            </p>

            <div className="chart-container">
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={sleepData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="day" />
                  <YAxis domain={[5, 9]} />
                  <Tooltip />

                  <Line
                    type="monotone"
                    dataKey="hours"
                    strokeWidth={3}
                    dot={{ r: 5 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* ANOMALY */}
          <div className="panel">
            <h2>Anomaly Detection</h2>

            <div className="anomaly-box">
              <span className="anomaly-number">3</span>

              <div>
                <strong>Anomalies detected</strong>

                <p>
                  Review unusual movement events from the sleep session.
                </p>
              </div>
            </div>

            <div className="anomaly-table">

              <div className="table-header">
                <span>Time</span>
                <span>Movement</span>
                <span>Type</span>
                <span>Severity</span>
              </div>

              <div className="table-row">
                <span>01:42 AM</span>
                <span>High</span>
                <span>Restless Movement</span>
                <span className="severity-medium">
                  Moderate
                </span>
              </div>

              <div className="table-row">
                <span>03:18 AM</span>
                <span>Low</span>
                <span>Sleep Duration</span>
                <span className="severity-low">
                  Low
                </span>
              </div>

              <div className="table-row">
                <span>05:06 AM</span>
                <span>High</span>
                <span>Repeated Movement</span>
                <span className="severity-medium">
                  Moderate
                </span>
              </div>

            </div>

            <button
              className="primary-button"
              onClick={() => setShowAnalysis(!showAnalysis)}
            >
              {showAnalysis ? "Hide Analysis" : "View Analysis"}
            </button>

            {showAnalysis && (
              <div className="analysis-panel">

                <h3>Sleep Analysis</h3>

                <div className="analysis-item">
                  <strong>Anomaly 1</strong>
                  <p>
                    Unusual movement detected during sleep.
                  </p>
                  <span>Severity: Moderate</span>
                </div>

                <div className="analysis-item">
                  <strong>Anomaly 2</strong>
                  <p>
                    Sleep duration variation detected.
                  </p>
                  <span>Severity: Low</span>
                </div>

                <div className="analysis-item">
                  <strong>Anomaly 3</strong>
                  <p>
                    Repeated movement pattern detected.
                  </p>
                  <span>Severity: Moderate</span>
                </div>

                <div className="analysis-summary">
                  <strong>Overall Result</strong>

                  <p>
                    The system detected 3 unusual sleep patterns.
                    Further monitoring is recommended.
                  </p>
                </div>

              </div>
            )}

          </div>

        </section>

      </main>
    </div>
  );
}

export default App;