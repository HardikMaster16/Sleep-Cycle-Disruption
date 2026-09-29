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
  const [fileName, setFileName] = useState("");

  const handleFileUpload = (event) => {
    const file = event.target.files[0];

    if (file) {
      setFileName(file.name);
    }
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
            <span className="card-info">
              Last recorded session
            </span>
          </div>

          <div className="card">
            <p className="card-label">Sleep Quality</p>
            <h2>Good</h2>
            <span className="card-info">
              Based on recorded movement
            </span>
          </div>

          <div className="card">
            <p className="card-label">Anomalies</p>
            <h2>3</h2>
            <span className="card-info">
              Detected this session
            </span>
          </div>

          <div className="card">
            <p className="card-label">Model Status</p>
            <h2>Ready</h2>
            <span className="card-info">
              Anomaly detector available
            </span>
          </div>

        </section>

        {/* UPLOAD */}
        <section className="panel upload-panel">

          <div>
            <h2>Sleep Data</h2>
            <p>
              Upload your sleep data CSV file for anomaly analysis.
            </p>
          </div>

          <label className="upload-button">
            Choose CSV File
            <input
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
            />
          </label>

          {fileName && (
            <div className="file-success">
              ✓ Selected file: <strong>{fileName}</strong>
            </div>
          )}

        </section>

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