import { useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";

import "./App.css";

function formatTimestamp(timestamp) {
  if (!timestamp) return "--";

  const date = new Date(timestamp);

  return date.toLocaleString("en-IN", {
    dateStyle: "short",
    timeStyle: "medium",
    hour12: false,
  });
}

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showAllAnomalies, setShowAllAnomalies] = useState(false);

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];

    setError("");
    setPredictions([]);

    if (!file) {
      setSelectedFile(null);
      return;
    }

    if (!file.name.toLowerCase().endsWith(".csv")) {
      setSelectedFile(null);
      setError("Please select a CSV file.");
      return;
    }

    setSelectedFile(file);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError("Please choose an accelerometer CSV file first.");
      return;
    }

    setLoading(true);
    setError("");
    setShowAllAnomalies(false);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(
        "http://127.0.0.1:8000/predict",
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "The movement analysis failed."
        );
      }

      setPredictions(data.predictions || []);
    } catch (err) {
      setPredictions([]);
      setError(
        err.message ||
          "Unable to connect to the movement analysis service."
      );
    } finally {
      setLoading(false);
    }
  };

  const totalEpochs = predictions.length;

const anomalousEpochs = predictions.filter(
  (item) => item.is_anomaly
).length;

const normalEpochs = totalEpochs - anomalousEpochs;

const normalPercentage =
  totalEpochs > 0
    ? ((normalEpochs / totalEpochs) * 100).toFixed(1)
    : "0.0";

const anomalousPercentage =
  totalEpochs > 0
    ? ((anomalousEpochs / totalEpochs) * 100).toFixed(1)
    : "0.0";

const chartData = predictions.map((item) => ({
  epoch: item.epoch,
  time: formatTimestamp(item.timestamp),
  score: Number(item.anomaly_score),
  threshold: Number(item.threshold),
}));

const anomalousPredictions = predictions.filter(
  (item) => item.is_anomaly
);

const visibleAnomalies = showAllAnomalies
  ? anomalousPredictions
  : anomalousPredictions.slice(0, 5);

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>Sleep Cycle Monitor</h1>
          <p>Movement anomaly detection during sleep</p>
        </div>

        <div className="system-status">
          <span className="status-dot"></span>
          System Ready
        </div>
      </header>

      <main className="dashboard">

        {/* SUMMARY */}
        <section className="summary-grid">

          <div className="card">
            <h3>Total Periods</h3>
            <div className="metric">
              {totalEpochs}
            </div>
            <p>30-second movement periods analyzed</p>
          </div>

          <div className="card">
            <h3>Typical Movement</h3>
            <div className="metric">
              {normalEpochs}
            </div>
            <p>
              {normalPercentage}% of analyzed periods
            </p>
          </div>

          <div className="card">
            <h3>Unusual Movement</h3>
            <div className="metric">
              {anomalousEpochs}
            </div>
            <p>
              {anomalousPercentage}% of analyzed periods
            </p>
          </div>

          <div className="card">
            <h3>Model Status</h3>
            <div className="metric">
              Ready
            </div>
            <p>Isolation Forest available</p>
          </div>

        </section>

        {/* UPLOAD */}
        <section className="panel upload-panel">
          <div>
            <h2>Upload Movement Data</h2>

            <p>
              Upload an accelerometer CSV containing
              Timestamp, x, y, and z measurements.
            </p>
          </div>

          <span className="upload-badge">
            CSV Input
          </span>

          <label className="upload-button">
            Choose CSV File
            <input
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              hidden
            />
          </label>

          {selectedFile && (
            <div className="file-success">
              <strong>Selected file:</strong>{" "}
              {selectedFile.name}
            </div>
          )}

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}

          <button
            className="analyze-button"
            onClick={handleAnalyze}
            disabled={!selectedFile || loading}
          >
            {loading
              ? "Analyzing Movement..."
              : "Analyze Movement"}
          </button>
        </section>

        {/* RESULTS */}
        {totalEpochs > 0 && (
          <>
            <section className="panel results-panel">

              <div className="results-header">
                <div>
                  <h2>Movement Analysis Results</h2>
                  <p>
                    Results generated from the uploaded
                    accelerometer data.
                  </p>
                </div>

                <span className="analysis-complete">
                  Analysis Complete
                </span>
              </div>

              <div className="result-summary-grid">

                <div className="result-card">
                  <span>Total Periods</span>
                  <strong>{totalEpochs}</strong>
                  <small>
                    30-second periods analyzed
                  </small>
                </div>

                <div className="result-card">
                  <span>Typical Movement</span>
                  <strong>{normalEpochs}</strong>
                  <small>
                    {normalPercentage}% of periods
                  </small>
                </div>

                <div className="result-card">
                  <span>Unusual Movement</span>
                  <strong>{anomalousEpochs}</strong>
                  <small>
                    {anomalousPercentage}% of periods
                  </small>
                </div>

              </div>

              {/* USER-FRIENDLY EXPLANATION */}
              <div className="model-explanation">

                <h3>What does this mean?</h3>

                <p>
                  The model identified{" "}
                  <strong>
                    {anomalousEpochs} unusual movement
                    {anomalousEpochs === 1 ? " period" : " periods"}
                  </strong>{" "}
                  during the recording.
                </p>

                <p>
                  An unusual movement period means the
                  movement pattern was different from the
                  patterns considered typical by the trained
                  model.
                </p>

                <p>
                  This does <strong>not necessarily mean
                  you woke up</strong>. The result indicates
                  unusual movement, not a confirmed
                  awakening or sleep disorder.
                </p>

              </div>

            </section>

            {/* GRAPH + TABLE */}
            <section className="content-grid">

              <div className="panel">

                <h2>Movement Pattern Over Time</h2>

                <p>
                  The graph shows how unusual each
                  30-second movement period was during the
                  recording.
                </p>

                <div className="chart-container">

                  <ResponsiveContainer
                    width="100%"
                    height={420}
                  >
                    <LineChart
                      data={chartData}
                      margin={{
                        top: 20,
                        right: 20,
                        left: 10,
                        bottom: 20,
                      }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />

                      <XAxis
                        dataKey="time"
                        interval="preserveStartEnd"
                        label={{
                          value:
                            "Time into recording",
                          position: "insideBottom",
                          offset: -10,
                        }}
                      />

                      <YAxis
                        domain={[0, "auto"]}
                        label={{
                          value:
                            "Movement anomaly score",
                          angle: -90,
                          position: "insideLeft",
                        }}
                      />

                      <Tooltip
                        formatter={(value, name) => {
                          if (name === "score") {
                            return [
                              Number(value).toFixed(4),
                              "Movement score",
                            ];
                          }

                          if (name === "threshold") {
                            return [
                              Number(value).toFixed(4),
                              "Detection threshold",
                            ];
                          }

                          return [value, name];
                        }}
                        labelFormatter={(label) =>
                          `Time: ${label}`
                        }
                      />

                      <ReferenceLine
                        y={
                          predictions[0]?.threshold ??
                          0
                        }
                        strokeDasharray="5 5"
                        label="Detection threshold"
                      />

                      <Line
                        type="monotone"
                        dataKey="score"
                        stroke="#2f80c9"
                        strokeWidth={2}
                        dot={false}
                        name="Movement score"
                      />

                    </LineChart>
                  </ResponsiveContainer>

                </div>

                <div className="chart-explanation">
                  <strong>
                    How to read this graph:
                  </strong>

                  <p>
                    Points above the detection threshold
                    represent movement periods that the
                    model classified as unusual.
                  </p>
                </div>

              </div>

              {/* ANOMALY TABLE */}
              <div className="panel">

                <h2>Unusual Movement Periods</h2>

                <div className="anomaly-count-box">
                  <strong>
                    {anomalousEpochs}
                  </strong>

                  <div>
                    <h3>
                      Unusual periods detected
                    </h3>

                    <p>
                      These periods had movement patterns
                      that crossed the model's detection
                      threshold.
                    </p>
                  </div>
                </div>

                {anomalousPredictions.length > 0 ? (
                  <>
                    <div className="table-wrapper">

                      <table>
                        <thead>
                          <tr>
                            <th>Time</th>
                            <th>Movement Score</th>
                            <th>Result</th>
                          </tr>
                        </thead>

                      <tbody>
                        {visibleAnomalies.map((item) => (
                          <tr key={item.epoch}>

                            <td>
                              {formatTimestamp(item.timestamp)}
                            </td>

                            <td>
                              {Number(item.anomaly_score).toFixed(4)}
                            </td>

                            <td>
                            <strong>
                              Unusual movement
                            </strong>
                          </td>

                        </tr>
                      ))}
                    </tbody>
                  </table>

                </div>

                {anomalousPredictions.length > 5 && (
                  <div className="anomaly-toggle">

                    {!showAllAnomalies ? (
                      <button
                        type="button"
                        onClick={() => setShowAllAnomalies(true)}
                        className="see-more-button"
                      >
                        See More
                        <span>
                          ({anomalousPredictions.length - 5} more)
                        </span>
                      </button>
                    ) : (
                      <button
                        type="button"
                        onClick={() => setShowAllAnomalies(false)}
                        className="see-more-button"
                      >
                        See Less
                      </button>
                    )}

                   </div>
                )}
              </>
            ) : (
              <div className="no-anomalies">
                No unusual movement periods were detected.
              </div>
            )}

              </div>

            </section>
          </>
        )}

        {/* BEFORE ANALYSIS */}
        {totalEpochs === 0 && !loading && (
          <section className="panel empty-state">

            <h2>
              Your movement analysis will appear here
            </h2>

            <p>
              Upload an accelerometer CSV and select
              <strong> Analyze Movement </strong>
              to see movement patterns and unusual
              periods detected by the model.
            </p>

          </section>
        )}

      </main>
    </div>
  );
}

export default App;