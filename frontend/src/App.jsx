import { useState, useEffect } from "react";
import UserSelector from "./components/UserSelector";
import ModelToggle from "./components/ModelToggle";
import ResultsGrid from "./components/ResultsGrid";
import MetricsDashboard from "./components/MetricsDashboard";
import { fetchRecommendations, fetchSampleUsers, fetchStatus, fetchMetrics } from "./api";

const ALL_MODELS = [
  { id: "neumf",    label: "NeuMF",     type: "baseline",   color: "#F59E0B" },
  { id: "diffrec",  label: "DiffRec",   type: "diffusion",  color: "#3B82F6" },
  { id: "ldiffrec", label: "L-DiffRec", type: "diffusion",  color: "#60A5FA" },
  { id: "giffcf",   label: "GiffCF",    type: "graph-diff", color: "#10B981" },
  { id: "cfdiff",   label: "CF-Diff",   type: "graph-diff", color: "#34D399" },
  { id: "gdmcf",    label: "GDMCF",     type: "graph-diff", color: "#6EE7B7" },
  { id: "lightgcn", label: "LightGCN",  type: "gnn",        color: "#8B5CF6" },
];

export default function App() {
  const [users,        setUsers]        = useState([]);
  const [selectedUser, setSelectedUser] = useState(null);
  const [activeModels, setActiveModels] = useState(ALL_MODELS.map(m => m.id));
  const [results,      setResults]      = useState({});
  const [metrics,      setMetrics]      = useState(null);
  const [loading,      setLoading]      = useState({});
  const [status,       setStatus]       = useState({});
  const [topK,         setTopK]         = useState(10);

  useEffect(() => {
    fetchSampleUsers().then(setUsers).catch(console.error);
    fetchMetrics().then(setMetrics).catch(console.error);
    fetchStatus().then(s => setStatus(s.models || {})).catch(console.error);
  }, []);

  async function handleRun() {
    if (!selectedUser) return;
    // Run all active models concurrently, populate panels as each resolves
    const pending = {};
    activeModels.forEach(m => { pending[m] = true; });
    setLoading(pending);
    setResults({});

    await Promise.allSettled(
      activeModels.map(async (modelId) => {
        try {
          const data = await fetchRecommendations(selectedUser, [modelId], topK);
          setResults(prev => ({
            ...prev,
            [modelId]: data.results[modelId] || { recommendations: [], latency_ms: 0 },
          }));
        } catch (e) {
          setResults(prev => ({ ...prev, [modelId]: { error: e.message } }));
        } finally {
          setLoading(prev => { const n = {...prev}; delete n[modelId]; return n; });
        }
      })
    );
  }

  return (
    <div style={{ fontFamily: "'IBM Plex Mono', monospace", background: "#0F172A",
                  minHeight: "100vh", color: "#E2E8F0", padding: "0" }}>
      {/* Header */}
      <header style={{ background: "#1E293B", borderBottom: "1px solid #334155",
                       padding: "16px 32px", display: "flex",
                       alignItems: "center", gap: 16 }}>
        <div style={{ fontSize: 22, fontWeight: 700, color: "#38BDF8",
                      letterSpacing: "-0.5px" }}>
          ⬡ DiffRec Study
        </div>
        <div style={{ fontSize: 12, color: "#64748B", flex: 1 }}>
          Comparative Evaluation · Amazon Books · 7 Models
        </div>
        <div style={{ fontSize: 11, color: "#475569" }}>
          device: {status._device || "…"}
        </div>
      </header>

      <div style={{ padding: "24px 32px" }}>
        {/* Controls */}
        <div style={{ display: "flex", gap: 16, alignItems: "flex-end",
                      flexWrap: "wrap", marginBottom: 24 }}>
          <UserSelector users={users} selected={selectedUser}
                        onSelect={setSelectedUser} />
          <div>
            <label style={{ display: "block", fontSize: 11, color: "#64748B",
                            marginBottom: 4 }}>TOP-K</label>
            <select value={topK} onChange={e => setTopK(Number(e.target.value))}
                    style={{ background: "#1E293B", color: "#E2E8F0",
                             border: "1px solid #334155", borderRadius: 6,
                             padding: "8px 12px", fontSize: 13 }}>
              {[5, 10, 20].map(k => <option key={k} value={k}>{k}</option>)}
            </select>
          </div>
          <button onClick={handleRun} disabled={!selectedUser}
                  style={{ background: selectedUser ? "#0EA5E9" : "#334155",
                           color: selectedUser ? "#fff" : "#64748B",
                           border: "none", borderRadius: 6, padding: "9px 24px",
                           fontSize: 13, fontWeight: 600, cursor: selectedUser ? "pointer" : "not-allowed",
                           transition: "background 0.2s" }}>
            ▶ Run Inference
          </button>
        </div>

        <ModelToggle models={ALL_MODELS} active={activeModels}
                     onChange={setActiveModels} status={status} />

        {/* Results */}
        {Object.keys(results).length > 0 || Object.keys(loading).length > 0 ? (
          <ResultsGrid models={ALL_MODELS.filter(m => activeModels.includes(m.id))}
                       results={results} loading={loading} />
        ) : (
          <div style={{ textAlign: "center", padding: "80px 0", color: "#334155",
                        fontSize: 14 }}>
            Select a user and click Run Inference to compare recommendations.
          </div>
        )}

        {/* Metrics */}
        {metrics && <MetricsDashboard metrics={metrics} models={ALL_MODELS} />}
      </div>
    </div>
  );
}
