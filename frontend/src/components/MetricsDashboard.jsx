import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid } from "recharts";

export default function MetricsDashboard({ metrics, models }) {
  const colorMap = Object.fromEntries(models.map(m => [m.id, m.color]));
  const metricKeys = ["Recall@10", "NDCG@10", "MRR"];

  const chartData = metricKeys.map(key => {
    const entry = { metric: key };
    Object.entries(metrics).forEach(([model, vals]) => {
      entry[model] = +(vals[key] ?? 0).toFixed(4);
    });
    return entry;
  });

  return (
    <div style={{ background: "#1E293B", borderRadius: 8,
                  border: "1px solid #334155", padding: "20px 24px" }}>
      <div style={{ fontSize: 14, fontWeight: 700, color: "#38BDF8",
                    marginBottom: 16 }}>
        Evaluation Metrics (Leave-One-Out, 99 Negatives)
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={chartData} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
          <XAxis dataKey="metric" tick={{ fill: "#94A3B8", fontSize: 12 }} />
          <YAxis tick={{ fill: "#64748B", fontSize: 11 }} domain={[0, "auto"]} />
          <Tooltip contentStyle={{ background: "#0F172A", border: "1px solid #334155",
                                    borderRadius: 6, color: "#E2E8F0" }} />
          <Legend wrapperStyle={{ fontSize: 12, color: "#94A3B8" }} />
          {Object.keys(metrics).map(modelId => (
            <Bar key={modelId} dataKey={modelId}
                 fill={colorMap[modelId] || "#64748B"}
                 radius={[3, 3, 0, 0]} />
          ))}
        </BarChart>
      </ResponsiveContainer>

      {/* Table */}
      <table style={{ width: "100%", marginTop: 20, borderCollapse: "collapse",
                      fontSize: 12 }}>
        <thead>
          <tr style={{ borderBottom: "1px solid #334155" }}>
            <th style={{ textAlign: "left", padding: "6px 8px", color: "#64748B" }}>Model</th>
            {metricKeys.map(k => (
              <th key={k} style={{ textAlign: "right", padding: "6px 8px", color: "#64748B" }}>
                {k}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Object.entries(metrics).map(([modelId, vals]) => (
            <tr key={modelId} style={{ borderBottom: "1px solid #1E293B" }}>
              <td style={{ padding: "6px 8px", color: colorMap[modelId] || "#E2E8F0",
                           fontWeight: 600 }}>
                {models.find(m => m.id === modelId)?.label || modelId}
              </td>
              {metricKeys.map(k => (
                <td key={k} style={{ textAlign: "right", padding: "6px 8px",
                                     color: "#94A3B8" }}>
                  {(vals[k] ?? 0).toFixed(4)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
