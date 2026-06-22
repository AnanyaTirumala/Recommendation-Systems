import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid } from "recharts";

// Checkpoint keys are "{dataset}_{model}" — strip the prefix to get the bare model id.
function bareModelId(key) {
  return key.replace(/^[^_]+_/, "");
}

export default function MetricsDashboard({ metrics, models }) {
  const colorMap = Object.fromEntries(models.map(m => [m.id, m.color]));
  const labelMap = Object.fromEntries(models.map(m => [m.id, m.label]));
  const metricKeys = ["Recall@10", "NDCG@10", "MRR"];

  const chartData = metricKeys.map(key => {
    const entry = { metric: key };
    Object.entries(metrics).forEach(([ckptKey, vals]) => {
      entry[ckptKey] = +(vals[key] ?? 0).toFixed(4);
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
                                    borderRadius: 6, color: "#E2E8F0" }}
                   formatter={(value, ckptKey) => [
                     value,
                     labelMap[bareModelId(ckptKey)] || ckptKey,
                   ]} />
          <Legend wrapperStyle={{ fontSize: 12, color: "#94A3B8" }}
                  formatter={ckptKey => labelMap[bareModelId(ckptKey)] || ckptKey} />
          {Object.keys(metrics).map(ckptKey => (
            <Bar key={ckptKey} dataKey={ckptKey}
                 fill={colorMap[bareModelId(ckptKey)] || "#64748B"}
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
            <th style={{ textAlign: "left", padding: "6px 8px", color: "#64748B" }}>Checkpoint</th>
            {metricKeys.map(k => (
              <th key={k} style={{ textAlign: "right", padding: "6px 8px", color: "#64748B" }}>
                {k}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Object.entries(metrics).map(([ckptKey, vals]) => {
            const mid = bareModelId(ckptKey);
            return (
              <tr key={ckptKey} style={{ borderBottom: "1px solid #1E293B" }}>
                <td style={{ padding: "6px 8px",
                             color: colorMap[mid] || "#E2E8F0", fontWeight: 600 }}>
                  {labelMap[mid] || mid}
                </td>
                <td style={{ padding: "6px 8px", color: "#475569", fontFamily: "monospace" }}>
                  {ckptKey}.pt
                </td>
                {metricKeys.map(k => (
                  <td key={k} style={{ textAlign: "right", padding: "6px 8px",
                                       color: "#94A3B8" }}>
                    {(vals[k] ?? 0).toFixed(4)}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
