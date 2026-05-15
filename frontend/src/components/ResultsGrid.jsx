import BookCard from "./BookCard";

function ModelPanel({ model, data, isLoading }) {
  const typeColors = {
    baseline: "#F59E0B", diffusion: "#3B82F6",
    "graph-diff": "#10B981", gnn: "#8B5CF6",
  };
  const typeLabel = {
    baseline: "Baseline", diffusion: "Gaussian Diff.",
    "graph-diff": "Graph Diff.", gnn: "GNN",
  };

  return (
    <div style={{ background: "#1E293B", borderRadius: 8,
                  border: `1px solid ${model.color}33`, overflow: "hidden",
                  minWidth: 0 }}>
      {/* Panel header */}
      <div style={{ padding: "10px 14px", borderBottom: "1px solid #0F172A",
                    display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{ fontWeight: 700, fontSize: 13, color: model.color }}>
          {model.label}
        </span>
        <span style={{ fontSize: 10, background: model.color + "22",
                       color: model.color, padding: "2px 7px", borderRadius: 10 }}>
          {typeLabel[model.type]}
        </span>
        {data?.latency_ms != null && (
          <span style={{ marginLeft: "auto", fontSize: 10, color: "#475569" }}>
            {data.latency_ms}ms
          </span>
        )}
      </div>

      {/* Body */}
      <div style={{ padding: "8px 14px" }}>
        {isLoading && (
          <div style={{ textAlign: "center", padding: "40px 0",
                        color: "#334155", fontSize: 12 }}>
            Running inference…
          </div>
        )}
        {!isLoading && data?.error && (
          <div style={{ color: "#EF4444", fontSize: 11, padding: "12px 0" }}>
            ✕ {data.error}
          </div>
        )}
        {!isLoading && data?.recommendations?.map(rec => (
          <BookCard key={rec.item_id} rec={rec} modelColor={model.color} />
        ))}
        {!isLoading && !data && (
          <div style={{ color: "#334155", fontSize: 11, padding: "20px 0",
                        textAlign: "center" }}>
            No checkpoint loaded
          </div>
        )}
      </div>
    </div>
  );
}

export default function ResultsGrid({ models, results, loading }) {
  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: `repeat(${Math.min(models.length, 4)}, minmax(200px, 1fr))`,
      gap: 16, marginBottom: 40,
    }}>
      {models.map(m => (
        <ModelPanel key={m.id} model={m}
                    data={results[m.id]}
                    isLoading={!!loading[m.id]} />
      ))}
    </div>
  );
}
