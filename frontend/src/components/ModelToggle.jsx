export default function ModelToggle({ models, active, onChange, status }) {
  const toggle = (id) => {
    onChange(active.includes(id) ? active.filter(m => m !== id) : [...active, id]);
  };
  const typeLabel = { baseline: "Baseline", diffusion: "Gaussian Diffusion",
                      "graph-diff": "Graph Diffusion", gnn: "GNN" };

  return (
    <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 20 }}>
      {models.map(m => {
        const on = active.includes(m.id);
        const loaded = status[m.id];
        return (
          <button key={m.id} onClick={() => toggle(m.id)}
                  title={typeLabel[m.type]}
                  style={{
                    background: on ? m.color + "22" : "#1E293B",
                    border: `1px solid ${on ? m.color : "#334155"}`,
                    color: on ? m.color : "#475569",
                    borderRadius: 20, padding: "5px 14px", fontSize: 12,
                    fontWeight: 600, cursor: "pointer",
                    display: "flex", alignItems: "center", gap: 6,
                    transition: "all 0.15s",
                  }}>
            <span style={{ width: 6, height: 6, borderRadius: "50%",
                           background: loaded ? "#22C55E" : "#475569" }} />
            {m.label}
          </button>
        );
      })}
    </div>
  );
}
