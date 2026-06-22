export default function ItemCard({ rec, modelColor }) {
  const { rank, score, item } = rec;
  const barWidth = `${Math.round(score * 100)}%`;
  return (
    <div style={{ display: "flex", gap: 10, padding: "8px 0",
                  borderBottom: "1px solid #1E293B", alignItems: "flex-start" }}>
      <span style={{ fontSize: 11, color: "#475569", minWidth: 20,
                     paddingTop: 2, textAlign: "right" }}>
        {rank}
      </span>
      {item.image_url ? (
        <img src={item.image_url} alt="" width={36} height={52}
             style={{ objectFit: "cover", borderRadius: 3, flexShrink: 0 }}
             onError={e => { e.target.style.display = "none"; }} />
      ) : (
        <div style={{ width: 36, height: 52, background: "#1E293B",
                      borderRadius: 3, flexShrink: 0 }} />
      )}
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: "#E2E8F0",
                      whiteSpace: "nowrap", overflow: "hidden",
                      textOverflow: "ellipsis" }} title={item.title}>
          {item.title || `Item #${item.internal_id}`}
        </div>
        <div style={{ fontSize: 10, color: "#94A3B8", marginBottom: 2,
                      whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
          ID: {item.asin || item.internal_id}
        </div>
        <div style={{ fontSize: 11, color: "#64748B", marginBottom: 4 }}>
          {item.author ? `Dev: ${item.author}` : "—"}
        </div>
        <div style={{ background: "#1E293B", borderRadius: 2, height: 3 }}>
          <div style={{ background: modelColor, borderRadius: 2,
                        height: "100%", width: barWidth,
                        transition: "width 0.4s ease" }} />
        </div>
      </div>
    </div>
  );
}
