export default function UserSelector({ users, selected, onSelect }) {
  return (
    <div>
      <label style={{ display: "block", fontSize: 11, color: "#64748B", marginBottom: 4 }}>
        TEST USER
      </label>
      <select value={selected ?? ""} onChange={e => onSelect(Number(e.target.value))}
              style={{ background: "#1E293B", color: "#E2E8F0",
                       border: "1px solid #334155", borderRadius: 6,
                       padding: "8px 12px", fontSize: 13, minWidth: 220 }}>
        <option value="">— select a user —</option>
        {users.map(u => (
          <option key={u.internal_id} value={u.internal_id}>
            #{u.internal_id} · {u.n_test} test items
          </option>
        ))}
      </select>
    </div>
  );
}
