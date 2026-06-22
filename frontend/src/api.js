// api.js – typed API layer for the Flask backend

const BASE = "/api";

export async function fetchRecommendations(userId, models, topK = 10, dataset = "videogames") {
  const res = await fetch(`${BASE}/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, top_k: topK, models, dataset }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function fetchMetrics(dataset = "videogames") {
  const res = await fetch(`${BASE}/metrics?dataset=${dataset}`);
  if (!res.ok) throw new Error("metrics not found — run evaluate.py first");
  return res.json();
}

export async function fetchSampleUsers() {
  const res = await fetch(`${BASE}/users/sample`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function fetchStatus(dataset = "videogames") {
  const res = await fetch(`${BASE}/status?dataset=${dataset}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}
