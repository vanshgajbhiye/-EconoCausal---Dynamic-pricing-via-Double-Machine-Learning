const API_BASE = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

async function request(path, options = {}) {
  if (!API_BASE) throw new Error("Backend is not configured. Add VITE_API_BASE_URL to your .env file.");
  const response = await fetch(`${API_BASE}${path}`, options);
  const contentType = response.headers.get("content-type") || "";
  const body = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) throw new Error(body?.message || body?.error || `Request failed (${response.status})`);
  return body;
}

export const api = {
  uploadDataset(file, mapping = {}) {
    const form = new FormData();
    form.append("file", file);
    form.append("mapping", JSON.stringify(mapping));
    return request("/datasets", { method: "POST", body: form });
  },
  runAnalysis(payload) {
    return request("/analysis", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  },
  getDataset(id) {
    return request(`/datasets/${encodeURIComponent(id)}`);
  },
  getReport(id) {
    return request(`/reports/${encodeURIComponent(id)}`);
  }
};