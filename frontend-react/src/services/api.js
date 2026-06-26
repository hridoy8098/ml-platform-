const API = import.meta.env.VITE_API_URL || '';
const TIMEOUT_MS = 30000;

async function apiFetch(path, opts = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(API + path, {
      headers: { 'Content-Type': 'application/json', ...opts.headers },
      signal: controller.signal,
      ...opts,
    });
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      let msg;
      try { const j = JSON.parse(text); msg = j.message || j.error || res.statusText; } catch { msg = text || res.statusText; }
      return { success: false, message: msg, status: res.status };
    }
    return res.json();
  } catch (err) {
    if (err.name === 'AbortError') {
      return { success: false, message: `Request timed out after ${TIMEOUT_MS / 1000}s` };
    }
    return { success: false, message: err.message || 'Network error' };
  } finally {
    clearTimeout(timeout);
  }
}

function apiPost(path, body) {
  return apiFetch(path, { method: 'POST', body: JSON.stringify(body) });
}

export async function uploadMLFile(file) {
  const fd = new FormData();
  fd.append('file', file);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(API + '/ml/upload', { method: 'POST', body: fd, signal: controller.signal });
    if (!res.ok) return { success: false, message: `Upload failed (${res.status})` };
    return res.json();
  } catch (err) {
    if (err.name === 'AbortError') return { success: false, message: 'Upload timed out' };
    return { success: false, message: err.message || 'Upload failed' };
  } finally {
    clearTimeout(timeout);
  }
}

export async function loadMLSample(name) {
  return apiPost('/ml/sample', { name });
}

export async function cleanMLData(sessionId, operations) {
  return apiPost('/ml/clean', { session_id: sessionId, operations });
}

export async function analyzeMLData(sessionId) {
  return apiPost('/ml/analyze', { session_id: sessionId });
}

export async function trainMLModel(params) {
  return apiPost('/ml/train', params);
}

export async function evaluateMLModel(sessionId, modelId) {
  return apiPost('/ml/evaluate', { session_id: sessionId, model_id: modelId });
}

export async function predictML(modelId, input) {
  return apiPost('/ml/predict', { model_id: modelId, input });
}

export async function listMLModels() {
  return apiFetch('/ml/models');
}

export async function deleteMLModel(modelId) {
  return apiPost('/ml/delete-model', { model_id: modelId });
}

export async function getMLModelInfo(modelId) {
  return apiFetch(`/ml/model-info/${modelId}`);
}

export async function getMLJobStatus(jobId) {
  return apiFetch(`/ml/job/${jobId}`);
}

export async function clusterMLData(params) {
  return apiPost('/ml/cluster', params);
}

export async function dimReduceML(params) {
  return apiPost('/ml/dim-reduce', params);
}

export async function trainMLNeural(params) {
  return apiPost('/ml/neural', params);
}

export async function tuneMLHyperparams(params) {
  return apiPost('/ml/tune', params);
}

export async function kfoldML(params) {
  return apiPost('/ml/kfold', params);
}

export async function featureEngineerML(params) {
  return apiPost('/ml/feature-engineer', params);
}

export async function getMLChartData(params) {
  return apiPost('/ml/chart-data', params);
}

export async function getMLConfusionMatrix(sessionId, modelId) {
  return apiPost('/ml/confusion-matrix', { session_id: sessionId, model_id: modelId });
}

export async function getMLROCCurve(sessionId, modelId) {
  return apiPost('/ml/roc-curve', { session_id: sessionId, model_id: modelId });
}

export async function getMLElbowPlot(sessionId, maxK = 10) {
  return apiPost('/ml/elbow-plot', { session_id: sessionId, max_k: maxK });
}

export async function getMLFeatureImportance(modelId) {
  return apiPost('/ml/feature-importance', { model_id: modelId });
}

export async function exportMLModel(modelId, format = 'joblib') {
  return apiPost('/ml/export-model', { model_id: modelId, format });
}
