import { useState } from 'react';
import { motion } from 'framer-motion';
import useAtlasStore from '../../store/useAtlasStore';
import { evaluateMLModel, getMLConfusionMatrix, getMLROCCurve, getMLFeatureImportance } from '../../services/api';
import { Label, Hint, Btn, ErrorBlock, MetricCard, card, select } from '../HUDComponents';

export default function EvaluateTab() {
  const { mlModels, mlDataset, setMLEvaluationData } = useAtlasStore();

  const [selectedModel, setSelectedModel] = useState('');
  const [metric, setMetric] = useState('standard');
  const [evalResult, setEvalResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const activeModels = Array.isArray(mlModels) ? mlModels : [];

  const run = async () => {
    if (!selectedModel || !mlDataset?.session_id) return;
    setLoading(true); setError(null); setEvalResult(null);
    let r;
    if (metric === 'standard') {
      r = await evaluateMLModel(mlDataset.session_id, selectedModel);
    } else if (metric === 'confusion_matrix') {
      r = await getMLConfusionMatrix(mlDataset.session_id, selectedModel);
    } else if (metric === 'roc_curve') {
      r = await getMLROCCurve(mlDataset.session_id, selectedModel);
    } else if (metric === 'feature_importance') {
      r = await getMLFeatureImportance(selectedModel);
    }
    r?.success ? setEvalResult(r) : setError(r || { message: 'Evaluation failed' });
    setLoading(false);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={card}><Label>METRIC</Label>
        <select value={metric} onChange={e => { setMetric(e.target.value); setEvalResult(null); }} style={{ ...select, marginTop: 8 }}>
          <option value="standard">Standard Metrics</option>
          <option value="confusion_matrix">Confusion Matrix</option>
          <option value="roc_curve">ROC Curve</option>
          <option value="feature_importance">Feature Importance</option>
        </select>
      </div>
      <div style={card}><Label>SELECT MODEL</Label>
        <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)} style={{ ...select, marginTop: 8 }}>
          <option value="">— Choose a trained model —</option>
          {activeModels.map(m => <option key={m.model_id} value={m.model_id}>{m.model_id} ({m.size})</option>)}
        </select>
      </div>
      <Btn onClick={run} disabled={loading || !selectedModel}>{loading ? '◌ EVALUATING...' : '📋 RUN EVALUATION'}</Btn>
      {error && <ErrorBlock err={error} />}

      {evalResult?.success && metric === 'standard' && (
        <div style={card}>
          <Label style={{ color: 'var(--cyan)' }}>✓ EVALUATION RESULTS</Label>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 8, marginTop: 10 }}>
            {evalResult.accuracy != null && <MetricCard label="ACCURACY" value={(evalResult.accuracy * 100).toFixed(1) + '%'} color="var(--cyan)" />}
            {evalResult.precision != null && <MetricCard label="PRECISION" value={evalResult.precision} color="var(--blue)" />}
            {evalResult.recall != null && <MetricCard label="RECALL" value={evalResult.recall} color="var(--green)" />}
            {evalResult.f1 != null && <MetricCard label="F1" value={evalResult.f1} color="var(--purple)" />}
            {evalResult.rmse != null && <MetricCard label="RMSE" value={evalResult.rmse} color="var(--blue)" />}
            {evalResult.mae != null && <MetricCard label="MAE" value={evalResult.mae} color="var(--yellow)" />}
            {evalResult.r2 != null && <MetricCard label="R²" value={evalResult.r2} color="var(--green)" />}
            {evalResult.auc_roc != null && <MetricCard label="AUC ROC" value={evalResult.auc_roc} color="var(--magenta)" />}
          </div>
        </div>
      )}

      {evalResult?.success && metric === 'confusion_matrix' && evalResult.matrix && (
        <div style={card}><Label>CONFUSION MATRIX</Label>
          {evalResult.labels && (
            <table style={{ borderCollapse: 'collapse', marginTop: 10, fontSize: 10, fontFamily: 'var(--font-mono)' }}>
              <thead><tr><td style={{ padding: 4, color: 'var(--text-3)' }} />{evalResult.labels.map(l => <th key={l} style={{ padding: '4px 8px', color: 'var(--cyan)', borderBottom: '1px solid var(--border)' }}>{l}</th>)}</tr></thead>
              <tbody>{evalResult.matrix.map((row, i) => (
                <tr key={i}><td style={{ padding: '4px 8px', color: 'var(--magenta)', borderRight: '1px solid var(--border)' }}>{evalResult.labels?.[i] || i}</td>
                  {row.map((v, j) => <td key={j} style={{ padding: '6px 10px', textAlign: 'center', background: v > 0 ? 'rgba(0,240,255,0.08)' : 'transparent', color: v > 0 ? 'var(--cyan)' : 'var(--text-3)' }}>{v}</td>)}
                </tr>
              ))}</tbody>
            </table>
          )}
        </div>
      )}

      {evalResult?.success && metric === 'roc_curve' && evalResult.fpr && (
        <div style={card}><Label>ROC CURVE {evalResult.auc != null && <span style={{ color: 'var(--cyan)' }}>(AUC: {evalResult.auc.toFixed(3)})</span>}</Label>
          <svg width={280} height={160} style={{ marginTop: 10 }}>
            <line x1={20} y1={140} x2={260} y2={140} stroke="rgba(0,240,255,0.15)" />
            <line x1={20} y1={140} x2={20} y2={20} stroke="rgba(0,240,255,0.15)" />
            <line x1={20} y1={140} x2={260} y2={20} stroke="rgba(0,240,255,0.1)" strokeDasharray="3" />
            <polyline fill="none" stroke="#00ffc8" strokeWidth={1.5}
              points={evalResult.fpr.map((fpr, i) => {
                const x = 20 + fpr * 240, y = 140 - evalResult.tpr[i] * 120;
                return `${x},${y}`;
              }).join(' ')} />
          </svg>
        </div>
      )}

      {evalResult?.success && metric === 'feature_importance' && evalResult.data && (
        <div style={card}><Label>FEATURE IMPORTANCE</Label>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 5, marginTop: 10 }}>
            {evalResult.data.map(({ feature, importance }) => {
              const pct = Math.abs(importance);
              return (
                <div key={feature} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ width: 120, fontSize: 10, color: 'var(--text-2)', fontFamily: 'var(--font-mono)', textAlign: 'right', flexShrink: 0 }}>{feature}</span>
                  <div style={{ flex: 1, height: 14, background: 'rgba(0,0,0,0.3)', borderRadius: 3, overflow: 'hidden', border: '1px solid var(--border)' }}>
                    <motion.div initial={{ width: 0 }} animate={{ width: `${pct * 100}%` }} transition={{ duration: 0.5, ease: 'easeOut' }}
                      style={{ height: '100%', background: 'linear-gradient(90deg, var(--cyan), var(--magenta))', borderRadius: 3 }} />
                  </div>
                  <span style={{ width: 40, fontSize: 9, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>{(pct * 100).toFixed(1)}%</span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}