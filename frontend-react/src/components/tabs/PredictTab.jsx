import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import useAtlasStore from '../../store/useAtlasStore';
import { predictML, getMLModelInfo } from '../../services/api';
import { Label, Hint, Btn, ErrorBlock, MetricCard, card, select, inputSt, StatusLine } from '../HUDComponents';

export default function PredictTab() {
  const { mlModels, mlPredictionResult, setMLPredictionResult } = useAtlasStore();
  const [selectedModel, setSelectedModel] = useState('');
  const [modelInfo, setModelInfo] = useState(null);
  const [inputValues, setInputValues] = useState({});
  const [loading, setLoading] = useState(false);
  const [infoLoading, setInfoLoading] = useState(false);
  const [error, setError] = useState(null);

  const activeModels = Array.isArray(mlModels) ? mlModels : [];

  useEffect(() => {
    if (!selectedModel) { setModelInfo(null); setInputValues({}); return; }
    setInfoLoading(true); setInputValues({}); setMLPredictionResult(null); setError(null);
    getMLModelInfo(selectedModel).then(res => {
      if (res?.success) { setModelInfo(res); const init = {}; (res.features || []).forEach(f => { init[f] = ''; }); setInputValues(init); }
    }).finally(() => setInfoLoading(false));
  }, [selectedModel]);

  const handlePredict = async () => {
    if (!selectedModel) return;
    setLoading(true); setError(null); setMLPredictionResult(null);
    const r = await predictML(selectedModel, inputValues);
    r?.success ? setMLPredictionResult(r) : setError(r || { message: 'Prediction failed' });
    setLoading(false);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={card}><Label>SELECT MODEL</Label>
        <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)} style={{ ...select, marginTop: 8 }}>
          <option value="">— Choose a trained model —</option>
          {activeModels.map(m => <option key={m.model_id} value={m.model_id}>{m.task ? `[${m.task}] ` : ''}{m.model_id} ({m.size})</option>)}
        </select>
      </div>
      {modelInfo && (
        <div style={{ ...card, display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center' }}>
          <span style={{ padding: '3px 10px', borderRadius: 4, fontSize: 10, fontFamily: 'var(--font-mono)', background: 'rgba(0,240,255,0.08)', border: '1px solid var(--border)', color: 'var(--cyan)' }}>{modelInfo.task}</span>
          <span style={{ fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>Target: <span style={{ color: 'var(--cyan)' }}>{modelInfo.target}</span></span>
        </div>
      )}
      {infoLoading && <StatusLine text="Loading model info..." />}
      {modelInfo && Object.keys(inputValues).length > 0 && (
        <div style={card}><Label>INPUT FEATURES</Label>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: 8, marginTop: 8 }}>
            {Object.entries(inputValues).map(([key, val]) => {
              const dtype = modelInfo.feature_types?.[key] || '';
              return (
                <div key={key}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                    <span style={{ fontSize: 9, color: 'var(--text-2)', fontFamily: 'var(--font-mono)' }}>{key}</span>
                    {dtype && <span style={{ fontSize: 8, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>{dtype}</span>}
                  </div>
                  <input value={val} onChange={e => setInputValues(p => ({ ...p, [key]: e.target.value }))} placeholder={dtype.includes('int') ? '0' : dtype.includes('float') ? '0.0' : '…'} style={inputSt} />
                </div>
              );
            })}
          </div>
        </div>
      )}
      {selectedModel && !infoLoading && <Btn onClick={handlePredict} disabled={loading}>{loading ? '◌ PREDICTING...' : '🔮 PREDICT'}</Btn>}
      {error && <ErrorBlock err={error} />}
      {mlPredictionResult?.success && (
        <div style={{ ...card, borderColor: 'rgba(0,240,255,0.35)' }}>
          <Label style={{ color: 'var(--cyan)' }}>✓ PREDICTION RESULT</Label>
          <div style={{ fontSize: 28, color: 'var(--cyan)', fontFamily: 'var(--font-mono)', marginTop: 8 }}>{String(mlPredictionResult.prediction)}</div>
          {mlPredictionResult.confidence != null && (
            <div style={{ marginTop: 10 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                <span style={{ fontSize: 9, color: 'var(--text-2)', fontFamily: 'var(--font-hud)', letterSpacing: 1 }}>CONFIDENCE</span>
                <span style={{ fontSize: 10, color: 'var(--cyan)', fontFamily: 'var(--font-mono)' }}>{(mlPredictionResult.confidence * 100).toFixed(1)}%</span>
              </div>
              <div style={{ height: 8, background: 'rgba(0,240,255,0.08)', borderRadius: 4, overflow: 'hidden' }}>
                <motion.div initial={{ width: 0 }} animate={{ width: `${(mlPredictionResult.confidence * 100).toFixed(0)}%` }} transition={{ duration: 0.8, ease: 'easeOut' }}
                  style={{ height: '100%', background: 'linear-gradient(90deg, var(--cyan), var(--blue))', borderRadius: 4 }} />
              </div>
            </div>
          )}
          {mlPredictionResult.probabilities && (
            <div style={{ marginTop: 10 }}><Label>CLASS PROBABILITIES</Label>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 6 }}>
                {Object.entries(mlPredictionResult.probabilities).map(([cls, prob]) => (
                  <span key={cls} style={{ padding: '3px 10px', borderRadius: 4, fontSize: 10, fontFamily: 'var(--font-mono)', background: 'rgba(0,240,255,0.06)', border: '1px solid var(--border)', color: 'var(--cyan)' }}>{cls}: {(prob * 100).toFixed(1)}%</span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}