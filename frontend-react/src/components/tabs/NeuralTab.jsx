import { useState } from 'react';
import useAtlasStore from '../../store/useAtlasStore';
import { trainMLNeural } from '../../services/api';
import { Label, Hint, Btn, ErrorBlock, MetricCard, card, select, inputSt } from '../HUDComponents';
import useJobPolling from '../../hooks/useJobPolling';
import ProgressBar from './ProgressBar';

export default function NeuralTab() {
  const { mlDataset } = useAtlasStore();
  const [target, setTarget] = useState('');
  const [taskType, setTaskType] = useState('classification');
  const [hiddenLayers, setHiddenLayers] = useState('100,50');
  const [maxIter, setMaxIter] = useState(500);
  const [result, setResult] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [error, setError] = useState(null);

  const onDone = (r) => setResult(r);
  const jobInfo = useJobPolling({ jobId, onDone, onError: (e) => setError(e) });

  const run = async () => {
    if (!mlDataset?.session_id || !target) return;
    setError(null); setResult(null);
    const r = await trainMLNeural({ session_id: mlDataset.session_id, target, task_type: taskType, hidden_layers: hiddenLayers, max_iter: maxIter });
    r?.job_id ? setJobId(r.job_id) : setError(r || { message: 'Neural training failed' });
  };

  const allCols = mlDataset?.rawColumns || [];
  const isRunning = !!jobId;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {!mlDataset && <Hint center>Upload a dataset first</Hint>}
      {mlDataset && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            <div style={card}><Label>TARGET COLUMN</Label><select value={target} onChange={e => setTarget(e.target.value)} style={{ ...select, marginTop: 8 }}><option value="">— Select target —</option>{allCols.map(c => <option key={c} value={c}>{c}</option>)}</select></div>
            <div style={card}><Label>TASK TYPE</Label><select value={taskType} onChange={e => setTaskType(e.target.value)} style={{ ...select, marginTop: 8 }}><option value="classification">Classification</option><option value="regression">Regression</option></select></div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            <div style={card}><Label>HIDDEN LAYERS</Label><input value={hiddenLayers} onChange={e => setHiddenLayers(e.target.value)} placeholder="100,50" style={{ ...inputSt, marginTop: 8 }} /><div style={{ fontSize: 8, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', marginTop: 4 }}>comma-separated</div></div>
            <div style={card}><Label>MAX ITERATIONS</Label><input type="number" value={maxIter} onChange={e => setMaxIter(Number(e.target.value))} min={100} max={5000} step={100} style={{ ...inputSt, marginTop: 8 }} /></div>
          </div>
          <Btn onClick={run} disabled={isRunning || !target}>{isRunning ? '◌ TRAINING MLP...' : '🧠 TRAIN NEURAL NETWORK'}</Btn>
          {error && <ErrorBlock err={error} />}
          {isRunning && jobInfo && <ProgressBar info={jobInfo} />}
          {result?.success && (
            <div style={{ ...card, borderColor: 'rgba(0,240,255,0.35)' }}>
              <Label style={{ color: 'var(--cyan)' }}>✓ MLP TRAINING COMPLETE</Label>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginTop: 10 }}>
                {result.accuracy != null && <MetricCard label="ACCURACY" value={(result.accuracy * 100).toFixed(1) + '%'} color="var(--cyan)" />}
                {result.rmse != null && <MetricCard label="RMSE" value={result.rmse} color="var(--blue)" />}
                {result.score != null && result.accuracy == null && <MetricCard label="SCORE" value={result.score} color="var(--cyan)" />}
                {result.loss != null && <MetricCard label="LOSS" value={result.loss} color="var(--magenta)" small />}
              </div>
              {result.model_id && <div style={{ marginTop: 10, fontSize: 9, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>ID: {result.model_id}</div>}
            </div>
          )}
        </>
      )}
    </div>
  );
}