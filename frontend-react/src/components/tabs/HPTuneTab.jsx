import { useState } from 'react';
import useAtlasStore from '../../store/useAtlasStore';
import { tuneMLHyperparams } from '../../services/api';
import { Label, Hint, Btn, ErrorBlock, MetricCard, card, select, inputSt } from '../HUDComponents';
import useJobPolling from '../../hooks/useJobPolling';
import ProgressBar from './ProgressBar';

export default function HPTuneTab() {
  const { mlDataset } = useAtlasStore();
  const [target, setTarget] = useState('');
  const [taskType, setTaskType] = useState('classification');
  const [algorithm, setAlgorithm] = useState('random_forest');
  const [cv, setCv] = useState(5);
  const [result, setResult] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [error, setError] = useState(null);

  const onDone = (r) => setResult(r);
  const jobInfo = useJobPolling({ jobId, onDone, onError: (e) => setError(e) });

  const run = async () => {
    if (!mlDataset?.session_id || !target) return;
    setError(null); setResult(null);
    const r = await tuneMLHyperparams({ session_id: mlDataset.session_id, target, task_type: taskType, algorithm, cv, scoring: 'accuracy' });
    r?.job_id ? setJobId(r.job_id) : setError(r || { message: 'Tuning failed' });
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
            <div style={card}><Label>ALGORITHM</Label><select value={algorithm} onChange={e => setAlgorithm(e.target.value)} style={{ ...select, marginTop: 8 }}><option value="random_forest">Random Forest</option><option value="svm">SVM</option><option value="logistic">Logistic Regression</option><option value="gradient_boost">Gradient Boost</option></select></div>
            <div style={card}><Label>CV FOLDS</Label><input type="number" value={cv} onChange={e => setCv(Number(e.target.value))} min={2} max={20} style={{ ...inputSt, marginTop: 8 }} /></div>
          </div>
          <Btn onClick={run} disabled={isRunning || !target}>{isRunning ? '◌ TUNING...' : '⚙️ START HYPERPARAMETER TUNING'}</Btn>
          {error && <ErrorBlock err={error} />}
          {isRunning && jobInfo && <ProgressBar info={jobInfo} />}
          {result?.success && (
            <div style={{ ...card, borderColor: 'rgba(0,240,255,0.35)' }}>
              <Label style={{ color: 'var(--cyan)' }}>✓ TUNING COMPLETE</Label>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginTop: 10 }}>
                {result.best_score != null && <MetricCard label="BEST SCORE" value={(result.best_score * 100).toFixed(2) + '%'} color="var(--cyan)" />}
                {result.best_params && <div style={{ fontSize: 10, color: 'var(--text-2)', fontFamily: 'var(--font-mono)' }}>Params: {JSON.stringify(result.best_params)}</div>}
                {result.cv && <MetricCard label="CV" value={result.cv} color="var(--yellow)" small />}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}