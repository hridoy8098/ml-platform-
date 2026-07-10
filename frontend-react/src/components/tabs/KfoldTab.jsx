import { useState } from 'react';
import { motion } from 'framer-motion';
import useAtlasStore from '../../store/useAtlasStore';
import { kfoldML } from '../../services/api';
import { Label, Hint, Btn, ErrorBlock, MetricCard, card, select, inputSt } from '../HUDComponents';
import useJobPolling from '../../hooks/useJobPolling';
import ProgressBar from './ProgressBar';

export default function KfoldTab() {
  const { mlDataset } = useAtlasStore();
  const [target, setTarget] = useState('');
  const [taskType, setTaskType] = useState('classification');
  const [algorithm, setAlgorithm] = useState('random_forest');
  const [nSplits, setNSplits] = useState(5);
  const [result, setResult] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [error, setError] = useState(null);

  const onDone = (r) => setResult(r);
  const jobInfo = useJobPolling({ jobId, onDone, onError: (e) => setError(e) });

  const run = async () => {
    if (!mlDataset?.session_id || !target) return;
    setError(null); setResult(null);
    const r = await kfoldML({ session_id: mlDataset.session_id, target, task_type: taskType, algorithm, n_splits: nSplits });
    r?.job_id ? setJobId(r.job_id) : setError(r || { message: 'K-Fold failed' });
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
            <div style={card}><Label>N SPLITS</Label><input type="number" value={nSplits} onChange={e => setNSplits(Number(e.target.value))} min={2} max={20} style={{ ...inputSt, marginTop: 8 }} /></div>
          </div>
          <Btn onClick={run} disabled={isRunning || !target}>{isRunning ? '◌ CROSS VALIDATING...' : '♻️ RUN K-FOLD CV'}</Btn>
          {error && <ErrorBlock err={error} />}
          {isRunning && jobInfo && <ProgressBar info={jobInfo} />}
          {result?.success && (
            <div style={{ ...card, borderColor: 'rgba(0,240,255,0.35)' }}>
              <Label style={{ color: 'var(--cyan)' }}>✓ K-FOLD CV COMPLETE</Label>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginTop: 10 }}>
                <MetricCard label="MEAN" value={(result.mean_score * 100).toFixed(2) + '%'} color="var(--cyan)" />
                <MetricCard label="STD" value={result.std_score ? (result.std_score * 100).toFixed(2) + '%' : '—'} color="var(--yellow)" small />
                <MetricCard label="FOLDS" value={result.n_splits} color="var(--magenta)" small />
              </div>
              {result.scores && (
                <div style={{ marginTop: 10 }}><Label>PER-FOLD SCORES</Label>
                  <div style={{ display: 'flex', alignItems: 'flex-end', gap: 6, height: 60, marginTop: 10 }}>
                    {result.scores.map((s, i) => (
                      <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flex: 1 }}>
                        <motion.div initial={{ height: 0 }} animate={{ height: `${s * 100}px` }} transition={{ duration: 0.4, delay: i * 0.05 }}
                          style={{ width: '100%', maxWidth: 24, background: 'rgba(0,240,255,0.25)', borderRadius: '3px 3px 0 0', border: '1px solid rgba(0,240,255,0.2)' }} />
                        <span style={{ fontSize: 7, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', marginTop: 3 }}>Fold {i + 1}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}