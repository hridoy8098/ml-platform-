import { useState } from 'react';
import { motion } from 'framer-motion';
import useAtlasStore from '../../store/useAtlasStore';
import { dimReduceML } from '../../services/api';
import { Label, Hint, Btn, ErrorBlock, MetricCard, card, select, inputSt } from '../HUDComponents';
import useJobPolling from '../../hooks/useJobPolling';
import ProgressBar from './ProgressBar';

function DimRedScatter({ points }) {
  if (!points?.length) return null;
  const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
  const xMin = Math.min(...xs), xMax = Math.max(...xs), yMin = Math.min(...ys), yMax = Math.max(...ys);
  const W = 280, H = 160;
  const px = v => ((v - xMin) / (xMax - xMin || 1)) * W;
  const py = v => H - ((v - yMin) / (yMax - yMin || 1)) * H;
  return (
    <svg width={W} height={H} style={{ marginTop: 10, overflow: 'visible' }}>
      {points.map((p, i) => <circle key={i} cx={px(p[0])} cy={py(p[1])} r={2.5} fill="rgba(0,240,255,0.45)" stroke="none" />)}
    </svg>
  );
}

export default function DimRedTab() {
  const { mlDataset } = useAtlasStore();
  const [algorithm, setAlgorithm] = useState('pca');
  const [nComponents, setNComponents] = useState(2);
  const [result, setResult] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [error, setError] = useState(null);

  const onDone = (r) => setResult(r);
  const jobInfo = useJobPolling({ jobId, onDone, onError: (e) => setError(e) });

  const run = async () => {
    if (!mlDataset?.session_id) return;
    setError(null); setResult(null);
    const r = await dimReduceML({ session_id: mlDataset.session_id, algorithm, n_components: nComponents });
    r?.job_id ? setJobId(r.job_id) : setError(r || { message: 'Dim reduction failed' });
  };

  const isRunning = !!jobId;
  const pts = result?.components || [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {!mlDataset && <Hint center>Upload a dataset first</Hint>}
      {mlDataset && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            <div style={card}><Label>ALGORITHM</Label><select value={algorithm} onChange={e => setAlgorithm(e.target.value)} style={{ ...select, marginTop: 8 }}><option value="pca">PCA</option><option value="tsne">t-SNE</option><option value="svd">SVD</option></select></div>
            <div style={card}><Label>N COMPONENTS</Label><input type="number" value={nComponents} onChange={e => setNComponents(Number(e.target.value))} min={2} max={10} style={{ ...inputSt, marginTop: 8 }} /></div>
          </div>
          <Btn onClick={run} disabled={isRunning}>{isRunning ? '◌ REDUCING...' : '📉 RUN DIM REDUCTION'}</Btn>
          {error && <ErrorBlock err={error} />}
          {isRunning && jobInfo && <ProgressBar info={jobInfo} />}
          {result?.success && (
            <>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
                <MetricCard label="ALGORITHM" value={result.algorithm?.toUpperCase()} color="var(--magenta)" small />
                <MetricCard label="COMPONENTS" value={result.n_components} color="var(--cyan)" />
                {result.explained_variance_ratio && <MetricCard label="VAR RATIO" value={result.explained_variance_ratio.map(v => (v * 100).toFixed(1) + '%').join(', ')} color="var(--blue)" small />}
                {result.cumulative_variance != null && <MetricCard label="CUM VAR" value={(result.cumulative_variance * 100).toFixed(1) + '%'} color="var(--cyan)" />}
              </div>
              {pts.length > 0 && nComponents >= 2 && (
                <div style={card}><Label>PC1 vs PC2 — SCATTER</Label><DimRedScatter points={pts} /></div>
              )}
              {result.features && (
                <div style={card}><Label>OUTPUT FEATURES</Label>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
                    {result.features.map(f => <span key={f} style={{ padding: '3px 8px', borderRadius: 4, fontSize: 10, fontFamily: 'var(--font-mono)', background: 'rgba(0,240,255,0.06)', border: '1px solid var(--border)', color: 'var(--cyan)' }}>{f}</span>)}
                  </div>
                </div>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}