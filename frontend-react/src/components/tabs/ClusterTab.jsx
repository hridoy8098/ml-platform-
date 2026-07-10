import { useState } from 'react';
import useAtlasStore from '../../store/useAtlasStore';
import { clusterMLData } from '../../services/api';
import { Label, Hint, Btn, ErrorBlock, MetricCard, card, select, inputSt } from '../HUDComponents';
import useJobPolling from '../../hooks/useJobPolling';
import ProgressBar from './ProgressBar';

export default function ClusterTab() {
  const { mlDataset } = useAtlasStore();
  const [algorithm, setAlgorithm] = useState('kmeans');
  const [nClusters, setNClusters] = useState(3);
  const [eps, setEps] = useState(0.5);
  const [minSamples, setMinSamples] = useState(5);
  const [result, setResult] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [error, setError] = useState(null);

  const onDone = (r) => setResult(r);
  const jobInfo = useJobPolling({ jobId, onDone, onError: (e) => setError(e) });

  const run = async () => {
    if (!mlDataset?.session_id) return;
    setError(null); setResult(null);
    const r = await clusterMLData({ session_id: mlDataset.session_id, algorithm, n_clusters: nClusters, eps, min_samples: minSamples });
    r?.job_id ? setJobId(r.job_id) : setError(r || { message: 'Clustering failed' });
  };

  const isRunning = !!jobId;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {!mlDataset && <Hint center>Upload a dataset first</Hint>}
      {mlDataset && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            <div style={card}><Label>ALGORITHM</Label><select value={algorithm} onChange={e => setAlgorithm(e.target.value)} style={{ ...select, marginTop: 8 }}><option value="kmeans">K-Means</option><option value="dbscan">DBSCAN</option><option value="agglomerative">Agglomerative</option></select></div>
            <div style={card}><Label>N CLUSTERS</Label><input type="number" value={nClusters} onChange={e => setNClusters(Number(e.target.value))} min={2} max={50} style={{ ...inputSt, marginTop: 8 }} /></div>
          </div>
          {algorithm === 'dbscan' && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              <div style={card}><Label>EPS</Label><input type="number" value={eps} onChange={e => setEps(Number(e.target.value))} step={0.1} style={{ ...inputSt, marginTop: 8 }} /></div>
              <div style={card}><Label>MIN SAMPLES</Label><input type="number" value={minSamples} onChange={e => setMinSamples(Number(e.target.value))} min={1} style={{ ...inputSt, marginTop: 8 }} /></div>
            </div>
          )}
          <Btn onClick={run} disabled={isRunning}>{isRunning ? '◌ CLUSTERING...' : '🔘 RUN CLUSTERING'}</Btn>
          {error && <ErrorBlock err={error} />}
          {isRunning && jobInfo && <ProgressBar info={jobInfo} />}
          {result?.success && (
            <div style={{ ...card, borderColor: 'rgba(0,240,255,0.35)' }}>
              <Label style={{ color: 'var(--cyan)' }}>✓ CLUSTERING COMPLETE</Label>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginTop: 10 }}>
                <MetricCard label="CLUSTERS" value={result.n_clusters || result.algorithm} color="var(--cyan)" />
                {result.inertia != null && <MetricCard label="INERTIA" value={result.inertia.toFixed(2)} color="var(--blue)" small />}
                {result.n_noise != null && <MetricCard label="NOISE" value={result.n_noise} color="var(--red)" small />}
              </div>
              {result.cluster_centers && (
                <div style={{ marginTop: 10 }}><Label>CENTERS</Label>
                  {result.cluster_centers.map((c, i) => (
                    <div key={i} style={{ fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', marginTop: 4 }}>Cluster {c.cluster} ({c.size} pts): [{c.center.map(v => v.toFixed(3)).join(', ')}]</div>
                  ))}
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}