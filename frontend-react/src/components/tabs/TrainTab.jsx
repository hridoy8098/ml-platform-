import { useState } from 'react';
import { motion } from 'framer-motion';
import useAtlasStore from '../../store/useAtlasStore';
import { trainMLModel, listMLModels } from '../../services/api';
import { Label, Hint, Btn, ErrorBlock, MetricCard, card, select, inputSt } from '../HUDComponents';
import useJobPolling from '../../hooks/useJobPolling';

export default function TrainTab() {
  const { mlDataset, mlTrainingResult, setMLTrainingResult, setMLModels } = useAtlasStore();
  const [task, setTask] = useState('classification');
  const [target, setTarget] = useState('');
  const [algorithm, setAlgorithm] = useState('auto');
  const [testSize, setTestSize] = useState(0.2);
  const [jobId, setJobId] = useState(null);
  const [error, setError] = useState(null);

  const onDone = (result) => {
    setMLTrainingResult(result);
    listMLModels().then(r2 => { if (r2?.success) setMLModels(r2.models); });
  };
  const jobInfo = useJobPolling({ jobId, onDone, onError: (e) => setError(e) });

  const runTrain = async () => {
    if (!mlDataset?.session_id || !target) return;
    setError(null); setMLTrainingResult(null);
    const r = await trainMLModel({ task, session_id: mlDataset.session_id, target, algorithm, test_size: testSize });
    r?.job_id ? setJobId(r.job_id) : setError(r || { message: 'Could not start training' });
  };

  const allCols = mlDataset?.rawColumns || [];
  const isTraining = !!jobId;
  const algoOpts = task === 'auto' ? [] : [
    { value: 'random_forest', label: '🌲 Random Forest' }, { value: 'gradient_boost', label: '🚀 Gradient Boost' },
    { value: task === 'classification' ? 'logistic' : 'linear', label: task === 'classification' ? '📈 Logistic' : '📈 Linear' },
    { value: task === 'classification' ? 'svm' : 'svr', label: '🔲 ' + (task === 'classification' ? 'SVM' : 'SVR') },
    { value: 'knn', label: '📊 KNN' }, { value: 'decision_tree', label: '🌳 Decision Tree' },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {!mlDataset && <Hint center>Upload a dataset first</Hint>}
      {mlDataset && (
        <>
          <div style={card}><Label>TASK TYPE</Label>
            <div style={{ display: 'flex', gap: 6, marginTop: 8 }}>
              {[{ id: 'classification', label: 'Classification', icon: '🏷️' }, { id: 'regression', label: 'Regression', icon: '📉' }, { id: 'auto', label: 'AutoML', icon: '⚡' }].map(t => (
                <motion.button key={t.id} whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.97 }}
                  onClick={() => { setTask(t.id); setAlgorithm('auto'); }}
                  style={{ flex: 1, padding: '9px 12px', borderRadius: 8, cursor: 'pointer', border: `1px solid ${task === t.id ? 'var(--border-glow)' : 'var(--border)'}`, background: task === t.id ? 'var(--cyan-dim)' : 'var(--surface)', color: task === t.id ? 'var(--cyan)' : 'var(--text-3)', fontFamily: 'var(--font)', fontSize: 13, transition: 'all 0.2s', fontWeight: 600 }}>
                  {t.icon} {t.label}
                </motion.button>
              ))}
            </div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            <div style={card}><Label>TARGET COLUMN</Label><select value={target} onChange={e => setTarget(e.target.value)} style={{ ...select, marginTop: 8 }}><option value="">— Select target —</option>{allCols.map(c => <option key={c} value={c}>{c}</option>)}</select>
              <div style={{ fontSize: 10, color: 'var(--text-3)', marginTop: 6, lineHeight: 1.5, opacity: 0.8 }}>Target = "answer" column model tries to predict. Rest = features (input). Model learns feature→target mapping during training.</div>
            </div>
            <div style={card}><Label>TEST SIZE</Label><input type="number" value={testSize} onChange={e => setTestSize(Number(e.target.value))} min={0.05} max={0.5} step={0.05} style={{ ...inputSt, marginTop: 8 }} /></div>
          </div>
          <div style={card}><Label>ALGORITHM</Label><select value={algorithm} onChange={e => setAlgorithm(e.target.value)} style={{ ...select, marginTop: 8 }} disabled={task === 'auto'}><option value="auto">🤖 Auto (Best)</option>{algoOpts.map(a => <option key={a.value} value={a.value}>{a.label}</option>)}</select></div>
          <Btn onClick={runTrain} disabled={isTraining || !target}>{isTraining ? '◌ TRAINING...' : '🤖 TRAIN MODEL'}</Btn>
          {error && <ErrorBlock err={error} />}
          {isTraining && jobInfo && (
            <div style={card}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                <span style={{ fontFamily: 'var(--font-hud)', fontSize: 9, color: 'var(--purple)', letterSpacing: 1 }}>{jobInfo.step || 'TRAINING...'}</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--purple)' }}>{jobInfo.progress || 0}%</span>
              </div>
              <div style={{ height: 6, background: 'rgba(170,68,255,0.12)', borderRadius: 3, overflow: 'hidden' }}>
                <motion.div animate={{ width: `${jobInfo.progress || 0}%` }} transition={{ duration: 0.4, ease: 'easeOut' }}
                  style={{ height: '100%', background: 'linear-gradient(90deg, var(--purple), var(--magenta))', borderRadius: 3, boxShadow: '0 0 10px rgba(170,68,255,0.3)' }} />
              </div>
            </div>
          )}
          {mlTrainingResult?.success && (
            <div style={{ ...card, borderColor: 'rgba(0,240,255,0.35)' }}>
              <Label style={{ color: 'var(--cyan)' }}>✓ TRAINING COMPLETE</Label>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 8, marginTop: 10 }}>
                {mlTrainingResult.algorithm && <MetricCard label="ALGORITHM" value={mlTrainingResult.algorithm.replace(/_/g, ' ').toUpperCase()} color="var(--magenta)" small />}
                {mlTrainingResult.accuracy != null && <MetricCard label="ACCURACY" value={(mlTrainingResult.accuracy * 100).toFixed(1) + '%'} color="var(--cyan)" />}
                {mlTrainingResult.rmse != null && <MetricCard label="RMSE" value={mlTrainingResult.rmse} color="var(--blue)" />}
                {mlTrainingResult.r2 != null && <MetricCard label="R²" value={mlTrainingResult.r2?.toFixed(4)} color="var(--blue)" />}
                {mlTrainingResult.features && <MetricCard label="FEATURES" value={`${mlTrainingResult.features.length} cols`} color="var(--yellow)" small />}
                {mlTrainingResult.cv_score != null && <MetricCard label="CV SCORE" value={(mlTrainingResult.cv_score * 100).toFixed(2) + '%'} color="var(--green)" />}
              </div>
              {mlTrainingResult.model_id && <div style={{ marginTop: 10, fontSize: 9, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>ID: {mlTrainingResult.model_id}</div>}
            </div>
          )}
        </>
      )}
    </div>
  );
}