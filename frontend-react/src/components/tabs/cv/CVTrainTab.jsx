import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import useAtlasStore from '../../../store/useAtlasStore';
import { trainCVModel, getMLJobStatus, listCVModels } from '../../../services/api';
import { Label, Stat, Btn, ErrorBlock, StatusLine, Hint, card, inputSt, select } from '../../HUDComponents';

export default function CVTrainTab() {
  const { cvSession, setCVModels, cvTrainingResult, setCVTrainingResult } = useAtlasStore();
  const [taskType, setTaskType] = useState('classification');
  const [epochs, setEpochs] = useState(10);
  const [batchSize, setBatchSize] = useState(16);
  const [learningRate, setLearningRate] = useState(0.001);
  const [jobId, setJobId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(null);

  useEffect(() => {
    let timer;
    if (jobId) {
      timer = setInterval(async () => {
        const res = await getMLJobStatus(jobId);
        if (!res?.success) { clearInterval(timer); return; }
        setProgress(res);
        if (res.status === 'done' || res.status === 'error') {
          clearInterval(timer);
          if (res.status === 'done' && res.result) {
            setCVTrainingResult(res.result);
          }
          setLoading(false);
          if (res.status === 'done') {
            const models = await listCVModels();
            if (models?.success) setCVModels(models.models);
          }
        }
      }, 1500);
    }
    return () => { if (timer) clearInterval(timer); };
  }, [jobId, setCVTrainingResult, setCVModels]);

  const startTrain = async () => {
    if (!cvSession?.session_id) { setError({ message: 'No CV dataset loaded' }); return; }
    setLoading(true); setError(null); setCVTrainingResult(null); setJobId(null); setProgress(null);
    const res = await trainCVModel({
      session_id: cvSession.session_id,
      task_type: taskType,
      epochs,
      batch_size: batchSize,
      learning_rate: learningRate,
    });
    if (res?.success) {
      setJobId(res.job_id);
    } else {
      setError(res || { message: 'Training failed to start' });
      setLoading(false);
    }
  };

  const hasData = !!cvSession?.session_id;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {!hasData && <Hint center>Upload a CV dataset (zip) first in CV DATA tab.</Hint>}
      <div style={card}>
        <Label>CV TRAINING</Label>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginTop: 10 }}>
          <div>
            <div style={{ fontSize: 9, color: 'var(--text-3)', fontFamily: 'var(--font-hud)', letterSpacing: 1, marginBottom: 4 }}>TASK TYPE</div>
            <select value={taskType} onChange={e => setTaskType(e.target.value)} style={select}>
              <option value="classification">Classification</option>
              <option value="segmentation">Segmentation (U-Net)</option>
            </select>
          </div>
          <div>
            <div style={{ fontSize: 9, color: 'var(--text-3)', fontFamily: 'var(--font-hud)', letterSpacing: 1, marginBottom: 4 }}>EPOCHS</div>
            <input type="number" value={epochs} min={1} max={500}
              onChange={e => setEpochs(parseInt(e.target.value) || 10)} style={inputSt} />
          </div>
          <div>
            <div style={{ fontSize: 9, color: 'var(--text-3)', fontFamily: 'var(--font-hud)', letterSpacing: 1, marginBottom: 4 }}>BATCH SIZE</div>
            <input type="number" value={batchSize} min={1} max={512}
              onChange={e => setBatchSize(parseInt(e.target.value) || 16)} style={inputSt} />
          </div>
          <div>
            <div style={{ fontSize: 9, color: 'var(--text-3)', fontFamily: 'var(--font-hud)', letterSpacing: 1, marginBottom: 4 }}>LEARNING RATE</div>
            <input type="number" value={learningRate} min={0.00001} max={1} step={0.0001}
              onChange={e => setLearningRate(parseFloat(e.target.value) || 0.001)} style={inputSt} />
          </div>
        </div>
        <Btn onClick={startTrain} disabled={loading || !hasData} style={{ marginTop: 10 }}>
          {loading ? '◌ TRAINING...' : '🚀 START TRAINING'}
        </Btn>
      </div>
      {loading && progress && (
        <div style={card}>
          <Label>PROGRESS</Label>
          <div style={{ marginTop: 8 }}>
            <div style={{
              height: 6, background: 'rgba(0,240,255,0.08)', borderRadius: 3,
              overflow: 'hidden',
            }}>
              <motion.div style={{
                height: '100%', background: 'var(--cyan)', borderRadius: 3,
                width: `${progress.progress || 0}%`,
              }} />
            </div>
            <div style={{
              display: 'flex', justifyContent: 'space-between', marginTop: 4,
              fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-3)',
            }}>
              <span>{progress.step || 'Processing...'}</span>
              <span>{progress.progress || 0}%</span>
            </div>
          </div>
        </div>
      )}
      {cvTrainingResult?.success && (
        <div style={card}>
          <Label>TRAINING COMPLETE</Label>
          <div style={{ display: 'flex', gap: 14, marginTop: 8, flexWrap: 'wrap' }}>
            {cvTrainingResult.val_accuracy != null && <Stat label="VAL ACCURACY" value={(cvTrainingResult.val_accuracy * 100).toFixed(1) + '%'} />}
            {cvTrainingResult.val_loss != null && <Stat label="VAL LOSS" value={cvTrainingResult.val_loss.toFixed(4)} color="var(--text-2)" />}
            {cvTrainingResult.model_id && <Stat label="MODEL ID" value={cvTrainingResult.model_id} />}
          </div>
        </div>
      )}
      {error && <ErrorBlock err={error} />}
    </div>
  );
}
