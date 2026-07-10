import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import useAtlasStore from '../../../store/useAtlasStore';
import { listCVModels, predictCVImage, predictCVFolder } from '../../../services/api';
import { Label, Stat, Btn, ErrorBlock, StatusLine, Hint, card, select } from '../../HUDComponents';
import CVImageOverlay from './CVImageOverlay';

export default function CVPredictTab() {
  const { cvSession, cvModels, setCVModels, cvPredictionResult, setCVPredictionResult } = useAtlasStore();
  const [selectedModel, setSelectedModel] = useState('');
  const [selectedImage, setSelectedImage] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState('single');
  const [folderResults, setFolderResults] = useState(null);
  const [folderLoading, setFolderLoading] = useState(false);

  useEffect(() => {
    listCVModels().then(r => { if (r?.success) setCVModels(r.models); });
  }, [setCVModels]);

  const doPredict = async () => {
    if (!selectedModel || !selectedImage) { setError({ message: 'Select a model and image' }); return; }
    setLoading(true); setError(null); setCVPredictionResult(null);
    const res = await predictCVImage(selectedModel, selectedImage, cvSession?.session_id);
    if (res?.success) setCVPredictionResult(res);
    else setError(res || { message: 'Prediction failed' });
    setLoading(false);
  };

  const doPredictFolder = async () => {
    if (!selectedModel || !cvSession?.session_id) { setError({ message: 'Select a model and load a dataset' }); return; }
    setFolderLoading(true); setError(null); setFolderResults(null);
    const res = await predictCVFolder(selectedModel, cvSession.session_id);
    if (res?.success) setFolderResults(res);
    else setError(res || { message: 'Batch prediction failed' });
    setFolderLoading(false);
  };

  const models = Array.isArray(cvModels) ? cvModels : [];
  const previews = cvSession?.preview || [];
  const result = cvPredictionResult;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ display: 'flex', gap: 6 }}>
        <Btn onClick={() => setMode('single')} style={{
          flex: 1, background: mode === 'single' ? 'rgba(0,240,255,0.15)' : 'transparent',
          borderColor: mode === 'single' ? 'rgba(0,240,255,0.3)' : 'var(--border)',
        }}>SINGLE PREDICT</Btn>
        <Btn onClick={() => setMode('folder')} style={{
          flex: 1, background: mode === 'folder' ? 'rgba(0,240,255,0.15)' : 'transparent',
          borderColor: mode === 'folder' ? 'rgba(0,240,255,0.3)' : 'var(--border)',
        }}>BATCH PREDICT</Btn>
      </div>
      <div style={card}>
        <Label>CV PREDICTION ({mode === 'single' ? 'SINGLE' : 'BATCH'})</Label>
        <div style={{ display: 'grid', gap: 8, marginTop: 10 }}>
          <div>
            <div style={{ fontSize: 9, color: 'var(--text-3)', fontFamily: 'var(--font-hud)', letterSpacing: 1, marginBottom: 4 }}>MODEL</div>
            <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)} style={select}>
              <option value="">— Select CV Model —</option>
              {models.map(m => (
                <option key={m.model_id} value={m.model_id}>{m.model_id} ({m.task_type})</option>
              ))}
            </select>
          </div>
          {mode === 'single' && previews.length > 0 && (
            <div>
              <div style={{ fontSize: 9, color: 'var(--text-3)', fontFamily: 'var(--font-hud)', letterSpacing: 1, marginBottom: 4 }}>IMAGE</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(72px, 1fr))', gap: 4, maxHeight: 180, overflowY: 'auto' }}>
                {previews.map((p, i) => (
                  <div key={p.image_id || i} onClick={() => setSelectedImage(p.filename)}
                    style={{
                      padding: 4, borderRadius: 4, cursor: 'pointer', textAlign: 'center',
                      background: selectedImage === p.filename ? 'rgba(0,240,255,0.15)' : 'rgba(5,15,25,0.6)',
                      border: `1px solid ${selectedImage === p.filename ? 'rgba(0,240,255,0.3)' : 'rgba(0,240,255,0.06)'}`,
                    }}>
                    <div style={{ fontSize: 14 }}>{p.format === 'dicom' ? '🩻' : '🖼️'}</div>
                    <div style={{ fontSize: 6, fontFamily: 'var(--font-mono)', color: 'var(--text-3)', marginTop: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.filename}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
          {mode === 'single' && (
            <Btn onClick={doPredict} disabled={loading || !selectedModel || !selectedImage}>
              {loading ? '◌ PREDICTING...' : '🔮 PREDICT'}
            </Btn>
          )}
          {mode === 'folder' && (
            <Btn onClick={doPredictFolder} disabled={folderLoading || !selectedModel || !cvSession?.session_id}>
              {folderLoading ? '◌ BATCH PREDICTING...' : '📊 PREDICT ALL'}
            </Btn>
          )}
        </div>
      </div>
      {loading && <StatusLine text="Running inference..." />}
      {folderLoading && <StatusLine text="Running batch inference..." />}
      {error && <ErrorBlock err={error} />}

      {result?.success && (
        <>
          <div style={card}>
            <Label>RESULT</Label>
            <div style={{ display: 'flex', gap: 14, marginTop: 8, flexWrap: 'wrap' }}>
              <Stat label="PREDICTION" value={result.prediction?.toUpperCase()} color={
                result.prediction === 'positive' || result.prediction === 'anomaly_detected' ? 'var(--red)' : 'var(--green)'
              } />
              <Stat label="CONFIDENCE" value={(result.confidence * 100).toFixed(1) + '%'} />
              {result.anomaly_percentage != null && (
                <Stat label="ANOMALY AREA" value={result.anomaly_percentage + '%'} color="var(--red)" />
              )}
              {result.regions && <Stat label="REGIONS" value={result.regions.length} />}
            </div>
            {result.findings?.length > 0 && (
              <div style={{ marginTop: 10, padding: 10, background: 'rgba(0,0,0,0.2)', borderRadius: 6 }}>
                <div style={{ fontSize: 9, color: 'var(--text-3)', fontFamily: 'var(--font-hud)', letterSpacing: 1, marginBottom: 6 }}>FINDINGS</div>
                {result.findings.map((f, i) => (
                  <div key={i} style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-2)', marginBottom: 4, lineHeight: 1.4 }}>{f}</div>
                ))}
              </div>
            )}
          </div>
          <CVImageOverlay result={result} />
        </>
      )}

      {folderResults?.success && (
        <div style={card}>
          <Label>BATCH RESULTS</Label>
          <div style={{ display: 'flex', gap: 14, marginTop: 8 }}>
            <Stat label="TOTAL" value={folderResults.total} />
            <Stat label="POSITIVE" value={folderResults.positive_count} color="var(--red)" />
            <Stat label="NEGATIVE" value={folderResults.negative_count} color="var(--green)" />
          </div>
          <div style={{ marginTop: 10, maxHeight: 300, overflowY: 'auto' }}>
            {folderResults.results?.map((r, i) => (
              <div key={i} style={{
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                padding: '4px 0', borderBottom: '1px solid rgba(0,240,255,0.04)',
                fontSize: 10, fontFamily: 'var(--font-mono)',
              }}>
                <span style={{ color: 'var(--text-2)' }}>{r.filename}</span>
                <span style={{
                  color: r.prediction === 'positive' ? 'var(--red)' : 'var(--green)',
                }}>
                  {r.prediction.toUpperCase()} ({(r.confidence * 100).toFixed(0)}%)
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
