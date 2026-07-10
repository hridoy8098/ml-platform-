import { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import useAtlasStore from '../../../store/useAtlasStore';
import { uploadCVZip, getCVSessionInfo } from '../../../services/api';
import { Label, Stat, Btn, ErrorBlock, StatusLine, Hint, card } from '../../HUDComponents';

export default function CVDataTab() {
  const { cvSession, setCVSession, setCVModels } = useAtlasStore();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const onDrop = useCallback(async (accepted) => {
    const file = accepted[0];
    if (!file) return;
    setLoading(true); setError(null);
    const res = await uploadCVZip(file);
    if (res?.success) {
      setCVSession({ ...res, fileName: file.name });
      if (res.session_id) {
        const info = await getCVSessionInfo(res.session_id);
        if (info?.success) setCVSession({ ...res, ...info, fileName: file.name });
      }
    } else setError(res || { message: 'Upload failed', suggestion: 'Only .zip files with images (jpg/png/tiff/bmp/dcm) are accepted.' });
    setLoading(false);
  }, [setCVSession]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, maxFiles: 1,
    accept: { 'application/zip': ['.zip'] },
  });

  const previews = cvSession?.preview || [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div {...getRootProps()} style={{
        border: `1px dashed ${isDragActive ? 'var(--cyan)' : 'var(--border)'}`,
        borderRadius: 10, padding: 20, textAlign: 'center', cursor: 'pointer',
        background: isDragActive ? 'rgba(0,240,255,0.06)' : 'var(--surface)',
        transition: 'all 0.2s', backdropFilter: 'blur(8px)',
      }}>
        <input {...getInputProps()} />
        <div style={{ fontSize: 28, marginBottom: 4 }}>🗜️</div>
        <div style={{ fontFamily: 'var(--font-hud)', fontSize: 10, color: isDragActive ? 'var(--cyan)' : 'var(--text-3)', letterSpacing: 1 }}>
          {isDragActive ? 'DROP ZIP HERE' : 'DROP IMAGE ZIP (JPG/PNG/TIFF/BMP/DCM)'}
        </div>
        <div style={{ fontSize: 10, color: 'var(--text-3)', marginTop: 4, opacity: 0.6 }}>or click to browse</div>
        <div style={{ fontSize: 9, color: 'var(--text-3)', marginTop: 6, opacity: 0.4 }}>Max 500MB — supports DICOM medical images</div>
      </div>
      {loading && <StatusLine text="Extracting and processing images..." />}
      {error && <ErrorBlock err={error} />}
      {cvSession?.total_images > 0 && (
        <div style={card}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
            <div><Label>CV DATASET</Label><span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--cyan)', marginLeft: 8 }}>{cvSession.fileName}</span></div>
            <div style={{ display: 'flex', gap: 14 }}>
              <Stat label="IMAGES" value={cvSession.total_images} />
              <Stat label="SHAPE" value={cvSession.image_shape ? `${cvSession.image_shape[0]}×${cvSession.image_shape[1]}` : '—'} />
            </div>
          </div>
          {previews.length > 0 && (
            <>
              <Label>PREVIEW ({previews.length} of {cvSession.total_images})</Label>
              <div style={{
                display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(100px, 1fr))',
                gap: 6, marginTop: 8, maxHeight: 300, overflowY: 'auto',
              }}>
                {previews.map((p, i) => (
                  <div key={p.image_id || i} style={{
                    background: 'rgba(5,15,25,0.6)', borderRadius: 6, padding: 4,
                    textAlign: 'center', border: '1px solid rgba(0,240,255,0.08)',
                  }}>
                    <div style={{
                      width: '100%', aspectRatio: '1', borderRadius: 4,
                      background: 'rgba(0,240,255,0.04)',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      fontSize: 20, color: 'var(--text-3)',
                    }}>
                      {p.format === 'dicom' ? '🩻' : '🖼️'}
                    </div>
                    <div style={{
                      fontSize: 7, fontFamily: 'var(--font-mono)', color: 'var(--text-3)',
                      marginTop: 2, overflow: 'hidden', textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}>{p.filename}</div>
                    <div style={{ fontSize: 7, color: 'var(--text-3)', opacity: 0.5 }}>
                      {p.shape ? `${p.shape[0]}×${p.shape[1]}` : ''}
                    </div>
                  </div>
                ))}
              </div>
              <Hint center>{cvSession.total_images} image(s) loaded and preprocessed. Switch to CV TRAIN to train a model.</Hint>
            </>
          )}
        </div>
      )}
    </div>
  );
}
