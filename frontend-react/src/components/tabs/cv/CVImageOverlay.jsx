import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Label, Btn, card } from '../../HUDComponents';

export default function CVImageOverlay({ result }) {
  const [viewMode, setViewMode] = useState('overlay');
  if (!result?.overlay_image) return null;
  const base64Url = `data:image/png;base64,${result.overlay_image}`;
  const maskUrl = result.mask_image ? `data:image/png;base64,${result.mask_image}` : null;
  const rawUrl = result.raw_overlay_image ? `data:image/png;base64,${result.raw_overlay_image}` : null;

  const src = viewMode === 'overlay' ? base64Url : viewMode === 'mask' ? maskUrl : rawUrl;

  return (
    <div style={card}>
      <Label>PREDICTION VISUALIZATION</Label>
      <div style={{ display: 'flex', gap: 6, marginTop: 8, flexWrap: 'wrap' }}>
        {result.overlay_image && (
          <Btn onClick={() => setViewMode('overlay')} style={{
            background: viewMode === 'overlay' ? 'rgba(0,240,255,0.15)' : 'transparent',
            borderColor: viewMode === 'overlay' ? 'rgba(0,240,255,0.3)' : 'var(--border)',
          }}>BLENDED OVERLAY</Btn>
        )}
        {result.mask_image && (
          <Btn onClick={() => setViewMode('mask')} style={{
            background: viewMode === 'mask' ? 'rgba(0,240,255,0.15)' : 'transparent',
            borderColor: viewMode === 'mask' ? 'rgba(0,240,255,0.3)' : 'var(--border)',
          }}>MASK ONLY</Btn>
        )}
        {result.raw_overlay_image && (
          <Btn onClick={() => setViewMode('raw')} style={{
            background: viewMode === 'raw' ? 'rgba(0,240,255,0.15)' : 'transparent',
            borderColor: viewMode === 'raw' ? 'rgba(0,240,255,0.3)' : 'var(--border)',
          }}>RAW OVERLAY</Btn>
        )}
      </div>
      <AnimatePresence mode="wait">
        <motion.div key={viewMode} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
          style={{ marginTop: 10, display: 'flex', justifyContent: 'center' }}>
          <img src={src} alt={`View: ${viewMode}`} style={{
            maxWidth: '100%', maxHeight: 400, borderRadius: 8,
            border: '1px solid var(--border)', objectFit: 'contain',
          }} />
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
