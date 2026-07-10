import { motion } from 'framer-motion';
import { card } from '../HUDComponents';

export default function ProgressBar({ info }) {
  return (
    <div style={card}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
        <span style={{ fontFamily: 'var(--font-hud)', fontSize: 9, color: 'var(--purple)' }}>{info.step || 'RUNNING...'}</span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--purple)' }}>{info.progress || 0}%</span>
      </div>
      <div style={{ height: 6, background: 'rgba(170,68,255,0.12)', borderRadius: 3, overflow: 'hidden' }}>
        <motion.div animate={{ width: `${info.progress || 0}%` }} transition={{ duration: 0.4 }} style={{ height: '100%', background: 'linear-gradient(90deg, var(--purple), var(--magenta))', borderRadius: 3 }} />
      </div>
    </div>
  );
}