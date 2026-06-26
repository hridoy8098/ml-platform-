import { motion } from 'framer-motion';

export const card = {
  background: 'rgba(5, 15, 25, 0.55)', border: '1px solid var(--border)',
  borderRadius: 10, padding: 14, backdropFilter: 'blur(8px)',
  position: 'relative', overflow: 'hidden',
};

export const btnSt = {
  padding: '9px 18px', borderRadius: 6, border: '1px solid var(--border)',
  background: 'var(--cyan-dim)', color: 'var(--cyan)',
  fontFamily: 'var(--font)', fontSize: 12, letterSpacing: 1, transition: 'all 0.2s',
  cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 6,
};

export const select = {
  width: '100%', padding: '8px 10px', borderRadius: 6,
  border: '1px solid var(--border)', background: 'rgba(5,5,15,0.6)',
  color: 'var(--text-1)', fontFamily: 'var(--font-mono)', fontSize: 11,
  outline: 'none', cursor: 'pointer',
};

export const inputSt = {
  width: '100%', padding: '8px 10px', borderRadius: 6, boxSizing: 'border-box',
  border: '1px solid var(--border)', background: 'rgba(5,5,15,0.6)',
  color: 'var(--text-1)', fontFamily: 'var(--font-mono)', fontSize: 11,
  outline: 'none',
};

export const tableStyles = {
  table:      { style: { background: 'transparent', color: 'var(--cyan)' } },
  headRow:    { style: { background: 'rgba(0,240,255,0.04)', borderBottom: '1px solid var(--border)', minHeight: 34 } },
  headCells:  { style: { color: 'var(--cyan)', fontSize: 10, fontFamily: 'var(--font-hud)', letterSpacing: 1, paddingLeft: 10 } },
  rows:       { style: { background: 'transparent', minHeight: 30, borderBottom: '1px solid rgba(0,240,255,0.04)', '&:hover': { background: 'rgba(0,240,255,0.03)' } } },
  cells:      { style: { color: 'var(--text-1)', fontSize: 11, fontFamily: 'var(--font-mono)', paddingLeft: 10 } },
  pagination: { style: { background: 'rgba(5,5,15,0.6)', borderTop: '1px solid var(--border)', color: 'var(--cyan)', minHeight: 38 } },
  noData:     { style: { color: 'var(--text-3)', fontFamily: 'var(--font)', fontSize: 13 } },
};

export function Label({ children, style: extra }) {
  return (
    <div style={{
      fontFamily: 'var(--font-hud)', fontSize: 9, letterSpacing: 2,
      color: 'rgba(0,240,255,0.45)', ...extra,
    }}>
      {children}
    </div>
  );
}

export function Stat({ label, value, color = 'var(--cyan)' }) {
  return (
    <div style={{ textAlign: 'right' }}>
      <div style={{ fontSize: 8, color: 'var(--text-3)', fontFamily: 'var(--font-hud)', letterSpacing: 1 }}>{label}</div>
      <div style={{ fontSize: 12, color, fontFamily: 'var(--font-mono)' }}>{value ?? '—'}</div>
    </div>
  );
}

export function Hint({ children, center }) {
  return (
    <div style={{
      color: 'var(--text-3)', fontSize: 11, fontFamily: 'var(--font-mono)',
      textAlign: center ? 'center' : undefined, padding: center ? 16 : undefined,
      opacity: 0.7,
    }}>
      {children}
    </div>
  );
}

export function StatusLine({ text }) {
  return (
    <div style={{
      textAlign: 'center', color: 'var(--cyan)', fontSize: 11,
      fontFamily: 'var(--font-mono)', padding: '8px 0',
    }}>
      <motion.span animate={{ opacity: [1, 0.4, 1] }} transition={{ duration: 1.5, repeat: Infinity }}>
        ◌
      </motion.span>{' '}
      {text}
    </div>
  );
}

export function Btn({ children, onClick, disabled, style: extra }) {
  return (
    <motion.button
      whileHover={disabled ? {} : { scale: 1.02 }}
      whileTap={disabled ? {} : { scale: 0.95 }}
      onClick={onClick} disabled={disabled}
      style={{
        ...btnSt, ...extra,
        opacity: disabled ? 0.4 : 1,
        cursor: disabled ? 'not-allowed' : 'pointer',
        alignSelf: 'flex-start',
      }}>
      {children}
    </motion.button>
  );
}

export function ErrorBlock({ err }) {
  if (!err) return null;
  const msg = err.message || String(err);
  const hint = err.suggestion;
  return (
    <div style={{
      background: 'rgba(255,51,85,0.06)', border: '1px solid rgba(255,51,85,0.25)',
      borderRadius: 8, padding: '10px 14px',
    }}>
      <div style={{ color: 'var(--red)', fontSize: 11, fontFamily: 'var(--font-mono)' }}>
        ❌ {msg}
      </div>
      {hint && <div style={{ color: 'rgba(255,170,100,0.8)', fontSize: 10, fontFamily: 'var(--font-mono)', marginTop: 4 }}>💡 {hint}</div>}
    </div>
  );
}

export function MetricCard({ label, value, color, small }) {
  return (
    <div style={{
      background: 'rgba(5,15,25,0.5)', border: `1px solid ${color || 'var(--border)'}22`,
      borderRadius: 8, padding: 10, textAlign: 'center',
    }}>
      <div style={{ fontFamily: 'var(--font-hud)', fontSize: 8, letterSpacing: 1, color: 'var(--text-3)', marginBottom: 4 }}>{label}</div>
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: small ? 11 : 16, color: color || 'var(--cyan)', wordBreak: 'break-all' }}>{value ?? '—'}</div>
    </div>
  );
}

export function Modal({ show, title, message, onConfirm, onCancel, confirmLabel = 'Confirm', cancelLabel = 'Cancel', danger }) {
  if (!show) return null;
  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 9999,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)',
    }}>
      <motion.div initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
        style={{
          background: 'rgba(10,15,30,0.95)', border: '1px solid var(--border)',
          borderRadius: 12, padding: 24, maxWidth: 400, width: '90%',
        }}>
        {title && <div style={{ fontFamily: 'var(--font-hud)', fontSize: 11, letterSpacing: 2, color: danger ? 'var(--red)' : 'var(--cyan)', marginBottom: 12 }}>{title}</div>}
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-2)', marginBottom: 20, lineHeight: 1.5 }}>{message}</div>
        <div style={{ display: 'flex', gap: 10, justifyContent: 'flex-end' }}>
          <button onClick={onCancel} style={{
            padding: '8px 16px', borderRadius: 6, border: '1px solid var(--border)',
            background: 'transparent', color: 'var(--text-3)', cursor: 'pointer',
            fontFamily: 'var(--font)', fontSize: 11,
          }}>{cancelLabel}</button>
          <button onClick={onConfirm} style={{
            padding: '8px 16px', borderRadius: 6,
            background: danger ? 'rgba(255,51,85,0.2)' : 'rgba(0,240,255,0.15)',
            color: danger ? 'var(--red)' : 'var(--cyan)', cursor: 'pointer',
            fontFamily: 'var(--font)', fontSize: 11,
            border: `1px solid ${danger ? 'rgba(255,51,85,0.3)' : 'rgba(0,240,255,0.3)'}`,
          }}>{confirmLabel}</button>
        </div>
      </motion.div>
    </div>
  );
}
