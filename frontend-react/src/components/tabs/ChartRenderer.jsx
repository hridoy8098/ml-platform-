import { motion } from 'framer-motion';
import { card, Label } from '../HUDComponents';

const BAR_COLOR = '#00ffc8';
const MAX_BAR_W = 260;

export default function ChartRenderer({ data }) {
  if (!data) return null;

  if (data.plot_type === 'histogram') {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {Object.entries(data.data || {}).map(([col, { bins, counts }]) => {
          const max = Math.max(...counts, 1);
          return (
            <div key={col} style={card}><Label>{col} — HISTOGRAM</Label>
              <div style={{ display: 'flex', alignItems: 'flex-end', gap: 2, height: 70, marginTop: 10, overflowX: 'auto' }}>
                {counts.map((c, i) => (
                  <motion.div key={i} initial={{ height: 0 }} animate={{ height: `${(c / max) * 100}%` }} transition={{ duration: 0.4, delay: i * 0.01 }}
                    title={`${bins[i]?.toFixed(2)}: ${c}`} style={{ flex: 1, minWidth: 4, background: BAR_COLOR, borderRadius: '2px 2px 0 0', opacity: 0.7 }} />
                ))}
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 8, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', marginTop: 2 }}>
                <span>{bins?.at(0)?.toFixed(2)}</span>
                <span>{bins?.at(-1)?.toFixed(2)}</span>
              </div>
            </div>
          );
        })}
      </div>
    );
  }

  if (data.plot_type === 'bar') {
    const max = Math.max(...(data.data || []).map(d => d.count), 1);
    return (
      <div style={card}><Label>{data.label} — BAR CHART</Label>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 5, marginTop: 10 }}>
          {(data.data || []).map(({ name, count }) => (
            <div key={name} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 10, color: 'var(--text-2)', fontFamily: 'var(--font-mono)', width: 90, textAlign: 'right', flexShrink: 0 }}>{name}</span>
              <motion.div initial={{ width: 0 }} animate={{ width: `${(count / max) * MAX_BAR_W}px` }} transition={{ duration: 0.5, ease: 'easeOut' }}
                style={{ height: 14, background: 'rgba(0,240,255,0.25)', borderRadius: 3, border: '1px solid rgba(0,240,255,0.2)' }} />
              <span style={{ fontSize: 10, color: 'var(--cyan)', fontFamily: 'var(--font-mono)' }}>{count}</span>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (data.plot_type === 'boxplot') {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {Object.entries(data.data || {}).map(([col, stats]) => (
          <div key={col} style={card}><Label>{col} — BOX PLOT</Label>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginTop: 8 }}>
              {[{ label: 'MIN', value: stats.min, color: 'var(--red)' }, { label: 'Q1', value: stats.q1, color: 'var(--orange)' }, { label: 'MEDIAN', value: stats.median, color: 'var(--cyan)' }, { label: 'Q3', value: stats.q3, color: 'var(--orange)' }, { label: 'MAX', value: stats.max, color: 'var(--red)' }].map(({ label, value, color }) => (
                <div key={label} style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 8, color: 'var(--text-3)', fontFamily: 'var(--font-hud)', letterSpacing: 1 }}>{label}</div>
                  <div style={{ fontSize: 12, color, fontFamily: 'var(--font-mono)', marginTop: 2 }}>{Number(value).toFixed(3)}</div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (data.plot_type === 'scatter') {
    const pts = data.data || [];
    const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
    const xMin = Math.min(...xs), xMax = Math.max(...xs), yMin = Math.min(...ys), yMax = Math.max(...ys);
    const W = 280, H = 160;
    const px = v => ((v - xMin) / (xMax - xMin || 1)) * W;
    const py = v => H - ((v - yMin) / (yMax - yMin || 1)) * H;
    return (
      <div style={card}><Label>{data.x_label} vs {data.y_label} — SCATTER</Label>
        <svg width={W} height={H} style={{ marginTop: 10, overflow: 'visible' }}>
          {pts.slice(0, 500).map((p, i) => <circle key={i} cx={px(p.x)} cy={py(p.y)} r={2.5} fill="rgba(0,240,255,0.4)" stroke="none" />)}
        </svg>
      </div>
    );
  }

  if (data.plot_type === 'correlation') {
    const cols = data.columns || [];
    const byPair = {};
    (data.data || []).forEach(({ row, col, value }) => { byPair[`${row}||${col}`] = value; });
    const cellW = Math.min(50, Math.floor(300 / (cols.length + 1)));
    return (
      <div style={{ ...card, overflowX: 'auto' }}><Label>CORRELATION MATRIX</Label>
        <table style={{ borderCollapse: 'collapse', marginTop: 10, fontSize: 9, fontFamily: 'var(--font-mono)' }}>
          <thead><tr><td style={{ width: cellW }} />{cols.map(c => <th key={c} style={{ width: cellW, color: 'var(--text-3)', padding: '2px 3px', textAlign: 'center', fontSize: 8, maxWidth: cellW }}>{c.slice(0, 5)}</th>)}</tr></thead>
          <tbody>{cols.map(row => (
            <tr key={row}><td style={{ color: 'var(--text-3)', fontSize: 8, paddingRight: 4, textAlign: 'right', maxWidth: cellW }}>{row.slice(0, 5)}</td>
              {cols.map(col => {
                const v = byPair[`${row}||${col}`] ?? 0;
                const heat = Math.abs(v) > 0.6 ? `rgba(0,255,200,${0.25 + Math.abs(v) * 0.4})` : Math.abs(v) > 0.3 ? `rgba(255,238,0,${Math.abs(v) * 0.3})` : `rgba(255,255,255,${Math.abs(v) * 0.1})`;
                return <td key={col} title={`${row} × ${col}: ${v.toFixed(3)}`} style={{ width: cellW, height: cellW, background: heat, textAlign: 'center', fontSize: 7, color: Math.abs(v) > 0.3 ? '#fff' : 'var(--text-3)', border: '1px solid rgba(0,240,255,0.06)' }}>{v.toFixed(2)}</td>;
              })}
            </tr>
          ))}</tbody>
        </table>
      </div>
    );
  }

  return <div style={{ ...card, color: 'var(--text-3)', fontSize: 11 }}>Unknown chart type: {data.plot_type}</div>;
}