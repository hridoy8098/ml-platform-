import { useState } from 'react';
import DataTable from 'react-data-table-component';
import useAtlasStore from '../../store/useAtlasStore';
import { analyzeMLData } from '../../services/api';
import { Label, Hint, Btn, ErrorBlock, card, tableStyles } from '../HUDComponents';

export default function AnalyzeTab() {
  const { mlDataset, mlAnalysis, setMLAnalysis } = useAtlasStore();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const run = async () => {
    if (!mlDataset?.session_id) return;
    setLoading(true); setError(null);
    const r = await analyzeMLData(mlDataset.session_id);
    r?.success ? setMLAnalysis(r) : setError(r || { message: 'Analysis failed' });
    setLoading(false);
  };
  const statsData = mlAnalysis?.analysis?.basic_stats ? Object.entries(mlAnalysis.analysis.basic_stats).map(([stat, vals]) => ({ stat, ...vals })) : [];
  const numericCols = statsData.length > 0 ? Object.keys(statsData[0]).filter(k => k !== 'stat').slice(0, 6) : [];
  const statsCols = mlAnalysis?.analysis?.basic_stats ? [
    { name: 'STAT', selector: r => r.stat, style: { color: 'var(--cyan)', fontFamily: 'var(--font-mono)', fontSize: 11 } },
    ...numericCols.map(col => ({
      name: col.toUpperCase(), selector: r => typeof r[col] === 'number' ? r[col].toFixed(3) : r[col] ?? '—',
      sortable: true, style: { fontFamily: 'var(--font-mono)', fontSize: 11 }
    }))
  ] : [];

  const analysis = mlAnalysis?.analysis;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <Btn onClick={run} disabled={loading || !mlDataset}>{loading ? '◌ ANALYZING...' : '📊 ANALYZE DATASET'}</Btn>
      {!mlDataset && <Hint center>Upload a dataset first</Hint>}
      {error && <ErrorBlock err={error} />}
      {analysis?.basic_stats && (
        <div style={card}><Label>DESCRIPTIVE STATISTICS</Label>
          <div style={{ marginTop: 8 }}><DataTable columns={statsCols} data={statsData} customStyles={tableStyles} dense pagination={false} /></div>
        </div>
      )}
      {analysis?.correlations && (
        <div style={card}><Label>CORRELATIONS (top 10)</Label>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
            {(() => {
              const pairs = [];
              const corr = analysis.correlations;
              Object.keys(corr).forEach(a => {
                Object.keys(corr[a]).forEach(b => {
                  if (a < b) pairs.push({ pair: `${a} × ${b}`, value: corr[a][b] });
                });
              });
              pairs.sort((x, y) => Math.abs(y.value) - Math.abs(x.value));
              return pairs.slice(0, 10).map(({ pair, value }) => {
                const v = Number(value);
                const color = Math.abs(v) > 0.7 ? 'var(--cyan)' : Math.abs(v) > 0.4 ? 'var(--yellow)' : 'var(--text-3)';
                return <span key={pair} style={{ padding: '3px 8px', borderRadius: 4, fontSize: 10, fontFamily: 'var(--font-mono)', border: `1px solid ${color}33`, color }}>{pair}: {v.toFixed(3)}</span>;
              });
            })()}
          </div>
        </div>
      )}
      {analysis?.outliers && Object.keys(analysis.outliers).length > 0 && (
        <div style={card}><Label>OUTLIERS</Label>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
            {Object.entries(analysis.outliers).map(([col, info]) => (
              <span key={col} style={{ padding: '3px 8px', borderRadius: 4, fontSize: 10, fontFamily: 'var(--font-mono)', background: 'rgba(255,136,0,0.08)', border: '1px solid rgba(255,136,0,0.3)', color: 'var(--orange)' }}>{col}: {info.count ?? info}</span>
            ))}
          </div>
        </div>
      )}
      {analysis?.skewness && Object.keys(analysis.skewness).length > 0 && (
        <div style={card}><Label>SKEWNESS</Label>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
            {Object.entries(analysis.skewness).map(([col, val]) => {
              const h = Math.abs(val) > 1;
              return <span key={col} style={{ padding: '3px 8px', borderRadius: 4, fontSize: 10, fontFamily: 'var(--font-mono)', background: h ? 'rgba(255,51,85,0.08)' : 'rgba(0,255,136,0.08)', border: `1px solid ${h ? 'rgba(255,51,85,0.3)' : 'rgba(0,255,136,0.2)'}`, color: h ? 'var(--red)' : 'var(--green)' }}>{col}: {Number(val).toFixed(3)}</span>;
            })}
          </div>
        </div>
      )}
      {!analysis && mlDataset && !loading && <Hint center>Click ANALYZE DATASET to see statistics</Hint>}
    </div>
  );
}