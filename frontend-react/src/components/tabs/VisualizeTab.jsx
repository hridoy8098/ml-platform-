import { useState } from 'react';
import useAtlasStore from '../../store/useAtlasStore';
import { getMLChartData } from '../../services/api';
import { Label, Hint, Btn, ErrorBlock, card, select } from '../HUDComponents';
import ChartRenderer from './ChartRenderer';

export default function VisualizeTab() {
  const { mlDataset, mlPlotData, setMLPlotData } = useAtlasStore();
  const [plotType, setPlotType] = useState('histogram');
  const [selectedCols, setSelectedCols] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const allCols = mlDataset?.rawColumns || [];
  const numericCols = mlDataset?.dtypes ? Object.entries(mlDataset.dtypes).filter(([, t]) => t.includes('int') || t.includes('float')).map(([c]) => c) : [];

  const generate = async () => {
    if (!mlDataset?.session_id) return;
    setLoading(true); setError(null); setMLPlotData(null);
    const r = await getMLChartData({ session_id: mlDataset.session_id, plot_type: plotType, columns: selectedCols.length > 0 ? selectedCols : undefined });
    r?.success ? setMLPlotData(r) : setError(r || { message: 'Chart generation failed' });
    setLoading(false);
  };

  const toggleCol = col => setSelectedCols(p => p.includes(col) ? p.filter(c => c !== col) : [...p, col]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {!mlDataset && <Hint center>Upload a dataset first</Hint>}
      {mlDataset && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            <div style={card}><Label>PLOT TYPE</Label>
              <select value={plotType} onChange={e => { setPlotType(e.target.value); setSelectedCols([]); }} style={{ ...select, marginTop: 8 }}>
                <option value="histogram">📊 Histogram</option>
                <option value="scatter">🔵 Scatter</option>
                <option value="correlation">🔥 Correlation</option>
                <option value="boxplot">📦 Box Plot</option>
                <option value="bar">📊 Bar Chart</option>
              </select>
            </div>
            <div style={card}><Label>COLUMNS</Label>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 8 }}>
                {(plotType === 'bar' ? allCols : numericCols).map(col => (
                  <span key={col} onClick={() => toggleCol(col)} style={{
                    padding: '3px 8px', borderRadius: 4, fontSize: 10, fontFamily: 'var(--font-mono)', cursor: 'pointer',
                    background: selectedCols.includes(col) ? 'rgba(0,240,255,0.15)' : 'rgba(0,240,255,0.04)',
                    border: `1px solid ${selectedCols.includes(col) ? 'rgba(0,240,255,0.5)' : 'var(--border)'}`,
                    color: selectedCols.includes(col) ? 'var(--cyan)' : 'var(--text-3)', transition: 'all 0.15s'
                  }}>{col}</span>
                ))}
              </div>
            </div>
          </div>
          <Btn onClick={generate} disabled={loading}>{loading ? '◌ GENERATING...' : '📈 GENERATE CHART'}</Btn>
          {error && <ErrorBlock err={error} />}
          {mlPlotData && <ChartRenderer data={mlPlotData} />}
        </>
      )}
    </div>
  );
}