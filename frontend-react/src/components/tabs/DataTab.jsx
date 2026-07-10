import { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import DataTable from 'react-data-table-component';
import { useDropzone } from 'react-dropzone';
import useAtlasStore from '../../store/useAtlasStore';
import { uploadMLFile, cleanMLData, loadMLSample } from '../../services/api';
import { Label, Stat, Hint, StatusLine, Btn, ErrorBlock, card, tableStyles } from '../HUDComponents';

export default function DataTab() {
  const { mlDataset, setMLDataset, setMLAnalysis, setMLTrainingResult, setMLPlotData } = useAtlasStore();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [cleaning, setCleaning] = useState(false);

  const dropMissingRows = async () => {
    if (!mlDataset?.session_id) return;
    setCleaning(true); setError(null);
    const res = await cleanMLData(mlDataset.session_id, [{ type: 'drop_na_all' }]);
    if (res?.success) {
      const cols = res.columns.map(c => ({
        name: c, selector: row => row[c], sortable: true, wrap: true,
        style: { fontSize: 11, fontFamily: 'var(--font-mono)' }
      }));
      setMLDataset({ ...res, columns: cols, rawColumns: res.columns, preview: res.preview, fileName: mlDataset.fileName, session_id: res.cleaned_session_id });
    } else setError(res || { message: 'Clean failed' });
    setCleaning(false);
  };

  const onDrop = useCallback(async (accepted) => {
    const file = accepted[0];
    if (!file) return;
    setLoading(true); setError(null);
    setMLAnalysis(null); setMLTrainingResult(null); setMLPlotData(null);
    const res = await uploadMLFile(file);
    if (res?.success) {
      const cols = res.columns.map(c => ({
        name: c, selector: row => row[c], sortable: true, wrap: true,
        style: { fontSize: 11, fontFamily: 'var(--font-mono)' }
      }));
      setMLDataset({ ...res, columns: cols, rawColumns: res.columns, preview: res.preview, fileName: file.name });
    } else setError(res || { message: 'Upload failed', suggestion: 'Check the file format.' });
    setLoading(false);
  }, [setMLDataset, setMLAnalysis, setMLTrainingResult, setMLPlotData]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, maxFiles: 1,
    accept: { 'text/csv': ['.csv'], 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'], 'application/json': ['.json'] },
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div {...getRootProps()} style={{
        border: `1px dashed ${isDragActive ? 'var(--cyan)' : 'var(--border)'}`,
        borderRadius: 10, padding: 20, textAlign: 'center', cursor: 'pointer',
        background: isDragActive ? 'rgba(0,240,255,0.06)' : 'var(--surface)',
        transition: 'all 0.2s', backdropFilter: 'blur(8px)',
      }}>
        <input {...getInputProps()} />
        <div style={{ fontSize: 28, marginBottom: 4 }}>📂</div>
        <div style={{ fontFamily: 'var(--font-hud)', fontSize: 10, color: isDragActive ? 'var(--cyan)' : 'var(--text-3)', letterSpacing: 1 }}>
          {isDragActive ? 'DROP FILE HERE' : 'DROP CSV / EXCEL / JSON'}
        </div>
        <div style={{ fontSize: 10, color: 'var(--text-3)', marginTop: 4, opacity: 0.6 }}>or click to browse</div>
      </div>
      {loading && <StatusLine text="Loading dataset..." />}
      {error && <ErrorBlock err={error} />}
      {mlDataset?.preview && (
        <>
          <div style={{ ...card, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div><Label>DATASET</Label><span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--cyan)' }}>{mlDataset.fileName}</span></div>
            <div style={{ display: 'flex', gap: 14 }}>
              <Stat label="ROWS" value={mlDataset.shape?.[0]?.toLocaleString()} />
              <Stat label="COLS" value={mlDataset.shape?.[1]} />
              <Stat label="PREVIEW" value={`${mlDataset.preview.length} rows`} color="var(--text-3)" />
            </div>
          </div>
          <div style={card}>
            <DataTable columns={mlDataset.columns} data={mlDataset.preview.slice(0, 50)}
              customStyles={tableStyles} pagination paginationPerPage={10}
              paginationRowsPerPageOptions={[5, 10, 20, 50]} highlightOnHover dense />
          </div>
          {mlDataset.dtypes && (
            <div style={card}><Label>COLUMN TYPES</Label>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
                {Object.entries(mlDataset.dtypes).map(([col, dt]) => {
                  const isF = dt.includes('float'), isI = dt.includes('int');
                  const c = isF ? 'var(--blue)' : isI ? 'var(--cyan)' : 'var(--magenta)';
                  const b = isF ? 'rgba(0,136,255,0.08)' : isI ? 'rgba(0,240,255,0.08)' : 'rgba(255,0,255,0.08)';
                  return <span key={col} style={{ padding: '3px 8px', borderRadius: 4, fontSize: 10, fontFamily: 'var(--font-mono)', background: b, border: `1px solid ${c}44`, color: c }}>{col}: {dt}</span>;
                })}
              </div>
            </div>
          )}
          {mlDataset.missing && Object.values(mlDataset.missing).some(v => v > 0) && (
            <div style={card}><Label>MISSING VALUES</Label>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
                {Object.entries(mlDataset.missing).filter(([, v]) => v > 0).map(([col, count]) => (
                  <span key={col} style={{ padding: '3px 8px', borderRadius: 4, fontSize: 10, fontFamily: 'var(--font-mono)', background: 'rgba(255,51,85,0.08)', border: '1px solid rgba(255,51,85,0.3)', color: 'var(--red)' }}>{col}: {count} missing</span>
                ))}
              </div>
              <Btn onClick={dropMissingRows} disabled={cleaning} style={{ marginTop: 8 }}>
                {cleaning ? '◌ CLEANING...' : '🗑 DROP MISSING ROWS'}
              </Btn>
            </div>
          )}
        </>
      )}
    </div>
  );
}