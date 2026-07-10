import { useState } from 'react';
import DataTable from 'react-data-table-component';
import useAtlasStore from '../../store/useAtlasStore';
import { listMLModels, deleteMLModel } from '../../services/api';
import { Label, Hint, Btn, card, tableStyles, Modal } from '../HUDComponents';

export default function ModelsTab() {
  const { mlModels, setMLModels } = useAtlasStore();
  const [loading, setLoading] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState(null);

  const refresh = async () => {
    setLoading(true);
    const r = await listMLModels();
    if (r?.success) setMLModels(r.models);
    setLoading(false);
  };

  const handleDeleteConfirm = async () => {
    if (!deleteTarget) return;
    await deleteMLModel(deleteTarget);
    setDeleteTarget(null);
    refresh();
  };

  const modelList = Array.isArray(mlModels) ? mlModels : [];
  const cols = [
    { name: 'MODEL', selector: r => r.model_id, sortable: true, grow: 2, cell: r => <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--cyan)', wordBreak: 'break-all' }}>{r.model_id}</span> },
    { name: 'TASK', selector: r => r.task, sortable: true, width: '100px', cell: r => { const c = r.task === 'classification' ? 'var(--cyan)' : r.task === 'regression' ? 'var(--blue)' : 'var(--text-3)'; return <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: c }}>{r.task || '—'}</span>; } },
    { name: 'SCORE', selector: r => r.accuracy ?? r.rmse, sortable: true, width: '90px', cell: r => { if (r.accuracy != null) return <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--cyan)' }}>{(r.accuracy * 100).toFixed(1)}%</span>; if (r.rmse != null) return <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--blue)' }}>RMSE {r.rmse}</span>; return <span style={{ color: 'var(--text-3)', fontSize: 10 }}>—</span>; } },
    { name: 'SIZE', selector: r => r.size, width: '70px', cell: r => <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-3)' }}>{r.size}</span> },
    { name: '', width: '70px', ignoreRowClick: true, cell: r => <button onClick={() => setDeleteTarget(r.model_id)} style={{ padding: '3px 10px', borderRadius: 4, border: '1px solid rgba(255,51,85,0.3)', background: 'rgba(255,51,85,0.08)', color: 'var(--red)', fontSize: 10, cursor: 'pointer', fontFamily: 'var(--font)' }}>DELETE</button> },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <Btn onClick={refresh} disabled={loading}>{loading ? '◌ REFRESHING...' : '🔄 REFRESH'}</Btn>
        <span style={{ fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>{modelList.length} model{modelList.length !== 1 ? 's' : ''}</span>
      </div>
      <div style={card}>
        {modelList.length === 0 ? <Hint center>No trained models yet</Hint> : <DataTable columns={cols} data={modelList} customStyles={tableStyles} pagination paginationPerPage={10} highlightOnHover dense />}
      </div>
      <Modal
        show={!!deleteTarget}
        title="DELETE MODEL"
        message={`Are you sure you want to delete model "${deleteTarget?.slice(0, 30)}..."? This action cannot be undone.`}
        onConfirm={handleDeleteConfirm}
        onCancel={() => setDeleteTarget(null)}
        confirmLabel="DELETE"
        danger
      />
    </div>
  );
}