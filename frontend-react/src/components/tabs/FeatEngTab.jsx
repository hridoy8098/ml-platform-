import { useState } from 'react';
import { motion } from 'framer-motion';
import useAtlasStore from '../../store/useAtlasStore';
import { featureEngineerML } from '../../services/api';
import { Label, Hint, Btn, ErrorBlock, MetricCard, card, select, inputSt } from '../HUDComponents';

export default function FeatEngTab() {
  const { mlDataset } = useAtlasStore();
  const [operations, setOperations] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const allCols = mlDataset?.rawColumns || [];

  const opTypes = [
    { value: 'add_ratio', label: 'Ratio (A/B)', needsCols: 2, params: [] },
    { value: 'add_product', label: 'Product (A×B)', needsCols: 2, params: [] },
    { value: 'add_sum', label: 'Sum (A+B)', needsCols: 2, params: [] },
    { value: 'add_poly', label: 'Polynomial (col^n)', needsCols: 1, params: [{ key: 'degree', label: 'Degree', default: 2 }] },
    { value: 'binarize', label: 'Binarize (col>t)', needsCols: 1, params: [{ key: 'threshold', label: 'Threshold', default: 0 }] },
    { value: 'log_transform', label: 'Log Transform', needsCols: 1, params: [] },
    { value: 'rolling_mean', label: 'Rolling Mean', needsCols: 1, params: [{ key: 'window', label: 'Window', default: 3 }] },
  ];

  const addOp = (type) => {
    const def = opTypes.find(o => o.value === type);
    const op = { type, id: Date.now(), col_a: '', col_b: '', name: '' };
    if (def?.params) def.params.forEach(p => { op[p.key] = p.default; });
    setOperations(p => [...p, op]);
  };
  const updateOp = (id, field, value) => setOperations(p => p.map(o => o.id === id ? { ...o, [field]: value } : o));
  const removeOp = (id) => setOperations(p => p.filter(o => o.id !== id));

  const buildPayload = () => operations.map(op => {
    const def = opTypes.find(o => o.value === op.type);
    const p = { type: op.type };
    if (def.needsCols >= 1 && op.col_a) p.col_a = op.col_a;
    if (def.needsCols >= 2 && op.col_b) p.col_b = op.col_b;
    if (op.name) p.name = op.name;
    def?.params?.forEach(param => { if (op[param.key] != null) p[param.key] = op[param.key]; });
    return p;
  });

  const run = async () => {
    if (!mlDataset?.session_id || operations.length === 0) return;
    setLoading(true); setError(null); setResult(null);
    const r = await featureEngineerML({ session_id: mlDataset.session_id, operations: buildPayload() });
    r?.success ? setResult(r) : setError(r || { message: 'Feature engineering failed' });
    setLoading(false);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {!mlDataset && <Hint center>Upload a dataset first</Hint>}
      {mlDataset && (
        <>
          <div style={card}><Label>OPERATIONS</Label>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
              {opTypes.map(op => (
                <motion.button key={op.value} whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.97 }}
                  onClick={() => addOp(op.value)}
                  style={{ padding: '6px 12px', borderRadius: 6, border: '1px solid var(--border)', background: 'rgba(0,240,255,0.06)', color: 'var(--cyan)', fontSize: 10, fontFamily: 'var(--font-mono)', cursor: 'pointer' }}>
                  + {op.label}
                </motion.button>
              ))}
            </div>
            {operations.length === 0 && <div style={{ fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', marginTop: 8, opacity: 0.6 }}>Click an operation to add it</div>}
            {operations.length > 0 && (
              <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 6 }}>
                {operations.map(op => {
                  const def = opTypes.find(t => t.value === op.type);
                  return (
                    <div key={op.id} style={{ padding: '8px 10px', borderRadius: 6, background: 'rgba(0,20,16,0.5)', border: '1px solid var(--border)' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                        <span style={{ fontSize: 10, color: 'var(--cyan)', fontFamily: 'var(--font-mono)', flex: 1 }}>{def?.label || op.type}</span>
                        <button onClick={() => removeOp(op.id)} style={{ padding: '2px 8px', borderRadius: 4, border: '1px solid rgba(255,51,85,0.3)', background: 'transparent', color: 'var(--red)', fontSize: 10, cursor: 'pointer' }}>✕</button>
                      </div>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                        <select value={op.col_a} onChange={e => updateOp(op.id, 'col_a', e.target.value)} style={{ ...select, width: 'auto', minWidth: 100, fontSize: 10 }}>
                          <option value="">— Col A —</option>{allCols.map(c => <option key={c} value={c}>{c}</option>)}
                        </select>
                        {def?.needsCols >= 2 && (
                          <select value={op.col_b} onChange={e => updateOp(op.id, 'col_b', e.target.value)} style={{ ...select, width: 'auto', minWidth: 100, fontSize: 10 }}>
                            <option value="">— Col B —</option>{allCols.map(c => <option key={c} value={c}>{c}</option>)}
                          </select>
                        )}
                        {def?.params?.map(param => (
                          <input key={param.key} type="number" value={op[param.key]} onChange={e => updateOp(op.id, param.key, Number(e.target.value))}
                            placeholder={param.label} style={{ ...inputSt, width: 70, fontSize: 10 }} title={param.label} />
                        ))}
                        <input value={op.name} onChange={e => updateOp(op.id, 'name', e.target.value)} placeholder="Name (opt)" style={{ ...inputSt, width: 100, fontSize: 10 }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
          <Btn onClick={run} disabled={loading || operations.length === 0}>{loading ? '◌ ENGINEERING...' : '🔧 RUN FEATURE ENGINEERING'}</Btn>
          {error && <ErrorBlock err={error} />}
          {result?.success && (
            <div style={{ ...card, borderColor: 'rgba(0,240,255,0.35)' }}>
              <Label style={{ color: 'var(--cyan)' }}>✓ ENGINEERING COMPLETE</Label>
              <div style={{ display: 'flex', gap: 10, marginTop: 10 }}>
                <MetricCard label="SHAPE" value={`${result.shape?.[0]} × ${result.shape?.[1]}`} color="var(--cyan)" small />
              </div>
              {result.columns && (
                <div style={{ marginTop: 10 }}><Label>COLUMNS</Label>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
                    {result.columns.map(c => <span key={c} style={{ padding: '3px 8px', borderRadius: 4, fontSize: 10, fontFamily: 'var(--font-mono)', background: 'rgba(0,170,255,0.08)', border: '1px solid rgba(0,170,255,0.2)', color: 'var(--blue)' }}>{c}</span>)}
                  </div>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}