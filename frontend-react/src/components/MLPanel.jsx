import { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import DataTable from 'react-data-table-component';
import { useDropzone } from 'react-dropzone';
import useAtlasStore from '../store/useAtlasStore';
import {
  uploadMLFile, analyzeMLData, trainMLModel,
  predictML, listMLModels, deleteMLModel,
  getMLModelInfo, getMLJobStatus, evaluateMLModel, loadMLSample,
  clusterMLData, dimReduceML, trainMLNeural, tuneMLHyperparams,
  kfoldML, featureEngineerML, getMLChartData,
  getMLConfusionMatrix, getMLROCCurve, getMLElbowPlot, getMLFeatureImportance,
  cleanMLData,
} from '../services/api';
import { Label, Stat, Hint, StatusLine, Btn, ErrorBlock, MetricCard, Modal, card, select, inputSt, btnSt, tableStyles } from './HUDComponents';
import useJobPolling from '../hooks/useJobPolling';

const TABS = [
  { id: 'data',      icon: '📂', label: 'DATA' },
  { id: 'analyze',   icon: '📊', label: 'ANALYZE' },
  { id: 'train',     icon: '🤖', label: 'TRAIN' },
  { id: 'cluster',   icon: '🔘', label: 'CLUSTER' },
  { id: 'dimred',    icon: '📉', label: 'DIMRED' },
  { id: 'neural',    icon: '🧠', label: 'NEURAL' },
  { id: 'hptune',    icon: '⚙️',  label: 'HPTUNE' },
  { id: 'kfold',     icon: '♻️',  label: 'KFOLD' },
  { id: 'feateng',   icon: '🔧', label: 'FEATENG' },
  { id: 'evaluate',  icon: '📋', label: 'EVALUATE' },
  { id: 'predict',   icon: '🔮', label: 'PREDICT' },
  { id: 'models',    icon: '📦', label: 'MODELS' },
  { id: 'visualize', icon: '📈', label: 'VISUALIZE' },
];

export default function MLPanel() {
  const [activeTab, setActiveTab] = useState('data');
  const { setMLModels } = useAtlasStore();

  useEffect(() => {
    listMLModels().then(r => { if (r?.success) setMLModels(r.models); });
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6, height: '100%' }}>
      <div style={{
        display: 'flex', gap: 2, flexWrap: 'wrap',
        background: 'rgba(5,5,15,0.6)', borderRadius: 8, padding: 3,
        border: '1px solid var(--border)',
      }}>
        {TABS.map(tab => (
          <motion.button key={tab.id}
            whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
            onClick={() => setActiveTab(tab.id)}
            style={{
              flex: '1 0 auto', padding: '6px 8px', borderRadius: 6, border: 'none', cursor: 'pointer',
              fontFamily: 'var(--font-hud)', fontSize: 8, letterSpacing: 1.2,
              background: activeTab === tab.id ? 'var(--cyan-dim)' : 'transparent',
              color: activeTab === tab.id ? 'var(--cyan)' : 'var(--text-3)',
              transition: 'all 0.2s', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 3,
              position: 'relative',
            }}>
            {activeTab === tab.id && (
              <motion.div layoutId="tab-indicator"
                style={{
                  position: 'absolute', inset: 0, borderRadius: 6,
                  border: '1px solid rgba(0,240,255,0.2)',
                  boxShadow: '0 0 12px rgba(0,240,255,0.06)',
                }}
                transition={{ type: 'spring', stiffness: 400, damping: 30 }}
              />
            )}
            <span style={{ fontSize: 11, position: 'relative', zIndex: 1 }}>{tab.icon}</span>
            <span style={{ position: 'relative', zIndex: 1 }}>{tab.label}</span>
          </motion.button>
        ))}
      </div>

      <div style={{ flex: 1, overflow: 'auto', paddingRight: 2 }}>
        <AnimatePresence mode="wait">
          <motion.div key={activeTab}
            initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }} transition={{ duration: 0.12 }}>
            {activeTab === 'data'      && <DataTab />}
            {activeTab === 'analyze'   && <AnalyzeTab />}
            {activeTab === 'train'     && <TrainTab />}
            {activeTab === 'cluster'   && <ClusterTab />}
            {activeTab === 'dimred'    && <DimRedTab />}
            {activeTab === 'neural'    && <NeuralTab />}
            {activeTab === 'hptune'    && <HPTuneTab />}
            {activeTab === 'kfold'     && <KfoldTab />}
            {activeTab === 'feateng'   && <FeatEngTab />}
            {activeTab === 'evaluate'  && <EvaluateTab />}
            {activeTab === 'predict'   && <PredictTab />}
            {activeTab === 'models'    && <ModelsTab />}
            {activeTab === 'visualize' && <VisualizeTab />}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}

function DataTab() {
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

function AnalyzeTab() {
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

function TrainTab() {
  const { mlDataset, mlTrainingResult, setMLTrainingResult, setMLModels } = useAtlasStore();
  const [task, setTask] = useState('classification');
  const [target, setTarget] = useState('');
  const [algorithm, setAlgorithm] = useState('auto');
  const [testSize, setTestSize] = useState(0.2);
  const [jobId, setJobId] = useState(null);
  const [error, setError] = useState(null);

  const onDone = (result) => {
    setMLTrainingResult(result);
    listMLModels().then(r2 => { if (r2?.success) setMLModels(r2.models); });
  };
  const jobInfo = useJobPolling({ jobId, onDone, onError: (e) => setError(e) });

  const runTrain = async () => {
    if (!mlDataset?.session_id || !target) return;
    setError(null); setMLTrainingResult(null);
    const r = await trainMLModel({ task, session_id: mlDataset.session_id, target, algorithm, test_size: testSize });
    r?.job_id ? setJobId(r.job_id) : setError(r || { message: 'Could not start training' });
  };

  const allCols = mlDataset?.rawColumns || [];
  const isTraining = !!jobId;
  const algoOpts = task === 'auto' ? [] : [
    { value: 'random_forest', label: '🌲 Random Forest' }, { value: 'gradient_boost', label: '🚀 Gradient Boost' },
    { value: task === 'classification' ? 'logistic' : 'linear', label: task === 'classification' ? '📈 Logistic' : '📈 Linear' },
    { value: task === 'classification' ? 'svm' : 'svr', label: '🔲 ' + (task === 'classification' ? 'SVM' : 'SVR') },
    { value: 'knn', label: '📊 KNN' }, { value: 'decision_tree', label: '🌳 Decision Tree' },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {!mlDataset && <Hint center>Upload a dataset first</Hint>}
      {mlDataset && (
        <>
          <div style={card}><Label>TASK TYPE</Label>
            <div style={{ display: 'flex', gap: 6, marginTop: 8 }}>
              {[{ id: 'classification', label: 'Classification', icon: '🏷️' }, { id: 'regression', label: 'Regression', icon: '📉' }, { id: 'auto', label: 'AutoML', icon: '⚡' }].map(t => (
                <motion.button key={t.id} whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.97 }}
                  onClick={() => { setTask(t.id); setAlgorithm('auto'); }}
                  style={{ flex: 1, padding: '9px 12px', borderRadius: 8, cursor: 'pointer', border: `1px solid ${task === t.id ? 'var(--border-glow)' : 'var(--border)'}`, background: task === t.id ? 'var(--cyan-dim)' : 'var(--surface)', color: task === t.id ? 'var(--cyan)' : 'var(--text-3)', fontFamily: 'var(--font)', fontSize: 13, transition: 'all 0.2s', fontWeight: 600 }}>
                  {t.icon} {t.label}
                </motion.button>
              ))}
            </div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            <div style={card}><Label>TARGET COLUMN</Label><select value={target} onChange={e => setTarget(e.target.value)} style={{ ...select, marginTop: 8 }}><option value="">— Select target —</option>{allCols.map(c => <option key={c} value={c}>{c}</option>)}</select>
              <div style={{ fontSize: 10, color: 'var(--text-3)', marginTop: 6, lineHeight: 1.5, opacity: 0.8 }}>Target = "answer" column model tries to predict. Rest = features (input). Model learns feature→target mapping during training.</div>
            </div>
            <div style={card}><Label>TEST SIZE</Label><input type="number" value={testSize} onChange={e => setTestSize(Number(e.target.value))} min={0.05} max={0.5} step={0.05} style={{ ...inputSt, marginTop: 8 }} /></div>
          </div>
          <div style={card}><Label>ALGORITHM</Label><select value={algorithm} onChange={e => setAlgorithm(e.target.value)} style={{ ...select, marginTop: 8 }} disabled={task === 'auto'}><option value="auto">🤖 Auto (Best)</option>{algoOpts.map(a => <option key={a.value} value={a.value}>{a.label}</option>)}</select></div>
          <Btn onClick={runTrain} disabled={isTraining || !target}>{isTraining ? '◌ TRAINING...' : '🤖 TRAIN MODEL'}</Btn>
          {error && <ErrorBlock err={error} />}
          {isTraining && jobInfo && (
            <div style={card}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                <span style={{ fontFamily: 'var(--font-hud)', fontSize: 9, color: 'var(--purple)', letterSpacing: 1 }}>{jobInfo.step || 'TRAINING...'}</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--purple)' }}>{jobInfo.progress || 0}%</span>
              </div>
              <div style={{ height: 6, background: 'rgba(170,68,255,0.12)', borderRadius: 3, overflow: 'hidden' }}>
                <motion.div animate={{ width: `${jobInfo.progress || 0}%` }} transition={{ duration: 0.4, ease: 'easeOut' }}
                  style={{ height: '100%', background: 'linear-gradient(90deg, var(--purple), var(--magenta))', borderRadius: 3, boxShadow: '0 0 10px rgba(170,68,255,0.3)' }} />
              </div>
            </div>
          )}
          {mlTrainingResult?.success && (
            <div style={{ ...card, borderColor: 'rgba(0,240,255,0.35)' }}>
              <Label style={{ color: 'var(--cyan)' }}>✓ TRAINING COMPLETE</Label>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 8, marginTop: 10 }}>
                {mlTrainingResult.algorithm && <MetricCard label="ALGORITHM" value={mlTrainingResult.algorithm.replace(/_/g, ' ').toUpperCase()} color="var(--magenta)" small />}
                {mlTrainingResult.accuracy != null && <MetricCard label="ACCURACY" value={(mlTrainingResult.accuracy * 100).toFixed(1) + '%'} color="var(--cyan)" />}
                {mlTrainingResult.rmse != null && <MetricCard label="RMSE" value={mlTrainingResult.rmse} color="var(--blue)" />}
                {mlTrainingResult.r2 != null && <MetricCard label="R²" value={mlTrainingResult.r2?.toFixed(4)} color="var(--blue)" />}
                {mlTrainingResult.features && <MetricCard label="FEATURES" value={`${mlTrainingResult.features.length} cols`} color="var(--yellow)" small />}
                {mlTrainingResult.cv_score != null && <MetricCard label="CV SCORE" value={(mlTrainingResult.cv_score * 100).toFixed(2) + '%'} color="var(--green)" />}
              </div>
              {mlTrainingResult.model_id && <div style={{ marginTop: 10, fontSize: 9, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>ID: {mlTrainingResult.model_id}</div>}
            </div>
          )}
        </>
      )}
    </div>
  );
}

function ClusterTab() {
  const { mlDataset } = useAtlasStore();
  const [algorithm, setAlgorithm] = useState('kmeans');
  const [nClusters, setNClusters] = useState(3);
  const [eps, setEps] = useState(0.5);
  const [minSamples, setMinSamples] = useState(5);
  const [result, setResult] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [error, setError] = useState(null);

  const onDone = (r) => setResult(r);
  const jobInfo = useJobPolling({ jobId, onDone, onError: (e) => setError(e) });

  const run = async () => {
    if (!mlDataset?.session_id) return;
    setError(null); setResult(null);
    const r = await clusterMLData({ session_id: mlDataset.session_id, algorithm, n_clusters: nClusters, eps, min_samples: minSamples });
    r?.job_id ? setJobId(r.job_id) : setError(r || { message: 'Clustering failed' });
  };

  const isRunning = !!jobId;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {!mlDataset && <Hint center>Upload a dataset first</Hint>}
      {mlDataset && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            <div style={card}><Label>ALGORITHM</Label><select value={algorithm} onChange={e => setAlgorithm(e.target.value)} style={{ ...select, marginTop: 8 }}><option value="kmeans">K-Means</option><option value="dbscan">DBSCAN</option><option value="meanshift">Mean Shift</option><option value="agglomerative">Agglomerative</option></select></div>
            <div style={card}><Label>N CLUSTERS</Label><input type="number" value={nClusters} onChange={e => setNClusters(Number(e.target.value))} min={2} max={50} style={{ ...inputSt, marginTop: 8 }} /></div>
          </div>
          {algorithm === 'dbscan' && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              <div style={card}><Label>EPS</Label><input type="number" value={eps} onChange={e => setEps(Number(e.target.value))} step={0.1} style={{ ...inputSt, marginTop: 8 }} /></div>
              <div style={card}><Label>MIN SAMPLES</Label><input type="number" value={minSamples} onChange={e => setMinSamples(Number(e.target.value))} min={1} style={{ ...inputSt, marginTop: 8 }} /></div>
            </div>
          )}
          <Btn onClick={run} disabled={isRunning}>{isRunning ? '◌ CLUSTERING...' : '🔘 RUN CLUSTERING'}</Btn>
          {error && <ErrorBlock err={error} />}
          {isRunning && jobInfo && <ProgressBar info={jobInfo} />}
          {result?.success && (
            <div style={{ ...card, borderColor: 'rgba(0,240,255,0.35)' }}>
              <Label style={{ color: 'var(--cyan)' }}>✓ CLUSTERING COMPLETE</Label>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginTop: 10 }}>
                <MetricCard label="CLUSTERS" value={result.n_clusters || result.algorithm} color="var(--cyan)" />
                {result.inertia != null && <MetricCard label="INERTIA" value={result.inertia.toFixed(2)} color="var(--blue)" small />}
                {result.n_noise != null && <MetricCard label="NOISE" value={result.n_noise} color="var(--red)" small />}
              </div>
              {result.cluster_centers && (
                <div style={{ marginTop: 10 }}><Label>CENTERS</Label>
                  {result.cluster_centers.map((c, i) => (
                    <div key={i} style={{ fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', marginTop: 4 }}>Cluster {c.cluster} ({c.size} pts): [{c.center.map(v => v.toFixed(3)).join(', ')}]</div>
                  ))}
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}

function DimRedTab() {
  const { mlDataset } = useAtlasStore();
  const [algorithm, setAlgorithm] = useState('pca');
  const [nComponents, setNComponents] = useState(2);
  const [result, setResult] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [error, setError] = useState(null);

  const onDone = (r) => setResult(r);
  const jobInfo = useJobPolling({ jobId, onDone, onError: (e) => setError(e) });

  const run = async () => {
    if (!mlDataset?.session_id) return;
    setError(null); setResult(null);
    const r = await dimReduceML({ session_id: mlDataset.session_id, algorithm, n_components: nComponents });
    r?.job_id ? setJobId(r.job_id) : setError(r || { message: 'Dim reduction failed' });
  };

  const isRunning = !!jobId;
  const pts = result?.components || [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {!mlDataset && <Hint center>Upload a dataset first</Hint>}
      {mlDataset && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            <div style={card}><Label>ALGORITHM</Label><select value={algorithm} onChange={e => setAlgorithm(e.target.value)} style={{ ...select, marginTop: 8 }}><option value="pca">PCA</option><option value="tsne">t-SNE</option></select></div>
            <div style={card}><Label>N COMPONENTS</Label><input type="number" value={nComponents} onChange={e => setNComponents(Number(e.target.value))} min={2} max={10} style={{ ...inputSt, marginTop: 8 }} /></div>
          </div>
          <Btn onClick={run} disabled={isRunning}>{isRunning ? '◌ REDUCING...' : '📉 RUN DIM REDUCTION'}</Btn>
          {error && <ErrorBlock err={error} />}
          {isRunning && jobInfo && <ProgressBar info={jobInfo} />}
          {result?.success && (
            <>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
                <MetricCard label="ALGORITHM" value={result.algorithm?.toUpperCase()} color="var(--magenta)" small />
                <MetricCard label="COMPONENTS" value={result.n_components} color="var(--cyan)" />
                {result.explained_variance_ratio && <MetricCard label="VAR RATIO" value={result.explained_variance_ratio.map(v => (v * 100).toFixed(1) + '%').join(', ')} color="var(--blue)" small />}
                {result.cumulative_variance != null && <MetricCard label="CUM VAR" value={(result.cumulative_variance * 100).toFixed(1) + '%'} color="var(--cyan)" />}
              </div>
              {pts.length > 0 && nComponents >= 2 && (
                <div style={card}><Label>PC1 vs PC2 — SCATTER</Label><DimRedScatter points={pts} /></div>
              )}
              {result.features && (
                <div style={card}><Label>OUTPUT FEATURES</Label>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
                    {result.features.map(f => <span key={f} style={{ padding: '3px 8px', borderRadius: 4, fontSize: 10, fontFamily: 'var(--font-mono)', background: 'rgba(0,240,255,0.06)', border: '1px solid var(--border)', color: 'var(--cyan)' }}>{f}</span>)}
                  </div>
                </div>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}

function DimRedScatter({ points }) {
  if (!points?.length) return null;
  const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
  const xMin = Math.min(...xs), xMax = Math.max(...xs), yMin = Math.min(...ys), yMax = Math.max(...ys);
  const W = 280, H = 160;
  const px = v => ((v - xMin) / (xMax - xMin || 1)) * W;
  const py = v => H - ((v - yMin) / (yMax - yMin || 1)) * H;
  return (
    <svg width={W} height={H} style={{ marginTop: 10, overflow: 'visible' }}>
      {points.map((p, i) => <circle key={i} cx={px(p[0])} cy={py(p[1])} r={2.5} fill="rgba(0,240,255,0.45)" stroke="none" />)}
    </svg>
  );
}

function ProgressBar({ info }) {
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

function NeuralTab() {
  const { mlDataset } = useAtlasStore();
  const [target, setTarget] = useState('');
  const [taskType, setTaskType] = useState('classification');
  const [hiddenLayers, setHiddenLayers] = useState('100,50');
  const [maxIter, setMaxIter] = useState(500);
  const [result, setResult] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [error, setError] = useState(null);

  const onDone = (r) => setResult(r);
  const jobInfo = useJobPolling({ jobId, onDone, onError: (e) => setError(e) });

  const run = async () => {
    if (!mlDataset?.session_id || !target) return;
    setError(null); setResult(null);
    const r = await trainMLNeural({ session_id: mlDataset.session_id, target, task_type: taskType, hidden_layers: hiddenLayers, max_iter: maxIter });
    r?.job_id ? setJobId(r.job_id) : setError(r || { message: 'Neural training failed' });
  };

  const allCols = mlDataset?.rawColumns || [];
  const isRunning = !!jobId;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {!mlDataset && <Hint center>Upload a dataset first</Hint>}
      {mlDataset && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            <div style={card}><Label>TARGET COLUMN</Label><select value={target} onChange={e => setTarget(e.target.value)} style={{ ...select, marginTop: 8 }}><option value="">— Select target —</option>{allCols.map(c => <option key={c} value={c}>{c}</option>)}</select></div>
            <div style={card}><Label>TASK TYPE</Label><select value={taskType} onChange={e => setTaskType(e.target.value)} style={{ ...select, marginTop: 8 }}><option value="classification">Classification</option><option value="regression">Regression</option></select></div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            <div style={card}><Label>HIDDEN LAYERS</Label><input value={hiddenLayers} onChange={e => setHiddenLayers(e.target.value)} placeholder="100,50" style={{ ...inputSt, marginTop: 8 }} /><div style={{ fontSize: 8, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', marginTop: 4 }}>comma-separated</div></div>
            <div style={card}><Label>MAX ITERATIONS</Label><input type="number" value={maxIter} onChange={e => setMaxIter(Number(e.target.value))} min={100} max={5000} step={100} style={{ ...inputSt, marginTop: 8 }} /></div>
          </div>
          <Btn onClick={run} disabled={isRunning || !target}>{isRunning ? '◌ TRAINING MLP...' : '🧠 TRAIN NEURAL NETWORK'}</Btn>
          {error && <ErrorBlock err={error} />}
          {isRunning && jobInfo && <ProgressBar info={jobInfo} />}
          {result?.success && (
            <div style={{ ...card, borderColor: 'rgba(0,240,255,0.35)' }}>
              <Label style={{ color: 'var(--cyan)' }}>✓ MLP TRAINING COMPLETE</Label>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginTop: 10 }}>
                {result.accuracy != null && <MetricCard label="ACCURACY" value={(result.accuracy * 100).toFixed(1) + '%'} color="var(--cyan)" />}
                {result.rmse != null && <MetricCard label="RMSE" value={result.rmse} color="var(--blue)" />}
                {result.score != null && result.accuracy == null && <MetricCard label="SCORE" value={result.score} color="var(--cyan)" />}
                {result.loss != null && <MetricCard label="LOSS" value={result.loss} color="var(--magenta)" small />}
              </div>
              {result.model_id && <div style={{ marginTop: 10, fontSize: 9, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>ID: {result.model_id}</div>}
            </div>
          )}
        </>
      )}
    </div>
  );
}

function HPTuneTab() {
  const { mlDataset } = useAtlasStore();
  const [target, setTarget] = useState('');
  const [taskType, setTaskType] = useState('classification');
  const [algorithm, setAlgorithm] = useState('random_forest');
  const [cv, setCv] = useState(5);
  const [result, setResult] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [error, setError] = useState(null);

  const onDone = (r) => setResult(r);
  const jobInfo = useJobPolling({ jobId, onDone, onError: (e) => setError(e) });

  const run = async () => {
    if (!mlDataset?.session_id || !target) return;
    setError(null); setResult(null);
    const r = await tuneMLHyperparams({ session_id: mlDataset.session_id, target, task_type: taskType, algorithm, cv, scoring: 'accuracy' });
    r?.job_id ? setJobId(r.job_id) : setError(r || { message: 'Tuning failed' });
  };

  const allCols = mlDataset?.rawColumns || [];
  const isRunning = !!jobId;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {!mlDataset && <Hint center>Upload a dataset first</Hint>}
      {mlDataset && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            <div style={card}><Label>TARGET COLUMN</Label><select value={target} onChange={e => setTarget(e.target.value)} style={{ ...select, marginTop: 8 }}><option value="">— Select target —</option>{allCols.map(c => <option key={c} value={c}>{c}</option>)}</select></div>
            <div style={card}><Label>TASK TYPE</Label><select value={taskType} onChange={e => setTaskType(e.target.value)} style={{ ...select, marginTop: 8 }}><option value="classification">Classification</option><option value="regression">Regression</option></select></div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            <div style={card}><Label>ALGORITHM</Label><select value={algorithm} onChange={e => setAlgorithm(e.target.value)} style={{ ...select, marginTop: 8 }}><option value="random_forest">Random Forest</option><option value="svm">SVM</option><option value="logistic">Logistic Regression</option><option value="gradient_boost">Gradient Boost</option></select></div>
            <div style={card}><Label>CV FOLDS</Label><input type="number" value={cv} onChange={e => setCv(Number(e.target.value))} min={2} max={20} style={{ ...inputSt, marginTop: 8 }} /></div>
          </div>
          <Btn onClick={run} disabled={isRunning || !target}>{isRunning ? '◌ TUNING...' : '⚙️ START HYPERPARAMETER TUNING'}</Btn>
          {error && <ErrorBlock err={error} />}
          {isRunning && jobInfo && <ProgressBar info={jobInfo} />}
          {result?.success && (
            <div style={{ ...card, borderColor: 'rgba(0,240,255,0.35)' }}>
              <Label style={{ color: 'var(--cyan)' }}>✓ TUNING COMPLETE</Label>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginTop: 10 }}>
                {result.best_score != null && <MetricCard label="BEST SCORE" value={(result.best_score * 100).toFixed(2) + '%'} color="var(--cyan)" />}
                {result.best_params && <div style={{ fontSize: 10, color: 'var(--text-2)', fontFamily: 'var(--font-mono)' }}>Params: {JSON.stringify(result.best_params)}</div>}
                {result.cv && <MetricCard label="CV" value={result.cv} color="var(--yellow)" small />}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function KfoldTab() {
  const { mlDataset } = useAtlasStore();
  const [target, setTarget] = useState('');
  const [taskType, setTaskType] = useState('classification');
  const [algorithm, setAlgorithm] = useState('random_forest');
  const [nSplits, setNSplits] = useState(5);
  const [result, setResult] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [error, setError] = useState(null);

  const onDone = (r) => setResult(r);
  const jobInfo = useJobPolling({ jobId, onDone, onError: (e) => setError(e) });

  const run = async () => {
    if (!mlDataset?.session_id || !target) return;
    setError(null); setResult(null);
    const r = await kfoldML({ session_id: mlDataset.session_id, target, task_type: taskType, algorithm, n_splits: nSplits });
    r?.job_id ? setJobId(r.job_id) : setError(r || { message: 'K-Fold failed' });
  };

  const allCols = mlDataset?.rawColumns || [];
  const isRunning = !!jobId;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {!mlDataset && <Hint center>Upload a dataset first</Hint>}
      {mlDataset && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            <div style={card}><Label>TARGET COLUMN</Label><select value={target} onChange={e => setTarget(e.target.value)} style={{ ...select, marginTop: 8 }}><option value="">— Select target —</option>{allCols.map(c => <option key={c} value={c}>{c}</option>)}</select></div>
            <div style={card}><Label>TASK TYPE</Label><select value={taskType} onChange={e => setTaskType(e.target.value)} style={{ ...select, marginTop: 8 }}><option value="classification">Classification</option><option value="regression">Regression</option></select></div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            <div style={card}><Label>ALGORITHM</Label><select value={algorithm} onChange={e => setAlgorithm(e.target.value)} style={{ ...select, marginTop: 8 }}><option value="random_forest">Random Forest</option><option value="svm">SVM</option><option value="logistic">Logistic Regression</option><option value="gradient_boost">Gradient Boost</option></select></div>
            <div style={card}><Label>N SPLITS</Label><input type="number" value={nSplits} onChange={e => setNSplits(Number(e.target.value))} min={2} max={20} style={{ ...inputSt, marginTop: 8 }} /></div>
          </div>
          <Btn onClick={run} disabled={isRunning || !target}>{isRunning ? '◌ CROSS VALIDATING...' : '♻️ RUN K-FOLD CV'}</Btn>
          {error && <ErrorBlock err={error} />}
          {isRunning && jobInfo && <ProgressBar info={jobInfo} />}
          {result?.success && (
            <div style={{ ...card, borderColor: 'rgba(0,240,255,0.35)' }}>
              <Label style={{ color: 'var(--cyan)' }}>✓ K-FOLD CV COMPLETE</Label>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginTop: 10 }}>
                <MetricCard label="MEAN" value={(result.mean_score * 100).toFixed(2) + '%'} color="var(--cyan)" />
                <MetricCard label="STD" value={result.std_score ? (result.std_score * 100).toFixed(2) + '%' : '—'} color="var(--yellow)" small />
                <MetricCard label="FOLDS" value={result.n_splits} color="var(--magenta)" small />
              </div>
              {result.scores && (
                <div style={{ marginTop: 10 }}><Label>PER-FOLD SCORES</Label>
                  <div style={{ display: 'flex', alignItems: 'flex-end', gap: 6, height: 60, marginTop: 10 }}>
                    {result.scores.map((s, i) => (
                      <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flex: 1 }}>
                        <motion.div initial={{ height: 0 }} animate={{ height: `${s * 100}px` }} transition={{ duration: 0.4, delay: i * 0.05 }}
                          style={{ width: '100%', maxWidth: 24, background: 'rgba(0,240,255,0.25)', borderRadius: '3px 3px 0 0', border: '1px solid rgba(0,240,255,0.2)' }} />
                        <span style={{ fontSize: 7, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', marginTop: 3 }}>Fold {i + 1}</span>
                      </div>
                    ))}
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

function FeatEngTab() {
  const { mlDataset } = useAtlasStore();
  const [operations, setOperations] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const opTypes = [
    { value: 'polynomial', label: 'Polynomial (x²)' },
    { value: 'interaction', label: 'Interaction (a × b)' },
    { value: 'bin', label: 'Binning' },
  ];

  const addOp = (type) => setOperations(p => [...p, { type, id: Date.now() }]);
  const removeOp = (id) => setOperations(p => p.filter(o => o.id !== id));

  const run = async () => {
    if (!mlDataset?.session_id || operations.length === 0) return;
    setLoading(true); setError(null); setResult(null);
    const r = await featureEngineerML({ session_id: mlDataset.session_id, operations: operations.map(o => ({ type: o.type })) });
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
                {operations.map(op => (
                  <div key={op.id} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '6px 10px', borderRadius: 6, background: 'rgba(0,20,16,0.5)', border: '1px solid var(--border)' }}>
                    <span style={{ fontSize: 10, color: 'var(--cyan)', fontFamily: 'var(--font-mono)', flex: 1 }}>{opTypes.find(t => t.value === op.type)?.label || op.type}</span>
                    <button onClick={() => removeOp(op.id)} style={{ padding: '2px 8px', borderRadius: 4, border: '1px solid rgba(255,51,85,0.3)', background: 'transparent', color: 'var(--red)', fontSize: 10, cursor: 'pointer' }}>✕</button>
                  </div>
                ))}
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

function EvaluateTab() {
  const { mlModels, mlDataset, mlEvaluationData, setMLEvaluationData } = useAtlasStore();

  const [selectedModel, setSelectedModel] = useState('');
  const [metric, setMetric] = useState('standard');
  const [evalResult, setEvalResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const activeModels = Array.isArray(mlModels) ? mlModels : [];

  const run = async () => {
    if (!selectedModel || !mlDataset?.session_id) return;
    setLoading(true); setError(null); setEvalResult(null);
    let r;
    if (metric === 'standard') {
      r = await evaluateMLModel(mlDataset.session_id, selectedModel);
    } else if (metric === 'confusion_matrix') {
      r = await getMLConfusionMatrix(mlDataset.session_id, selectedModel);
    } else if (metric === 'roc_curve') {
      r = await getMLROCCurve(mlDataset.session_id, selectedModel);
    } else if (metric === 'feature_importance') {
      r = await getMLFeatureImportance(selectedModel);
    }
    r?.success ? setEvalResult(r) : setError(r || { message: 'Evaluation failed' });
    setLoading(false);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={card}><Label>METRIC</Label>
        <select value={metric} onChange={e => { setMetric(e.target.value); setEvalResult(null); }} style={{ ...select, marginTop: 8 }}>
          <option value="standard">Standard Metrics</option>
          <option value="confusion_matrix">Confusion Matrix</option>
          <option value="roc_curve">ROC Curve</option>
          <option value="feature_importance">Feature Importance</option>
        </select>
      </div>
      <div style={card}><Label>SELECT MODEL</Label>
        <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)} style={{ ...select, marginTop: 8 }}>
          <option value="">— Choose a trained model —</option>
          {activeModels.map(m => <option key={m.model_id} value={m.model_id}>{m.model_id} ({m.size})</option>)}
        </select>
      </div>
      <Btn onClick={run} disabled={loading || !selectedModel}>{loading ? '◌ EVALUATING...' : '📋 RUN EVALUATION'}</Btn>
      {error && <ErrorBlock err={error} />}

      {evalResult?.success && metric === 'standard' && (
        <div style={card}>
          <Label style={{ color: 'var(--cyan)' }}>✓ EVALUATION RESULTS</Label>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 8, marginTop: 10 }}>
            {evalResult.accuracy != null && <MetricCard label="ACCURACY" value={(evalResult.accuracy * 100).toFixed(1) + '%'} color="var(--cyan)" />}
            {evalResult.precision != null && <MetricCard label="PRECISION" value={evalResult.precision} color="var(--blue)" />}
            {evalResult.recall != null && <MetricCard label="RECALL" value={evalResult.recall} color="var(--green)" />}
            {evalResult.f1 != null && <MetricCard label="F1" value={evalResult.f1} color="var(--purple)" />}
            {evalResult.rmse != null && <MetricCard label="RMSE" value={evalResult.rmse} color="var(--blue)" />}
            {evalResult.mae != null && <MetricCard label="MAE" value={evalResult.mae} color="var(--yellow)" />}
            {evalResult.r2 != null && <MetricCard label="R²" value={evalResult.r2} color="var(--green)" />}
            {evalResult.auc_roc != null && <MetricCard label="AUC ROC" value={evalResult.auc_roc} color="var(--magenta)" />}
          </div>
        </div>
      )}

      {evalResult?.success && metric === 'confusion_matrix' && evalResult.matrix && (
        <div style={card}><Label>CONFUSION MATRIX</Label>
          {evalResult.labels && (
            <table style={{ borderCollapse: 'collapse', marginTop: 10, fontSize: 10, fontFamily: 'var(--font-mono)' }}>
              <thead><tr><td style={{ padding: 4, color: 'var(--text-3)' }} />{evalResult.labels.map(l => <th key={l} style={{ padding: '4px 8px', color: 'var(--cyan)', borderBottom: '1px solid var(--border)' }}>{l}</th>)}</tr></thead>
              <tbody>{evalResult.matrix.map((row, i) => (
                <tr key={i}><td style={{ padding: '4px 8px', color: 'var(--magenta)', borderRight: '1px solid var(--border)' }}>{evalResult.labels?.[i] || i}</td>
                  {row.map((v, j) => <td key={j} style={{ padding: '6px 10px', textAlign: 'center', background: v > 0 ? 'rgba(0,240,255,0.08)' : 'transparent', color: v > 0 ? 'var(--cyan)' : 'var(--text-3)' }}>{v}</td>)}
                </tr>
              ))}</tbody>
            </table>
          )}
        </div>
      )}

      {evalResult?.success && metric === 'roc_curve' && evalResult.fpr && (
        <div style={card}><Label>ROC CURVE {evalResult.auc != null && <span style={{ color: 'var(--cyan)' }}>(AUC: {evalResult.auc.toFixed(3)})</span>}</Label>
          <svg width={280} height={160} style={{ marginTop: 10 }}>
            <line x1={20} y1={140} x2={260} y2={140} stroke="rgba(0,240,255,0.15)" />
            <line x1={20} y1={140} x2={20} y2={20} stroke="rgba(0,240,255,0.15)" />
            <line x1={20} y1={140} x2={260} y2={20} stroke="rgba(0,240,255,0.1)" strokeDasharray="3" />
            <polyline fill="none" stroke="#00ffc8" strokeWidth={1.5}
              points={evalResult.fpr.map((fpr, i) => {
                const x = 20 + fpr * 240, y = 140 - evalResult.tpr[i] * 120;
                return `${x},${y}`;
              }).join(' ')} />
          </svg>
        </div>
      )}

      {evalResult?.success && metric === 'feature_importance' && evalResult.data && (
        <div style={card}><Label>FEATURE IMPORTANCE</Label>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 5, marginTop: 10 }}>
            {evalResult.data.map(({ feature, importance }) => {
              const pct = Math.abs(importance);
              return (
                <div key={feature} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ width: 120, fontSize: 10, color: 'var(--text-2)', fontFamily: 'var(--font-mono)', textAlign: 'right', flexShrink: 0 }}>{feature}</span>
                  <div style={{ flex: 1, height: 14, background: 'rgba(0,0,0,0.3)', borderRadius: 3, overflow: 'hidden', border: '1px solid var(--border)' }}>
                    <motion.div initial={{ width: 0 }} animate={{ width: `${pct * 100}%` }} transition={{ duration: 0.5, ease: 'easeOut' }}
                      style={{ height: '100%', background: 'linear-gradient(90deg, var(--cyan), var(--magenta))', borderRadius: 3 }} />
                  </div>
                  <span style={{ width: 40, fontSize: 9, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>{(pct * 100).toFixed(1)}%</span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

function PredictTab() {
  const { mlModels, mlPredictionResult, setMLPredictionResult } = useAtlasStore();
  const [selectedModel, setSelectedModel] = useState('');
  const [modelInfo, setModelInfo] = useState(null);
  const [inputValues, setInputValues] = useState({});
  const [loading, setLoading] = useState(false);
  const [infoLoading, setInfoLoading] = useState(false);
  const [error, setError] = useState(null);

  const activeModels = Array.isArray(mlModels) ? mlModels : [];

  useEffect(() => {
    if (!selectedModel) { setModelInfo(null); setInputValues({}); return; }
    setInfoLoading(true); setInputValues({}); setMLPredictionResult(null); setError(null);
    getMLModelInfo(selectedModel).then(res => {
      if (res?.success) { setModelInfo(res); const init = {}; (res.features || []).forEach(f => { init[f] = ''; }); setInputValues(init); }
    }).finally(() => setInfoLoading(false));
  }, [selectedModel]);

  const handlePredict = async () => {
    if (!selectedModel) return;
    setLoading(true); setError(null); setMLPredictionResult(null);
    const r = await predictML(selectedModel, inputValues);
    r?.success ? setMLPredictionResult(r) : setError(r || { message: 'Prediction failed' });
    setLoading(false);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={card}><Label>SELECT MODEL</Label>
        <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)} style={{ ...select, marginTop: 8 }}>
          <option value="">— Choose a trained model —</option>
          {activeModels.map(m => <option key={m.model_id} value={m.model_id}>{m.task ? `[${m.task}] ` : ''}{m.model_id} ({m.size})</option>)}
        </select>
      </div>
      {modelInfo && (
        <div style={{ ...card, display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center' }}>
          <span style={{ padding: '3px 10px', borderRadius: 4, fontSize: 10, fontFamily: 'var(--font-mono)', background: 'rgba(0,240,255,0.08)', border: '1px solid var(--border)', color: 'var(--cyan)' }}>{modelInfo.task}</span>
          <span style={{ fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>Target: <span style={{ color: 'var(--cyan)' }}>{modelInfo.target}</span></span>
        </div>
      )}
      {infoLoading && <StatusLine text="Loading model info..." />}
      {modelInfo && Object.keys(inputValues).length > 0 && (
        <div style={card}><Label>INPUT FEATURES</Label>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: 8, marginTop: 8 }}>
            {Object.entries(inputValues).map(([key, val]) => {
              const dtype = modelInfo.feature_types?.[key] || '';
              return (
                <div key={key}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                    <span style={{ fontSize: 9, color: 'var(--text-2)', fontFamily: 'var(--font-mono)' }}>{key}</span>
                    {dtype && <span style={{ fontSize: 8, color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>{dtype}</span>}
                  </div>
                  <input value={val} onChange={e => setInputValues(p => ({ ...p, [key]: e.target.value }))} placeholder={dtype.includes('int') ? '0' : dtype.includes('float') ? '0.0' : '…'} style={inputSt} />
                </div>
              );
            })}
          </div>
        </div>
      )}
      {selectedModel && !infoLoading && <Btn onClick={handlePredict} disabled={loading}>{loading ? '◌ PREDICTING...' : '🔮 PREDICT'}</Btn>}
      {error && <ErrorBlock err={error} />}
      {mlPredictionResult?.success && (
        <div style={{ ...card, borderColor: 'rgba(0,240,255,0.35)' }}>
          <Label style={{ color: 'var(--cyan)' }}>✓ PREDICTION RESULT</Label>
          <div style={{ fontSize: 28, color: 'var(--cyan)', fontFamily: 'var(--font-mono)', marginTop: 8 }}>{String(mlPredictionResult.prediction)}</div>
          {mlPredictionResult.confidence != null && (
            <div style={{ marginTop: 10 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                <span style={{ fontSize: 9, color: 'var(--text-2)', fontFamily: 'var(--font-hud)', letterSpacing: 1 }}>CONFIDENCE</span>
                <span style={{ fontSize: 10, color: 'var(--cyan)', fontFamily: 'var(--font-mono)' }}>{(mlPredictionResult.confidence * 100).toFixed(1)}%</span>
              </div>
              <div style={{ height: 8, background: 'rgba(0,240,255,0.08)', borderRadius: 4, overflow: 'hidden' }}>
                <motion.div initial={{ width: 0 }} animate={{ width: `${(mlPredictionResult.confidence * 100).toFixed(0)}%` }} transition={{ duration: 0.8, ease: 'easeOut' }}
                  style={{ height: '100%', background: 'linear-gradient(90deg, var(--cyan), var(--blue))', borderRadius: 4 }} />
              </div>
            </div>
          )}
          {mlPredictionResult.probabilities && (
            <div style={{ marginTop: 10 }}><Label>CLASS PROBABILITIES</Label>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 6 }}>
                {Object.entries(mlPredictionResult.probabilities).map(([cls, prob]) => (
                  <span key={cls} style={{ padding: '3px 10px', borderRadius: 4, fontSize: 10, fontFamily: 'var(--font-mono)', background: 'rgba(0,240,255,0.06)', border: '1px solid var(--border)', color: 'var(--cyan)' }}>{cls}: {(prob * 100).toFixed(1)}%</span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function ModelsTab() {
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

function VisualizeTab() {
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

function ChartRenderer({ data }) {
  if (!data) return null;
  const BAR_COLOR = '#00ffc8';
  const MAX_BAR_W = 260;

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
