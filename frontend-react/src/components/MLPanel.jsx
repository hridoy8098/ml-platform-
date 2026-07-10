import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import useAtlasStore from '../store/useAtlasStore';
import { listMLModels } from '../services/api';

import DataTab from './tabs/DataTab';
import AnalyzeTab from './tabs/AnalyzeTab';
import TrainTab from './tabs/TrainTab';
import ClusterTab from './tabs/ClusterTab';
import DimRedTab from './tabs/DimRedTab';
import NeuralTab from './tabs/NeuralTab';
import HPTuneTab from './tabs/HPTuneTab';
import KfoldTab from './tabs/KfoldTab';
import FeatEngTab from './tabs/FeatEngTab';
import EvaluateTab from './tabs/EvaluateTab';
import PredictTab from './tabs/PredictTab';
import ModelsTab from './tabs/ModelsTab';
import VisualizeTab from './tabs/VisualizeTab';
import CVDataTab from './tabs/cv/CVDataTab';
import CVTrainTab from './tabs/cv/CVTrainTab';
import CVPredictTab from './tabs/cv/CVPredictTab';

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
  { id: 'cvdata',    icon: '🗜️', label: 'CV DATA' },
  { id: 'cvtrain',   icon: '🧬', label: 'CV TRAIN' },
  { id: 'cvpredict', icon: '🩻', label: 'CV PREDICT' },
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
            {activeTab === 'cvdata'    && <CVDataTab />}
            {activeTab === 'cvtrain'   && <CVTrainTab />}
            {activeTab === 'cvpredict' && <CVPredictTab />}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}