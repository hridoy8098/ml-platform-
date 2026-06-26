import { create } from 'zustand';
import { persist } from 'zustand/middleware';

const useAtlasStore = create(
  persist(
    (set) => ({
      mlDataset: null,
      setMLDataset: (data) => set({ mlDataset: data }),

      mlAnalysis: null,
      setMLAnalysis: (data) => set({ mlAnalysis: data }),

      mlTrainingResult: null,
      setMLTrainingResult: (data) => set({ mlTrainingResult: data }),

      mlEvaluationData: null,
      setMLEvaluationData: (data) => set({ mlEvaluationData: data }),

      mlPredictionResult: null,
      setMLPredictionResult: (data) => set({ mlPredictionResult: data }),

      mlModels: [],
      setMLModels: (models) => set({ mlModels: models }),

      mlPlotData: null,
      setMLPlotData: (data) => set({ mlPlotData: data }),
    }),
    {
      name: 'ml-platform-storage',
      partialize: (state) => ({
        mlModels: state.mlModels,
        mlDataset: state.mlDataset ? {
          ...state.mlDataset,
          preview: undefined,
          columns: undefined,
        } : null,
      }),
    }
  )
);

export default useAtlasStore;
