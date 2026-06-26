import { useState, useEffect, useRef, useCallback } from 'react';
import { getMLJobStatus } from '../services/api';

export default function useJobPolling({ jobId, interval = 800, onDone, onError }) {
  const [jobInfo, setJobInfo] = useState(null);
  const pollRef = useRef(null);
  const onDoneRef = useRef(onDone);
  const onErrorRef = useRef(onError);
  onDoneRef.current = onDone;
  onErrorRef.current = onError;

  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    const poll = async () => {
      try {
        const s = await getMLJobStatus(jobId);
        if (!s?.success || cancelled) return;
        setJobInfo(s);
        if (s.status === 'done') {
          if (s.result?.success) {
            onDoneRef.current?.(s.result);
          } else {
            onErrorRef.current?.(s.result || { message: 'Job completed with errors' });
          }
        } else if (s.status === 'error') {
          onErrorRef.current?.(s.result || { message: 'Job failed' });
        }
      } catch {
        if (!cancelled) onErrorRef.current?.({ message: 'Job polling failed' });
      }
    };
    poll();
    pollRef.current = setInterval(poll, interval);
    return () => { cancelled = true; clearInterval(pollRef.current); };
  }, [jobId, interval]);

  return jobInfo;
}
