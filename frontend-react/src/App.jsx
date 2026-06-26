import { useEffect, useState, useRef } from 'react';
import MLPanel from './components/MLPanel';

function Particles() {
  const [dots] = useState(() =>
    Array.from({ length: 25 }, (_, i) => ({
      id: i,
      left: Math.random() * 100 + '%',
      delay: Math.random() * 8 + 's',
      duration: 10 + Math.random() * 12 + 's',
      size: 1 + Math.random() * 2 + 'px',
    }))
  );
  return (
    <div style={{ position: 'fixed', inset: 0, pointerEvents: 'none', zIndex: 0, overflow: 'hidden' }}>
      {dots.map(d => (
        <div key={d.id}
          style={{
            position: 'absolute', left: d.left, bottom: '-5px',
            width: d.size, height: d.size,
            borderRadius: '50%', background: 'var(--cyan)',
            boxShadow: '0 0 6px rgba(0,240,255,0.4)',
            animation: `floatUp ${d.duration} linear ${d.delay} infinite`,
            opacity: 0,
          }}
        />
      ))}
      <style>{`
        @keyframes floatUp {
          0%   { transform: translateY(0) scale(0); opacity: 0; }
          10%  { opacity: 0.8; }
          90%  { opacity: 0.6; }
          100% { transform: translateY(-100vh) scale(1); opacity: 0; }
        }
      `}</style>
    </div>
  );
}

function ScanLine() {
  return (
    <div style={{
      position: 'fixed', left: '5%', width: '90%', height: '2px',
      background: 'linear-gradient(90deg, transparent, rgba(0,240,255,0.08), transparent)',
      pointerEvents: 'none', zIndex: 1,
      animation: 'scanLine 4s linear infinite',
    }} />
  );
}

export default function App() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', position: 'relative' }}>
      <Particles />
      <ScanLine />

      {/* Top Bar */}
      <header style={{
        height: 52, display: 'flex', alignItems: 'center', padding: '0 20px',
        background: 'rgba(5,5,15,0.8)', borderBottom: '1px solid var(--border)',
        backdropFilter: 'blur(16px)', zIndex: 10, position: 'relative',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginRight: 28 }}>
          <svg viewBox="0 0 36 36" width="28" height="28" fill="none">
            <rect x="2" y="2" width="32" height="32" rx="4" stroke="#00f0ff" strokeWidth="1.5" opacity="0.5" />
            <rect x="6" y="6" width="24" height="24" rx="2" stroke="#00f0ff" strokeWidth="1" opacity="0.25" />
            <path d="M10 24 L14 14 L18 20 L22 12 L26 22" stroke="#00f0ff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            <circle cx="22" cy="12" r="2" fill="#00f0ff" opacity="0.7" />
            <circle cx="10" cy="24" r="1.5" fill="#ff00ff" opacity="0.5" />
          </svg>
          <span style={{
            fontFamily: 'var(--font-hud)', fontSize: 13, fontWeight: 700,
            color: 'var(--text-1)', letterSpacing: 3, textTransform: 'uppercase',
          }}>
            ML Platform
          </span>
        </div>

        {/* Status bar */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 14, marginLeft: 'auto',
          fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-3)',
        }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <span style={{
              width: 6, height: 6, borderRadius: '50%',
              background: 'var(--green)', boxShadow: '0 0 8px var(--green)',
              animation: 'pulse 2s ease infinite',
            }} />
            ONLINE
          </span>
          <HudClock />
          <span style={{ color: 'rgba(0,240,255,0.3)', fontSize: 9 }}>v1.0</span>
        </div>
      </header>

      {/* Bottom border glow */}
      <div style={{
        position: 'absolute', top: 51, left: '10%', width: '80%', height: 1,
        background: 'linear-gradient(90deg, transparent, var(--cyan), var(--magenta), transparent)',
        opacity: 0.3, zIndex: 11,
      }} />

      {/* Main content */}
      <main style={{ flex: 1, overflow: 'hidden', padding: 12 }}>
        <MLPanel />
      </main>
    </div>
  );
}

function HudClock() {
  const [time, setTime] = useState(new Date());
  useEffect(() => {
    const iv = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(iv);
  }, []);
  return <span suppressHydrationWarning>{time.toLocaleTimeString('en', { hour12: false })}</span>;
}
