import React, { useState, useEffect, useCallback, useMemo, useRef, memo } from 'react';
import './index.css';

// --- Types ---
interface Sample {
  name: string;
  status: 'approved' | 'rejected' | 'pending';
}

const API_BASE = 'http://100.119.217.51:8000';
const ITEM_HEIGHT = 44; // Slightly taller for better touch/click targets
const VISIBLE_HEIGHT = window.innerHeight - 300; // Dynamic-ish
const OVERSCAN = 20;

// --- Sub-Components ---

const SampleRow = memo(({
  item,
  isActive,
  index,
  onClick
}: {
  item: Sample;
  isActive: boolean;
  index: number;
  onClick: (idx: number) => void
}) => {
  const statusColor = {
    approved: 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]',
    rejected: 'bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.5)]',
    pending: 'bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.5)]',
  }[item.status];

  return (
    <div
      className={`group absolute left-0 w-full flex items-center px-4 cursor-pointer transition-all duration-150 rounded-lg mx-2 w-[calc(100%-1rem)]
        ${isActive ? 'bg-slate-800/80 shadow-inner' : 'hover:bg-slate-800/40'}`}
      onClick={() => onClick(index)}
      style={{
        top: index * ITEM_HEIGHT,
        height: ITEM_HEIGHT - 4, // Gap between items
        willChange: 'transform',
      }}
    >
      <div className={`w-1.5 h-6 rounded-full mr-3 transition-all duration-300 ${isActive ? 'bg-sky-400 scale-y-100' : 'bg-transparent scale-y-0'}`} />
      <span className={`flex-1 truncate text-sm font-medium transition-colors ${isActive ? 'text-white' : 'text-slate-400 group-hover:text-slate-200'}`}>
        {item.name}
      </span>
      <div className={`w-2 h-2 rounded-full ${statusColor}`} />
    </div>
  );
});

export default function App() {
  const [sampleMap, setSampleMap] = useState<Record<string, Sample>>({});
  const [allNames, setAllNames] = useState<string[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [statusFilter, setStatusFilter] = useState('all');
  const [split, setSplit] = useState<'train' | 'valid'>('train');
  const [scrollTop, setScrollTop] = useState(0);

  const scrollContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setLoading(true);
    fetch(`${API_BASE}/samples?split=${split}`)
      .then(res => res.json())
      .then((data: Sample[]) => {
        const mapping: Record<string, Sample> = {};
        const names: string[] = [];
        data.forEach(s => {
          mapping[s.name] = s;
          names.push(s.name);
        });
        setSampleMap(mapping);
        setAllNames(names);
        setLoading(false);

        const firstPending = data.findIndex(s => s.status === 'pending');
        setCurrentIndex(firstPending !== -1 ? firstPending : 0);

        if (scrollContainerRef.current) {
          scrollContainerRef.current.scrollTop = 0;
        }
      });
  }, [split]);

  const filteredIndices = useMemo(() => {
    if (statusFilter === 'all') return allNames.map((_, i) => i);
    return allNames
      .map((name, i) => (sampleMap[name].status === statusFilter ? i : -1))
      .filter(i => i !== -1);
  }, [sampleMap, allNames, statusFilter]);

  const counts = useMemo(() => {
    const stats = { pending: 0, approved: 0, rejected: 0, total: allNames.length };
    allNames.forEach(name => stats[sampleMap[name].status]++);
    return stats;
  }, [sampleMap, allNames]);

  useEffect(() => {
    if (!scrollContainerRef.current) return;
    const visualIndex = filteredIndices.indexOf(currentIndex);
    if (visualIndex === -1) return;

    const container = scrollContainerRef.current;
    const itemTop = visualIndex * ITEM_HEIGHT;
    const itemBottom = itemTop + ITEM_HEIGHT;

    if (itemTop < container.scrollTop) {
      container.scrollTo({ top: itemTop, behavior: 'auto' });
    } else if (itemBottom > container.scrollTop + VISIBLE_HEIGHT) {
      container.scrollTo({ top: itemBottom - VISIBLE_HEIGHT, behavior: 'auto' });
    }
  }, [currentIndex, filteredIndices]);

  const navigate = useCallback((step: number) => {
    if (filteredIndices.length === 0) return;
    const visualIndex = filteredIndices.indexOf(currentIndex);

    if (visualIndex === -1) {
      if (step > 0) {
        const next = filteredIndices.find(idx => idx > currentIndex) ?? filteredIndices[0];
        setCurrentIndex(next);
      } else {
        const prev = [...filteredIndices].reverse().find(idx => idx < currentIndex) ?? filteredIndices[filteredIndices.length - 1];
        setCurrentIndex(prev);
      }
      return;
    }
    const nextVisualIndex = (visualIndex + step + filteredIndices.length) % filteredIndices.length;
    setCurrentIndex(filteredIndices[nextVisualIndex]);
  }, [currentIndex, filteredIndices]);

  const updateStatus = useCallback((status: 'approved' | 'rejected' | 'pending') => {
    const name = allNames[currentIndex];
    if (!name) return;

    // Optimistically update the status in the sidebar/UI
    setSampleMap(prev => ({
      ...prev,
      [name]: { ...prev[name], status }
    }));

    // Immediately navigate to the next sample to eliminate "dead time"
    navigate(1);

    // Send the review to the backend in the background
    fetch(`${API_BASE}/review`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sample_name: name, status, split }),
    }).catch(err => {
      console.error('Failed to update status on server:', err);
      // Optional: Add a toast notification or revert local state on failure
    });
  }, [currentIndex, allNames, navigate, split]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'a') updateStatus('approved');
      if (e.key === 'd') updateStatus('rejected');
      if (e.key === 's' || e.key === 'ArrowRight') navigate(1);
      if (e.key === 'ArrowLeft') navigate(-1);
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [updateStatus, navigate]);

  const preloadedIndices = useMemo(() => {
    if (filteredIndices.length <= 1) return [];
    const visualIndex = filteredIndices.indexOf(currentIndex);
    if (visualIndex === -1) return [];

    const indices = new Set<number>();
    for (let i = 1; i <= 8; i++) indices.add(filteredIndices[(visualIndex + i) % filteredIndices.length]);
    for (let i = 1; i <= 2; i++) indices.add(filteredIndices[(visualIndex - i + filteredIndices.length) % filteredIndices.length]);
    indices.delete(currentIndex);
    return Array.from(indices);
  }, [currentIndex, filteredIndices]);

  const onScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    setScrollTop(e.currentTarget.scrollTop);
  }, []);

  const startIndex = Math.max(0, Math.floor(scrollTop / ITEM_HEIGHT) - OVERSCAN);
  const endIndex = Math.min(filteredIndices.length, Math.ceil((scrollTop + VISIBLE_HEIGHT) / ITEM_HEIGHT) + OVERSCAN);

  if (loading) return (
    <div className="flex items-center justify-center h-screen bg-slate-950 font-mono text-sky-400">
      <div className="flex flex-col items-center gap-4">
        <div className="w-12 h-12 border-4 border-sky-400/20 border-t-sky-400 rounded-full animate-spin" />
        <span className="animate-pulse">LOADING_DATASET_CORE...</span>
      </div>
    </div>
  );

  const currentSample = sampleMap[allNames[currentIndex]];

  return (
    <div className="flex h-screen overflow-hidden text-slate-200">
      {/* Sidebar */}
      <aside className="w-80 flex flex-col bg-slate-900/50 backdrop-blur-xl border-r border-slate-800/60 shadow-2xl z-20">
        <div className="p-6 border-b border-slate-800/60">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-bold bg-gradient-to-r from-sky-400 to-indigo-400 bg-clip-text text-transparent">
              Vlx²
            </h2>
            <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
              v1.0
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2 p-1 bg-slate-950/50 rounded-lg border border-slate-800/50 mb-4">
            {(['train', 'valid'] as const).map(s => (
              <button
                key={s}
                onClick={() => setSplit(s)}
                className={`py-1.5 rounded-md text-[10px] font-black tracking-widest transition-all ${split === s ? 'bg-sky-500 text-white shadow-lg' : 'text-slate-500 hover:text-slate-300'
                  }`}
              >
                {s.toUpperCase()}
              </button>
            ))}
          </div>

          <div className="flex gap-1 overflow-x-auto pb-1 no-scrollbar">
            {['all', 'pending', 'approved', 'rejected'].map(f => (
              <button
                key={f}
                onClick={() => setStatusFilter(f)}
                className={`whitespace-nowrap px-3 py-1 rounded-full text-[10px] font-bold transition-all border ${statusFilter === f
                  ? 'bg-slate-50 border-white text-slate-950'
                  : 'bg-transparent border-slate-700 text-slate-500 hover:border-slate-500 hover:text-slate-300'
                  }`}
              >
                {f.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        <div
          ref={scrollContainerRef}
          onScroll={onScroll}
          className="flex-1 overflow-y-auto relative no-scrollbar"
        >
          <div style={{ height: filteredIndices.length * ITEM_HEIGHT, width: '100%', position: 'relative' }}>
            {filteredIndices.slice(startIndex, endIndex).map((actualIdx) => {
              const visualIdx = filteredIndices.indexOf(actualIdx);
              return (
                <SampleRow
                  key={allNames[actualIdx]}
                  index={visualIdx}
                  item={sampleMap[allNames[actualIdx]]}
                  isActive={actualIdx === currentIndex}
                  onClick={() => setCurrentIndex(actualIdx)}
                />
              );
            })}
          </div>
        </div>

        <div className="p-4 bg-slate-950/30 border-t border-slate-800/60 text-[10px] grid grid-cols-3 gap-1 text-center font-mono">
          <div className="text-emerald-400 p-1 rounded bg-emerald-500/5 border border-emerald-500/10">OK {counts.approved}</div>
          <div className="text-rose-400 p-1 rounded bg-rose-500/5 border border-rose-500/10">NO {counts.rejected}</div>
          <div className="text-sky-400 p-1 rounded bg-sky-500/5 border border-sky-500/10">TR {counts.total}</div>
        </div>
      </aside>

      {/* Main Viewport */}
      <main className="flex-1 flex flex-col relative z-10">
        {currentSample && (
          <>
            <header className="px-8 py-5 border-b border-slate-800/60 flex items-center justify-between bg-slate-950/20 backdrop-blur-sm">
              <div>
                <h1 className="text-lg font-bold text-white mb-0.5">{currentSample.name}</h1>
                <div className="flex items-center gap-3 text-[10px] font-mono text-slate-500 uppercase tracking-tighter">
                  <span>Index: {currentIndex} / {allNames.length}</span>
                  <span className="w-1 h-1 rounded-full bg-slate-700" />
                  <span>Split: {split}</span>
                </div>
              </div>
              <div className={`px-4 py-1.5 rounded-full text-[10px] font-black tracking-widest shadow-lg ${{
                approved: 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30',
                rejected: 'bg-rose-500/10 text-rose-400 border border-rose-500/30',
                pending: 'bg-sky-500/10 text-sky-400 border border-sky-500/30',
              }[currentSample.status]
                }`}>
                {currentSample.status.toUpperCase()}
              </div>
            </header>

            <div className="flex-1 p-8 flex gap-8 items-center justify-center overflow-auto no-scrollbar scroll-smooth">
              {/* Original Image Card */}
              <div className="relative group max-w-[45%]">
                <div className="absolute -inset-0.5 bg-gradient-to-br from-sky-500 to-indigo-500 rounded-2xl blur opacity-25 group-hover:opacity-40 transition duration-1000 group-hover:duration-200"></div>
                <div className="relative bg-slate-900 rounded-2xl overflow-hidden shadow-2xl border border-slate-800">
                  <div className="absolute top-3 left-3 px-2 py-1 bg-black/60 backdrop-blur-md rounded text-[8px] font-bold text-white z-10 border border-white/10 uppercase tracking-widest">Original Image</div>
                  <img
                    key={`i-${currentSample.name}`}
                    src={`${API_BASE}/image/${currentSample.name}?split=${split}`}
                    alt="Original"
                    className="w-full object-contain"
                  />
                </div>
              </div>

              {/* Mask Image Card */}
              <div className="relative group max-w-[45%]">
                <div className="absolute -inset-0.5 bg-gradient-to-br from-indigo-500 to-purple-500 rounded-2xl blur opacity-25 group-hover:opacity-40 transition duration-1000 group-hover:duration-200"></div>
                <div className="relative bg-slate-900 rounded-2xl overflow-hidden shadow-2xl border border-slate-800">
                  <div className="absolute top-3 left-3 px-2 py-1 bg-black/60 backdrop-blur-md rounded text-[8px] font-bold text-white z-10 border border-white/10 uppercase tracking-widest">Segmentation Mask</div>
                  <img
                    key={`m-${currentSample.name}`}
                    src={`${API_BASE}/mask/${currentSample.name}?split=${split}`}
                    alt="Mask"
                    className="w-full object-contain mix-blend-screen"
                  />
                </div>
              </div>
            </div>

            {/* Bottom Navigation */}
            <div className="absolute bottom-10 left-1/2 -translate-x-1/2 flex items-center gap-4 bg-slate-900/80 backdrop-blur-2xl border border-slate-700/50 px-6 py-4 rounded-3xl shadow-[0_20px_50px_rgba(0,0,0,0.5)] z-30">
              <button
                onClick={() => navigate(-1)}
                className="w-10 h-10 flex items-center justify-center rounded-full hover:bg-slate-800 text-slate-400 hover:text-white transition-all border border-transparent hover:border-slate-600"
              >
                ←
              </button>

              <button
                onClick={() => updateStatus('rejected')}
                className="px-6 py-2.5 bg-rose-500 hover:bg-rose-400 text-white font-black text-[11px] uppercase tracking-widest rounded-full shadow-[0_0_20px_rgba(244,63,94,0.3)] transition-all active:scale-95"
              >
                Reject <span className="ml-2 text-[8px] opacity-70 border border-white/20 px-1 rounded">D</span>
              </button>

              <button
                onClick={() => updateStatus('approved')}
                className="px-6 py-2.5 bg-emerald-500 hover:bg-emerald-400 text-white font-black text-[11px] uppercase tracking-widest rounded-full shadow-[0_0_20px_rgba(16,185,129,0.3)] transition-all active:scale-95"
              >
                Approve <span className="ml-2 text-[8px] opacity-70 border border-white/20 px-1 rounded">A</span>
              </button>

              <button
                onClick={() => navigate(1)}
                className="w-10 h-10 flex items-center justify-center rounded-full hover:bg-slate-800 text-slate-400 hover:text-white transition-all border border-transparent hover:border-slate-600"
              >
                →
              </button>
            </div>
          </>
        )}
      </main>

      {/* Image Preloader */}
      <div style={{ display: 'none', visibility: 'hidden', height: 0, width: 0, overflow: 'hidden' }} aria-hidden="true">
        {preloadedIndices.map(idx => (
          <React.Fragment key={allNames[idx]}>
            <img src={`${API_BASE}/image/${allNames[idx]}?split=${split}`} alt="" />
            <img src={`${API_BASE}/mask/${allNames[idx]}?split=${split}`} alt="" />
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}
