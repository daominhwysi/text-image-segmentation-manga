import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import './index.css';

interface Sample {
  name: string;
  status: 'approved' | 'rejected' | 'pending';
}

const API_BASE = 'http://localhost:8000';
const ITEM_HEIGHT = 40;
const VISIBLE_HEIGHT = 600; // Should match sidebar height roughly

function App() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [statusFilter, setStatusFilter] = useState<'all' | 'pending' | 'approved' | 'rejected'>('all');
  const [scrollTop, setScrollTop] = useState(0);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  const fetchSamples = async () => {
    try {
      const response = await fetch(`${API_BASE}/samples`);
      const data = await response.json();
      setSamples(data);
      setLoading(false);

      // Auto-select first pending if exists
      const firstPending = data.findIndex((s: any) => s.status === 'pending');
      if (firstPending !== -1) {
        setCurrentIndex(firstPending);
      }
    } catch (error) {
      console.error('Failed to fetch samples:', error);
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSamples();
  }, []);

  // Filter samples based on selection
  const filteredSamples = useMemo(() => {
    if (statusFilter === 'all') return samples.map((s, i) => ({ ...s, originalIndex: i }));
    return samples
      .map((s, i) => ({ ...s, originalIndex: i }))
      .filter(s => s.status === statusFilter);
  }, [samples, statusFilter]);

  // Total counts for the header
  const counts = useMemo(() => {
    return {
      pending: samples.filter(s => s.status === 'pending').length,
      approved: samples.filter(s => s.status === 'approved').length,
      rejected: samples.filter(s => s.status === 'rejected').length,
      total: samples.length
    };
  }, [samples]);

  const updateStatus = async (status: 'approved' | 'rejected' | 'pending') => {
    if (samples.length === 0) return;
    const sample = samples[currentIndex];

    try {
      await fetch(`${API_BASE}/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sample_name: sample.name, status }),
      });

      const newSamples = [...samples];
      newSamples[currentIndex].status = status;
      setSamples(newSamples);

      // Move to next pending
      if (status !== 'pending') {
        goToNext();
      }
    } catch (error) {
      console.error('Failed to update status:', error);
    }
  };

  const goToNext = () => {
    setCurrentIndex((prev) => (prev + 1) % samples.length);
  };

  const goToPrev = () => {
    setCurrentIndex((prev) => (prev - 1 + samples.length) % samples.length);
  };

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.key === 'a') updateStatus('approved');
    if (e.key === 'r') updateStatus('rejected');
    if (e.key === 's' || e.key === 'ArrowRight') goToNext();
    if (e.key === 'ArrowLeft') goToPrev();
  }, [samples, currentIndex]);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  // Virtual List logic
  const onScroll = (e: React.UIEvent<HTMLDivElement>) => {
    setScrollTop(e.currentTarget.scrollTop);
  };

  const startIndex = Math.max(0, Math.floor(scrollTop / ITEM_HEIGHT) - 5);
  const endIndex = Math.min(filteredSamples.length, Math.ceil((scrollTop + VISIBLE_HEIGHT) / ITEM_HEIGHT) + 5);

  const visibleItems = filteredSamples.slice(startIndex, endIndex);

  // Auto-scroll sidebar when currentIndex changes
  useEffect(() => {
    if (scrollContainerRef.current) {
      // Find the index of current sample in the filtered list
      const filteredIndex = filteredSamples.findIndex(s => s.originalIndex === currentIndex);
      if (filteredIndex !== -1) {
        const itemTop = filteredIndex * ITEM_HEIGHT;
        const itemBottom = itemTop + ITEM_HEIGHT;
        const containerScrollTop = scrollContainerRef.current.scrollTop;
        const containerBottom = containerScrollTop + VISIBLE_HEIGHT;

        if (itemTop < containerScrollTop) {
          scrollContainerRef.current.scrollTop = itemTop;
        } else if (itemBottom > containerBottom) {
          scrollContainerRef.current.scrollTop = itemBottom - VISIBLE_HEIGHT;
        }
      }
    }
  }, [currentIndex, filteredSamples]);

  if (loading) return <div className="loading">Loading samples...</div>;
  if (samples.length === 0) return <div className="no-data">No samples found in {API_BASE}</div>;

  const currentSample = samples[currentIndex];

  return (
    <div className="app-container">
      <aside className="sidebar">
        <header className="sidebar-header">
          <h2>Samples ({counts.pending} pending)</h2>
          <div className="filter-tabs">
            <button className={statusFilter === 'all' ? 'active' : ''} onClick={() => setStatusFilter('all')}>All</button>
            <button className={statusFilter === 'pending' ? 'active' : ''} onClick={() => setStatusFilter('pending')}>Pending</button>
            <button className={statusFilter === 'approved' ? 'active' : ''} onClick={() => setStatusFilter('approved')}>OK</button>
            <button className={statusFilter === 'rejected' ? 'active' : ''} onClick={() => setStatusFilter('rejected')}>NO</button>
          </div>
        </header>

        <div
          className="sample-list"
          ref={scrollContainerRef}
          onScroll={onScroll}
          style={{ height: VISIBLE_HEIGHT, overflowY: 'auto', position: 'relative' }}
        >
          <div style={{ height: filteredSamples.length * ITEM_HEIGHT, position: 'relative' }}>
            {visibleItems.map((item, index) => {
              const actualIndex = startIndex + index;
              return (
                <div
                  key={item.name}
                  className={`sample-item ${item.originalIndex === currentIndex ? 'active' : ''}`}
                  onClick={() => setCurrentIndex(item.originalIndex)}
                  style={{
                    position: 'absolute',
                    top: actualIndex * ITEM_HEIGHT,
                    height: ITEM_HEIGHT,
                    width: '100%',
                    display: 'flex',
                    alignItems: 'center',
                    padding: '0 1rem',
                    boxSizing: 'border-box'
                  }}
                >
                  <span className="sample-name" title={item.name}>{item.name}</span>
                  <span className={`status-dot status-${item.status}`}></span>
                </div>
              );
            })}
          </div>
        </div>

        <div className="stats-footer">
          Total: {counts.total} | OK: {counts.approved} | NO: {counts.rejected}
        </div>
      </aside>

      <main className="main-content">
        <header className="viewer-header">
          <div className="header-info">
            <h1>{currentSample.name}</h1>
            <span className="index-indicator">Sample {currentIndex + 1} of {samples.length}</span>
          </div>
          <div className="status-badge" style={{ color: `var(--status-${currentSample.status})` }}>
            {currentSample.status.toUpperCase()}
          </div>
        </header>

        <div className="viewer-container">
          <div className="image-card">
            <img src={`${API_BASE}/image/${currentSample.name}`} alt="Original" />
            <div className="image-label">Original Image</div>
          </div>
          <div className="image-card">
            <img src={`${API_BASE}/mask/${currentSample.name}`} alt="Mask" />
            <div className="image-label">Segmentation Mask</div>
          </div>
        </div>

        <div className="controls">
          <button className="btn btn-skip" onClick={goToPrev}>Prev [←]</button>
          <button className="btn btn-reject" onClick={() => updateStatus('rejected')}>Reject [R]</button>
          <button className="btn btn-approve" onClick={() => updateStatus('approved')}>Approve [A]</button>
          <button className="btn btn-skip" onClick={goToNext}>Next/Skip [S/→]</button>
        </div>

        <div className="shortcuts-hint">
          Shortcuts: A (Approve), R (Reject), S/Right (Skip), Left (Prev)
        </div>
      </main>
    </div>
  );
}

export default App;

