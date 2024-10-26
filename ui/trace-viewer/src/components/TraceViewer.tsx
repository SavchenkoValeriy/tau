import React, { useCallback, useState, useEffect } from 'react';
import { Panel, useReactFlow } from '@xyflow/react';
import { BasicBlockData, TraceEvent } from '../types';
import { ValueViewer } from './ValueViewer';

interface TraceViewerProps {
  trace: TraceEvent[];
  blocks: BasicBlockData[];
}

const StateView: React.FC<{ state: TraceEvent['state'], blocks: BasicBlockData[] }> = ({ state, blocks }) => {
  return (
    <div style={{
      fontSize: '0.875rem',
      fontFamily: 'monospace'
    }}>
      {state.map((entry, i) => (
        <div key={i} style={{ marginBottom: '0.5rem' }}>
          <div>Value: <ValueViewer value={entry.value} blocks={blocks} /></div>
          {entry.state.map((s, j) => (
            <div key={j} style={{ paddingLeft: '1rem' }}>
              {s.checker}: state {s.state}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
};

const MemoryView: React.FC<{ memory: TraceEvent['memory'], blocks: Array<{ name: string, code: string[] }> }> = ({ memory, blocks }) => {
  return (
    <div style={{
      fontSize: '0.875rem',
      fontFamily: 'monospace'
    }}>
      {memory.model.map((entry, i) => (
        <div key={i} style={{ marginBottom: '0.5rem' }}>
          <div>Edge: {entry.edge}</div>
          {entry.target && (
            <div style={{ paddingLeft: '1rem' }}>
              Target: {entry.target.map(t => (
                typeof t === 'string' ? t : <ValueViewer value={t} blocks={blocks} />
              )).join(', ')}
            </div>
          )}
          <div style={{ paddingLeft: '1rem' }}>
            Value: <ValueViewer value={entry.value} blocks={blocks} />
          </div>
        </div>
      ))}
    </div>
  );
};


export const TraceViewer: React.FC<{
  trace: TraceEvent[];
  blocks: Array<{ name: string, code: string[] }>
}> = ({ trace, blocks }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const { setCenter, getNode } = useReactFlow();
  const currentEvent = trace[currentIndex];
  const nextEvent = trace[currentIndex + 1];

  const focusOnInstruction = useCallback(() => {
    if (currentEvent) {
      const [blockIdx, instIdx] = currentEvent.operation;
      const blockName = blocks[blockIdx].name;
      const node = getNode(blockName);

      // Remove previous highlights
      document.querySelectorAll('[role="instruction"]').forEach(inst => {
        (inst as HTMLElement).style.backgroundColor = 'transparent';
        (inst as HTMLElement).style.boxShadow = 'none';
      });

      if (node) {
        const blockElement = document.querySelector(`[data-id="${blockName}"]`);
        if (blockElement) {
          const instructions = blockElement.querySelectorAll('[role="instruction"]');
          const instruction = instructions[instIdx];

          if (instruction) {
            // Highlight current instruction
            (instruction as HTMLElement).style.backgroundColor = '#e6efff';
            (instruction as HTMLElement).style.boxShadow = '0 0 0 2px #2563eb';

            const rect = instruction.getBoundingClientRect();
            const nodeRect = blockElement.getBoundingClientRect();
            const scale = node.width ? node.width / nodeRect.width : 1;

            const x = node.position.x + (rect.left - nodeRect.left) * scale + 200;
            const y = node.position.y + (rect.top - nodeRect.top) * scale;

            setCenter(x, y, { duration: 800, zoom: 1.5 });
          }
        }
      }
    }
  }, [currentEvent, blocks, getNode, setCenter]);

  useEffect(() => {
    focusOnInstruction();
  }, [currentIndex, focusOnInstruction]);

  return (
    <>
      <Panel position="top-right" style={{
        maxHeight: '40vh',
        overflow: 'auto',
        background: 'white',
        border: '1px solid #e5e7eb',
        borderRadius: '8px',
        padding: '16px'
      }}>
        <h3 style={{ margin: '0 0 1rem', fontSize: '1rem', fontWeight: 'bold' }}>Before</h3>
        <h4 style={{ margin: '0.5rem 0', fontSize: '0.875rem' }}>State</h4>
        <StateView state={currentEvent.state} blocks={blocks} />
        <h4 style={{ margin: '0.5rem 0', fontSize: '0.875rem' }}>Memory</h4>
        <MemoryView memory={currentEvent.memory} blocks={blocks} />
      </Panel>

      <Panel position="bottom-right" style={{
        maxHeight: '40vh',
        overflow: 'auto',
        background: 'white',
        border: '1px solid #e5e7eb',
        borderRadius: '8px',
        padding: '16px'
      }}>
        <h3 style={{ margin: '0 0 1rem', fontSize: '1rem', fontWeight: 'bold' }}>After</h3>
        <h4 style={{ margin: '0.5rem 0', fontSize: '0.875rem' }}>State</h4>
        <StateView state={nextEvent.state} blocks={blocks} />
        <h4 style={{ margin: '0.5rem 0', fontSize: '0.875rem' }}>Memory</h4>
        <MemoryView memory={nextEvent.memory} blocks={blocks} />
      </Panel>

      <Panel position="bottom-center" style={{
        background: 'white',
        border: '1px solid #e5e7eb',
        borderRadius: '8px',
        padding: '16px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '8px'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '16px'
        }}>
          <button
            onClick={() => setCurrentIndex(Math.max(0, currentIndex - 2))}
            disabled={currentIndex === 0}
            style={{
              padding: '4px 8px',
              border: '1px solid #e5e7eb',
              borderRadius: '4px',
              background: currentIndex === 0 ? '#f5f5f5' : 'white',
              cursor: currentIndex === 0 ? 'not-allowed' : 'pointer'
            }}
          >
            Previous
          </button>
          <input
            type="range"
            min={0}
            max={Math.floor((trace.length - 1) / 2) * 2}
            step={2}
            value={currentIndex}
            onChange={(e) => setCurrentIndex(parseInt(e.target.value))}
            style={{
              width: '300px',
              margin: '0'
            }}
          />
          <button
            onClick={() => setCurrentIndex(Math.min((trace.length - 1), currentIndex + 2))}
            disabled={currentIndex >= trace.length - 2}
            style={{
              padding: '4px 8px',
              border: '1px solid #e5e7eb',
              borderRadius: '4px',
              background: currentIndex >= trace.length - 2 ? '#f5f5f5' : 'white',
              cursor: currentIndex >= trace.length - 2 ? 'not-allowed' : 'pointer'
            }}
          >
            Next
          </button>
        </div>
        <div style={{
          fontSize: '0.875rem',
          textAlign: 'center'
        }}>
          Step {Math.floor(currentIndex / 2) + 1} of {Math.floor(trace.length / 2)}
        </div>
      </Panel>
    </>
  );
};
