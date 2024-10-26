import React, { useCallback, useState, useEffect } from 'react';
import { Panel, useReactFlow } from '@xyflow/react';
import { BasicBlockData, MemoryEntry, ModelEdge, ModelEntry, StateEntry, TraceEvent, Value } from '../types';
import { ValueViewer } from './ValueViewer';

interface TraceViewerProps {
  trace: TraceEvent[];
  blocks: BasicBlockData[];
}

const ValuesView: React.FC<{ values: Iterable<Value>, blocks: BasicBlockData[] }> = ({ values, blocks }) => {
  return (<>
    {'{'}
    {Array.from(values).map((value, i, arr) => (
      <React.Fragment key={value.toString()}>
        <ValueViewer value={value} blocks={blocks} />
        {i < arr.length - 1 && ', '}
      </React.Fragment>
    ))}
    {'}'}
  </>);
};

const StateView: React.FC<{
  state: StateEntry[],
  blocks: BasicBlockData[]
}> = ({ state, blocks }) => {
  // First, organize states by checker and state id
  const groupedStates = state.reduce((acc, entry) => {
    entry.state.forEach(s => {
      if (!acc[s.checker]) {
        acc[s.checker] = {};
      }
      if (!acc[s.checker][s.state]) {
        acc[s.checker][s.state] = new Set<[number, number]>();
      }
      acc[s.checker][s.state].add(entry.value);
    });
    return acc;
  }, {} as Record<string, Record<number, Set<Value>>>);

  return (
    <div style={{
      fontSize: '0.875rem',
      fontFamily: 'monospace'
    }}>
      {Object.entries(groupedStates).map(([checker, states]) => (
        <div key={checker} style={{ marginBottom: '1rem' }}>
          <div style={{
            fontWeight: 'bold',
            borderBottom: '1px solid #e5e7eb',
            paddingBottom: '4px',
            marginBottom: '4px'
          }}>
            {checker}:
          </div>
          {Object.entries(states).map(([stateId, values]) => (
            <div key={stateId} style={{
              paddingLeft: '1rem',
              marginBottom: '4px'
            }}>
              <ValuesView values={values} blocks={blocks} /> → {stateId}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
};

function ModelEdgeString(edge: ModelEdge): string {
  switch (edge.kind) {
    case "PointsTo":
      return ' -> ';

    case "Field":
      return `.${edge.name} -> `;

    case "Element":
      return `[${edge.index}] -> `;
  }
}

const ModelEdgeView: React.FC<{ model: ModelEntry, blocks: BasicBlockData[] }> = ({ model, blocks }) => {
  return (
    <div>
      <ValueViewer value={model.value} blocks={blocks} />
      {ModelEdgeString(model.edge)}
      <ValuesView values={model.target} blocks={blocks} />
    </div>
  )
};

const MemoryView: React.FC<{ memory: MemoryEntry, blocks: BasicBlockData[] }> = ({ memory, blocks }) => {
  return (
    <>
      <h5 style={{ margin: '0.5rem 0', fontSize: '0.875rem' }}>Model</h5>
      <div style={{
        fontSize: '0.875rem',
        fontFamily: 'monospace'
      }}>
        {memory.model.map((entry, i) => (
          <div key={i} style={{ marginBottom: '0.5rem' }}>
            <ModelEdgeView model={entry} blocks={blocks} />
          </div>
        ))}
      </div>
      <h5 style={{ margin: '0.5rem 0', fontSize: '0.875rem' }}>Canonicals</h5>
      <div style={{
        fontSize: '0.875rem',
        fontFamily: 'monospace'
      }}>
        {memory.canonicals.map((entry, i) => (
          <div key={i} style={{ marginBottom: '0.5rem' }}>
            <ValueViewer value={entry.value} blocks={blocks} />
            {' = '}
            <ValuesView values={entry.canonicals} blocks={blocks} />
          </div>
        ))}
      </div>
    </>
  );
};

export const TraceViewer: React.FC<TraceViewerProps> = ({ trace, blocks }) => {
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
