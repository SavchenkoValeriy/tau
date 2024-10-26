import React, { useState } from 'react';
import { Handle, Position } from '@xyflow/react';
import { BasicBlockProps } from '../types';

export interface BasicBlockData extends Record<string, unknown> {
  name: string;
  code: string[];
}

const Instruction: React.FC<{ instruction: string }> = ({ instruction }) => {
  const [mainPart, type] = instruction.split(' : ');
  const parts = mainPart.split(/(%\w+)/g);
  const [showHint, setShowHint] = useState(false);

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'flex-start',
        gap: '4px',
        position: 'relative',
        padding: '4px',
        borderRadius: '4px',
        cursor: 'default',
        transition: 'background-color 0.2s',
        backgroundColor: showHint ? '#f5f5f5' : 'transparent',
        whiteSpace: 'nowrap'
      }}
      onMouseEnter={() => setShowHint(true)}
      onMouseLeave={() => setShowHint(false)}
    >
      {parts.map((part, index) => (
        part.startsWith('%') ? (
          <code
            key={index}
            style={{
              fontWeight: 600,
              color: '#2563eb',
              fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace'
            }}
          >
            {part}
          </code>
        ) : (
          <span key={index}>{part}</span>
        )
      ))}
      {type && showHint && (
        <div style={{
          position: 'absolute',
          left: '100%',
          top: '0',
          marginLeft: '8px',
          backgroundColor: '#000',
          color: '#fff',
          fontSize: '0.75rem',
          padding: '2px 8px',
          borderRadius: '4px',
          whiteSpace: 'nowrap',
          zIndex: 100
        }}>
          : {type}
        </div>
      )}
    </div>
  );
};

export function BasicBlock(props: BasicBlockProps) {
  return (
    <div style={{
      backgroundColor: '#fff',
      border: '1px solid #e5e7eb',
      borderRadius: '0.5rem',
      padding: '1rem',
      boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
      width: 'fit-content'
    }}>
      <Handle
        type="target"
        position={Position.Top}
        style={{
          background: '#fff',
          border: '1px solid #e5e7eb',
          width: '8px',
          height: '8px'
        }}
      />
      <div style={{
        fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
        fontSize: '0.875rem',
      }}>
        <div style={{
          fontWeight: 700,
          marginBottom: '0.5rem',
          color: '#374151'
        }}>
          {props.data.name}
        </div>
        {props.data.code.map((line: string, i: number) => (
          <Instruction key={i} instruction={line} />
        ))}
      </div>
      <Handle
        type="source"
        position={Position.Bottom}
        style={{
          background: '#fff',
          border: '1px solid #e5e7eb',
          width: '8px',
          height: '8px'
        }}
      />
    </div>
  );
}
