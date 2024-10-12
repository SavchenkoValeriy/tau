import React, { useEffect, useRef, useState } from 'react';
import { Handle, Position, NodeProps } from 'react-flow-renderer';

interface BasicBlockProps extends NodeProps {
  data: {
    name: string;
    code: string[];
    edges: number[];
  };
  updateNodeDimensions: (id: string, width: number, height: number) => void;
}

const Instruction: React.FC<{ instruction: string }> = ({ instruction }) => {
  const [mainPart, type] = instruction.split(' : ');
  const parts = mainPart.split(/(%\w+)/g);
  const [showHint, setShowHint] = useState(false);

  return (
    <div 
      style={{ 
        whiteSpace: 'nowrap', 
        overflow: 'visible',
        position: 'relative',
        marginBottom: '2px'
      }}
      onMouseEnter={() => setShowHint(true)}
      onMouseLeave={() => setShowHint(false)}
    >
      {parts.map((part, index) => (
        part.startsWith('%') ? 
          <strong key={index}>{part}</strong> : 
          <span key={index}>{part}</span>
      ))}
      {type && showHint && (
        <span style={{
          position: 'absolute',
          left: '100%',
          top: '-2px',
          backgroundColor: 'black',
          color: 'white',
          padding: '2px 5px',
          borderRadius: '3px',
          whiteSpace: 'nowrap',
          zIndex: 1000,
          pointerEvents: 'none',
        }}>
          : {type}
        </span>
      )}
    </div>
  );
};

const BasicBlock: React.FC<BasicBlockProps> = ({ data, id, updateNodeDimensions }) => {
  const blockRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (blockRef.current) {
      const resizeObserver = new ResizeObserver((entries) => {
        for (let entry of entries) {
          const { width, height } = entry.contentRect;
          updateNodeDimensions(id, width, height);
        }
      });

      resizeObserver.observe(blockRef.current);

      return () => {
        resizeObserver.disconnect();
      };
    }
  }, [id, updateNodeDimensions]);


  return (
    <div ref={blockRef} style={{ padding: '10px', fontFamily: 'monospace', fontSize: '12px', minWidth: '200px' }}>
      <Handle type="target" position={Position.Top} />
      <strong>{data.name}</strong>
      {data.code.map((line: string, i: number) => (
        <Instruction key={i} instruction={line} />
      ))}
    {data.edges.map((_, index) => (
        <Handle
          key={index}
          type="source"
          position={Position.Bottom}
          id={`${id}-${index}`}
          style={{ left: `${(index + 1) * 100 / (data.edges.length + 1)}%` }}
        />
      ))}
    </div>
  );
};

export default BasicBlock;
