import React, { useEffect, useRef, useState } from 'react';
import { Handle, Position } from '@xyflow/react';

export interface BasicBlockData extends Record<string, unknown> {
  name: string;
  code: string[];
}

export interface BasicBlockProps {
  id: string;
  data: BasicBlockData;
  updateNodeDimensions: (id: string, width: number, height: number) => void;
}

const Instruction: React.FC<{ instruction: string }> = ({ instruction }) => {
  const [mainPart, type] = instruction.split(' : ');
  const parts = mainPart.split(/(%\w+)/g);
  const [showHint, setShowHint] = useState(false);

  return (
    <div 
      className="flex items-start space-x-1 relative hover:bg-gray-50 p-1 rounded"
      onMouseEnter={() => setShowHint(true)}
      onMouseLeave={() => setShowHint(false)}
    >
      {parts.map((part, index) => (
        part.startsWith('%') ? 
          <code key={index} className="font-semibold text-blue-600">{part}</code> : 
          <span key={index}>{part}</span>
      ))}
      {type && showHint && (
        <div className="absolute left-full top-0 ml-2 bg-black text-white text-xs px-2 py-1 rounded">
          : {type}
        </div>
      )}
    </div>
  );
};

export const BasicBlock: React.FC<BasicBlockProps> = ({ 
  id, 
  data, 
  updateNodeDimensions 
}) => {
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
      return () => resizeObserver.disconnect();
    }
  }, [id, updateNodeDimensions]);

  return (
    <div
      ref={blockRef}
      className="bg-white border border-gray-200 rounded-lg shadow-sm p-4"
    >
      <Handle type="target" position={Position.Top} />
      <div className="font-mono text-sm">
        <div className="font-bold mb-2 text-gray-700">{data.name}</div>
        {data.code.map((line: string, i: number) => (
          <Instruction key={i} instruction={line} />
        ))}
      </div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
};
