import React from 'react';
import { Value } from  '../types';

interface ValueViewerProps {
  value: Value;
  blocks: Array<{name: string, code: string[]}>;
}

export const ValueViewer: React.FC<ValueViewerProps> = ({ value, blocks }) => {
  if (typeof value === 'string') {
    return <span>{value}</span>;
  }

  const [blockIdx, instIdx] = value;
  const instruction = blocks[blockIdx].code[instIdx];
  const parts = instruction.split(' = ');
  return <span>{parts[0]}</span>;
};
