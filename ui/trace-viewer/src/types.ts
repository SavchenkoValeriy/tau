import { Node as FlowNode, NodeProps as FlowNodeProps } from '@xyflow/react';

export interface CFG {
  func: FunctionCFG;
}

export interface FunctionCFG {
  name: string;
  blocks: Block[];
}

export interface Block {
  name: string;
  code: string[];
  edges: number[];
}

export interface Value {
  value: [number, number] | string;
}

export interface StateEntry {
  state: {
    checker: string;
    state: number;
  }[];
  value: [number, number];
}

export interface ModelEntry {
  edge: 'PointsTo' | 'Field';
  target?: string[];
  value: [number, number] | string;
  canonicals?: string[];
}

export interface Memory {
  canonicals: string[];
  model: ModelEntry[];
}

export interface TraceEvent {
  kind: 'before' | 'after';
  memory: Memory;
  operation: [number, number];
  state: StateEntry[];
}

export interface CFGData {
  func: {
    name: string;
    blocks: {
      name: string;
      code: string[];
      edges: number[];
    }[];
  };
  trace: TraceEvent[];
}

export interface BasicBlockData extends Record<string, unknown> {
  name: string;
  code: string[];
}

export type BasicBlockNode = FlowNode<BasicBlockData>
export type BasicBlockProps = FlowNodeProps<BasicBlockNode>
