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

export type Operation = [number, number]

export type Value = Operation | string

export interface Memory {
  canonicals: string[];
  model: ModelEntry[];
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
}

export interface BasicBlockData extends Record<string, unknown> {
  name: string;
  code: string[];
}

export type BasicBlockNode = FlowNode<BasicBlockData>
export type BasicBlockProps = FlowNodeProps<BasicBlockNode>

export interface StateEntry {
  state: {
    checker: string;
    state: number;
  }[];
  value: Value;
}

export interface CanonicalMapping {
  value: Value;
  canonicals: Value[];
}

export interface PointsToEdge {
  kind: 'PointsTo';
}

export interface FieldEdge {
  kind: 'Field';
  name: string;
}

export interface ElementEdge {
  kind: 'Element';
  index: number;
}

export type ModelEdge = PointsToEdge | FieldEdge | ElementEdge;

export interface ModelEntry {
  edge: ModelEdge;
  value: Value;
  target: Value[];
}

export interface MemoryEntry {
  canonicals: CanonicalMapping[];
  model: ModelEntry[];
}

export interface TraceEvent {
  kind: 'before' | 'after';
  memory: MemoryEntry;
  operation: Operation;
  state: StateEntry[];
}

export interface CFGWithTrace extends CFGData {
  trace: TraceEvent[];
}
