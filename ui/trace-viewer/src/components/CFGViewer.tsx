import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  Controls,
  useReactFlow,
  useNodesState,
  useEdgesState,
  Panel,
  addEdge,
  Node as FlowNode,
  Edge as FlowEdge,
  NodeTypes,
  useNodesInitialized
} from '@xyflow/react';
import { useGraphLayout } from '../hooks/useGraphLayout';
import { useGraphInitialization, CFGData } from '../hooks/useGraphInitialization';
import { BasicBlock, BasicBlockData } from './BasicBlock';
import '@xyflow/react/dist/style.css';

type Node = FlowNode<BasicBlockData>
type Edge = FlowEdge<{}>

const nodeTypes = {
  basicBlock: BasicBlock,
};

const options = {
  includeHiddenNodes: false,
};

const CFGViewer: React.FC<{ data: CFGData }> = ({ data }) => {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const nodesInitialized = useNodesInitialized(options);
  const [isLayoutApplied, setIsLayoutApplied] = useState(false);
  const { getLayoutedElements } = useGraphLayout();
  const { initializeGraph } = useGraphInitialization();

  useEffect(() => {
    const { nodes: initialNodes, edges: initialEdges } = initializeGraph(data);
    setNodes(initialNodes);
    setEdges(initialEdges);
    setIsLayoutApplied(false);
  }, [data, initializeGraph, setNodes, setEdges]);

  useEffect(() => {
    if (!isLayoutApplied && nodes.length > 0 && edges.length > 0 && nodesInitialized) {
      const { nodes: layoutedNodes, edges: layoutedEdges } = getLayoutedElements(
        nodes,
        edges
      );
      setNodes([...layoutedNodes]);
      setEdges([...layoutedEdges]);
      setIsLayoutApplied(true);
    }
  }, [nodes, edges, isLayoutApplied, getLayoutedElements, setNodes, setEdges]);

  if (nodes.length === 0) {
    return <div className="flex items-center justify-center h-64">Loading CFG...</div>;
  }

  return (
    <div className="h-screen w-full" style={{ height: 800 }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        fitView
        minZoom={0.1}
        nodesDraggable={true}
        nodesConnectable={false}
      >
        <Background color="#f0f0f0" gap={16} />
        <Controls />
      </ReactFlow>
    </div>
  );
};

export default CFGViewer;
