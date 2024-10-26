import React, { useEffect } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  useReactFlow,
  useNodesState,
  useEdgesState,
  Node as FlowNode,
  Edge as FlowEdge,
  useNodesInitialized,
} from '@xyflow/react';
import { initializeGraph } from '../utils/initializeGraph';
import { getLayoutedElements } from '../utils/layout';
import { BasicBlock, BasicBlockData } from './BasicBlock';
import { BasicBlockNode, CFGData } from '../types';
import '@xyflow/react/dist/style.css';

type Node = FlowNode<BasicBlockData>
type Edge = FlowEdge<{}>

const nodeTypes = {
  basicBlock: BasicBlock,
};

const options = {
  includeHiddenNodes: true,
};

const CFGViewer: React.FC<{ data: CFGData }> = ({ data }) => {
  const { nodes: initialNodes, edges: initialEdges } = initializeGraph(data);
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>(initialEdges);
  const { getNodes, getEdges } = useReactFlow<BasicBlockNode>();
  const nodesInitialized = useNodesInitialized(options);

  useEffect(() => {
    if (nodesInitialized) {
      const { nodes: layoutedNodes, edges: layoutedEdges } = getLayoutedElements(
        getNodes(),
        getEdges()
      );
      setNodes([...layoutedNodes]);
      setEdges([...layoutedEdges]);
    }
  }, [nodesInitialized]);

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
