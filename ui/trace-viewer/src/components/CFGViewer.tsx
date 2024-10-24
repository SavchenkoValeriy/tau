import React, { useCallback, useEffect, useMemo, useState } from 'react';
import ReactFlow, {
  Node,
  Edge,
  ConnectionLineType,
  MarkerType,
  Background,
  ReactFlowInstance,
  useNodesState,
  useEdgesState,
  NodeProps
} from 'react-flow-renderer';
import BasicBlock from './BasicBlock';
import { useGraphLayout } from '../hooks/useGraphLayout';
import { useGraphInitialization, CFGData } from '../hooks/useGraphInitialization';

const CFGViewer: React.FC<{ data: CFGData }> = ({ data }) => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [isLayoutApplied, setIsLayoutApplied] = useState(false);
  const { getLayoutedElements } = useGraphLayout();
  const { initializeGraph } = useGraphInitialization();

  const updateNodeDimensions = useCallback((id: string, width: number, height: number) => {
    setNodes((nds) =>
      nds.map((node) => {
        if (node.id === id) {
          if (node.width !== width || node.height !== height) {
            return { ...node, width, height };
          }
        }
        return node;
      })
    );
    setIsLayoutApplied(false);
  }, [setNodes]);

  const nodeTypes = useMemo(() => ({
    basicBlock: (props: NodeProps) => (
      <BasicBlock {...props} updateNodeDimensions={updateNodeDimensions} />
    ),
  }), [updateNodeDimensions]);

  useEffect(() => {
    const { nodes: initialNodes, edges: initialEdges } = initializeGraph(data);
    setNodes(initialNodes);
    setEdges(initialEdges);
    setIsLayoutApplied(false);
  }, [data, initializeGraph, setNodes, setEdges]);

  useEffect(() => {
    if (!isLayoutApplied && nodes.length > 0 && edges.length > 0) {
      const { nodes: layoutedNodes, edges: layoutedEdges } = getLayoutedElements(nodes, edges);
      setNodes(layoutedNodes);
      setEdges(layoutedEdges);
      setIsLayoutApplied(true);
    }
  }, [nodes, edges, isLayoutApplied, getLayoutedElements, setNodes, setEdges]);

  const onInit = useCallback((reactFlowInstance: ReactFlowInstance) => {
    reactFlowInstance.fitView({ padding: 0.2 });
  }, []);

  if (nodes.length === 0) {
    return <div>Loading CFG...</div>;
  }

  return (
    <div style={{ height: '800px', width: '100%' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        onInit={onInit}
        fitView
        attributionPosition="bottom-left"
        minZoom={0.1}
        nodesDraggable={true}
        nodesConnectable={false}
      >
        <Background color="#f0f0f0" gap={16} />
      </ReactFlow>
    </div>
  );
};

export default CFGViewer;
