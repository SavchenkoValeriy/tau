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
import dagre from 'dagre';
import BasicBlock from './BasicBlock';

interface CFGData {
  func: {
    name: string;
    blocks: {
      name: string;
      code: string[];
      edges: number[];
    }[];
  };
}

const getLayoutedElements = (nodes: Node[], edges: Edge[], direction = 'TB') => {
  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));
  dagreGraph.setGraph({ rankdir: direction, ranksep: 150, nodesep: 150 });

  nodes.forEach((node) => {
    dagreGraph.setNode(node.id, { width: node.width || 200, height: node.height || 100 });
  });

  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });

  dagre.layout(dagreGraph);

  const layoutedNodes = nodes.map((node) => {
    const nodeWithPosition = dagreGraph.node(node.id);
    return {
      ...node,
      position: {
        x: nodeWithPosition.x - (node.width || 200) / 2,
        y: nodeWithPosition.y - (node.height || 100) / 2,
      },
    };
  });

  return { nodes: layoutedNodes, edges };
};

const CFGViewer: React.FC<{ data: CFGData }> = ({ data }) => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [isLayoutApplied, setIsLayoutApplied] = useState(false);

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

  const initializeGraph = useCallback(() => {
    if (data && data.func && data.func.blocks) {
      const initialNodes: Node[] = data.func.blocks.map((block) => ({
        id: block.name,
        type: 'basicBlock',
        data: { 
          name: block.name,
          code: block.code,
        },
        position: { x: 0, y: 0 },
        style: { 
          border: '1px solid #ddd', 
          borderRadius: '5px',
          backgroundColor: 'white',
        },
      }));

      const initialEdges: Edge[] = data.func.blocks.flatMap((block) =>
        block.edges.map((target) => ({
          id: `${block.name}-${data.func.blocks[target].name}`,
          source: block.name,
          target: data.func.blocks[target].name,
          type: ConnectionLineType.SmoothStep,
          style: { stroke: '#000', strokeWidth: 2 },
          markerEnd: { type: MarkerType.ArrowClosed, color: '#000' },
        }))
      );

      setNodes(initialNodes);
      setEdges(initialEdges);
      setIsLayoutApplied(false);
    }
  }, [data, setNodes, setEdges]);

  useEffect(() => {
    initializeGraph();
  }, [initializeGraph]);

  useEffect(() => {
    if (!isLayoutApplied && nodes.length > 0 && edges.length > 0) {
      const { nodes: layoutedNodes, edges: layoutedEdges } = getLayoutedElements(nodes, edges);
      setNodes(layoutedNodes);
      setEdges(layoutedEdges);
      setIsLayoutApplied(true);
    }
  }, [nodes, edges, isLayoutApplied, setNodes, setEdges]);

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
        nodesDraggable={false}
        nodesConnectable={false}
      >
        <Background color="#f0f0f0" gap={16} />
      </ReactFlow>
    </div>
  );
};

export default CFGViewer;
