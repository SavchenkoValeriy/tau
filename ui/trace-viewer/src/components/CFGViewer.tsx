import React from 'react';
import ReactFlow, { 
  Node, 
  Edge, 
  ConnectionLineType,
  useNodesState,
  useEdgesState,
  MarkerType,
  Background
} from 'react-flow-renderer';
import dagre from 'dagre';

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

const getNodeDimensions = (block: CFGData['func']['blocks'][0]) => {
  const maxLineLength = Math.max(...block.code.map(line => line.length));
  const width = Math.max(300, maxLineLength * 7 + 40); // Adjust multiplier as needed
  const height = Math.max(100, block.code.length * 20 + 40);
  return { width, height };
};

const getLayoutedElements = (nodes: Node[], edges: Edge[], direction = 'TB') => {
  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));
  dagreGraph.setGraph({ rankdir: direction, ranksep: 80, nodesep: 50 });

  nodes.forEach((node) => {
    dagreGraph.setNode(node.id, { width: node.width, height: node.height });
  });

  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });

  dagre.layout(dagreGraph);

  return {
    nodes: nodes.map((node) => {
      const nodeWithPosition = dagreGraph.node(node.id);
      return {
        ...node,
        position: {
          x: nodeWithPosition.x - node.width! / 2,
          y: nodeWithPosition.y - node.height! / 2,
        },
      };
    }),
    edges,
  };
};

const CFGViewer: React.FC<{ data: CFGData }> = ({ data }) => {
  const initialNodes: Node[] = data.func.blocks.map((block) => {
    const { width, height } = getNodeDimensions(block);
    return {
      id: block.name,
      data: { 
        label: (
          <div style={{ textAlign: 'left', padding: '10px', fontSize: '12px', fontFamily: 'monospace' }}>
            <strong>{block.name}</strong>
            {block.code.map((line, i) => (
              <div key={i} style={{ whiteSpace: 'pre', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {line}
              </div>
            ))}
          </div>
        )
      },
      style: { 
        width, 
        height, 
        border: '1px solid #ddd', 
        borderRadius: '5px',
        backgroundColor: 'white',
      },
      position: { x: 0, y: 0 },
      width,
      height
    };
  });

  const initialEdges: Edge[] = data.func.blocks.flatMap((block) =>
    block.edges.map((target) => ({
      id: `${block.name}-${data.func.blocks[target].name}`,
      source: block.name,
      target: data.func.blocks[target].name,
      type: ConnectionLineType.Step,
      style: { stroke: '#000', strokeWidth: 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#000' }
    }))
  );

  const { nodes: layoutedNodes, edges: layoutedEdges } = getLayoutedElements(
    initialNodes,
    initialEdges
  );

  const [nodes, setNodes, onNodesChange] = useNodesState(layoutedNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(layoutedEdges);

  return (
    <div style={{ height: '800px', width: '100%' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        fitView
        attributionPosition="bottom-left"
        minZoom={0.1}
      >
        <Background color="#f0f0f0" gap={16} />
      </ReactFlow>
    </div>
  );
};

export default CFGViewer;
