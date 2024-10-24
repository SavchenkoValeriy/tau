import { Node, Edge, ConnectionLineType, MarkerType } from 'react-flow-renderer';
import { useCallback } from 'react';

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

export const useGraphInitialization = () => {
  const initializeGraph = useCallback((data: CFGData) => {
    if (data && data.func && data.func.blocks) {
      const initialNodes: Node[] = data.func.blocks.map((block) => ({
        id: block.name,
        type: 'basicBlock',
        data: {
          name: block.name,
          code: block.code,
          edges: block.edges,
        },
        position: { x: 0, y: 0 },
        style: {
          border: '1px solid #ddd',
          borderRadius: '5px',
          backgroundColor: 'white',
        },
      }));

      const initialEdges: Edge[] = data.func.blocks.flatMap((block, blockIndex) =>
        block.edges.map((target, edgeIndex) => ({
          id: `${block.name}-${data.func.blocks[target].name}-${edgeIndex}`,
          source: block.name,
          target: data.func.blocks[target].name,
          sourceHandle: `${block.name}-${edgeIndex}`,
          type: ConnectionLineType.SmoothStep,
          style: { stroke: '#000', strokeWidth: 2 },
          markerEnd: { type: MarkerType.ArrowClosed, color: '#000' },
        }))
      );

      return { nodes: initialNodes, edges: initialEdges };
    }
    return { nodes: [], edges: [] };
  }, []);

  return { initializeGraph };
};
