import { Edge, MarkerType } from '@xyflow/react';
import { BasicBlockNode, CFGData } from '../types';

export const initializeGraph = (data: CFGData) => {
  if (data && data.func && data.func.blocks) {
    const initialNodes: BasicBlockNode[] = data.func.blocks.map((block) => ({
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
        padding: '10px',
      },
    }));

    const initialEdges: Edge[] = data.func.blocks.flatMap((block) =>
      block.edges.map((targetIndex, edgeIndex) => {
        const targetBlock = data.func.blocks[targetIndex];
        return {
          id: `${block.name}-${targetBlock.name}-${edgeIndex}`,
          source: block.name,
          target: targetBlock.name,
          type: 'smoothstep',
          style: { stroke: '#000', strokeWidth: 2 },
          markerEnd: { type: MarkerType.ArrowClosed, color: '#000' },
        };
      })
    );

    return { nodes: initialNodes, edges: initialEdges };
  }
  return { nodes: [], edges: [] };
};
