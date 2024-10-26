// hooks/useGraphLayout.ts
import { Node, Edge } from '@xyflow/react';
import dagre from 'dagre';
import { useCallback } from 'react';
import { BasicBlockNode } from '../types';

export const useGraphLayout = () => {
  const getLayoutedElements = useCallback(
    (nodes: BasicBlockNode[], edges: Edge[], direction = 'TB') => {
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
    },
    []
  );

  return { getLayoutedElements };
};
