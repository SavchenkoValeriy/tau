import { Edge } from '@xyflow/react';
import dagre from 'dagre';
import { BasicBlockNode } from '../types';

export const getLayoutedElements = (nodes: BasicBlockNode[], edges: Edge[], direction = 'TB') => {
  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));
  dagreGraph.setGraph({ rankdir: direction, ranksep: 50, nodesep: 50 });

  nodes.forEach((node) => {
    dagreGraph.setNode(node.id, { width: node.measured?.width || 200, height: node.measured?.height || 100 });
  });

  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });

  dagre.layout(dagreGraph);

  const layoutedNodes = nodes.map((node) => {
    const position = dagreGraph.node(node.id);
    const x = position.x - (node.measured?.width ?? 0) / 2;
    const y = position.y - (node.measured?.height ?? 0) / 2;

    return { ...node, position: { x, y } };
  });

  return { nodes: layoutedNodes, edges };
};
