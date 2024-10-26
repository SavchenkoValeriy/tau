import React from 'react';
import { ReactFlowProvider } from '@xyflow/react';
import CFGViewer from './components/CFGViewer';
import cfgData from './data/cfg.json';

const App: React.FC = () => {
  return (
    <div className="App">
      <h1>Control Flow Graph Viewer</h1>
      <ReactFlowProvider>
        <CFGViewer data={cfgData} />
      </ReactFlowProvider>
    </div>
  );
};

export default App;
