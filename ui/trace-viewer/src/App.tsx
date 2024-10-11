import React, { useEffect, useState } from 'react';
import CFGViewer from './components/CFGViewer';
import cfgData from './data/cfg.json';

const App: React.FC = () => {
  return (
    <div className="App">
      <h1>Control Flow Graph Viewer</h1>
      <CFGViewer data={cfgData} />
    </div>
  );
};

export default App;
