import React, { useState } from 'react';
import { ReactFlowProvider } from '@xyflow/react';
import CFGViewer from './components/CFGViewer';
import { CFGWithTrace } from './types';

const App: React.FC = () => {
  const [data, setData] = useState<CFGWithTrace | null>(null);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const content = JSON.parse(e.target?.result as string) as CFGWithTrace;
          setData(content);
        } catch (error) {
          console.error('Failed to parse JSON:', error);
        }
      };
      reader.readAsText(file);
    }
  };

  return (
    <div className="App">
      <h1>Control Flow Graph Viewer</h1>
      {data ? (
        <ReactFlowProvider>
          <CFGViewer data={data} />
        </ReactFlowProvider>
      ) : (
        <>
          <input
            type="file"
            accept=".json"
            onChange={handleFileUpload}
            style={{ margin: '20px' }}
          />
          <div style={{
            padding: '20px',
            textAlign: 'center',
            color: '#666'
          }}>
            Please upload a trace JSON file
          </div>
        </>
      )}
    </div>
  );
};

export default App;
