import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

// Service Worker Registration for Model Caching
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js')
      .then((registration) => {
        console.log('Service Worker registered successfully:', registration.scope);
        
        // Check if models are already cached
        checkModelCache();
        
        // Listen for service worker updates
        registration.addEventListener('updatefound', () => {
          console.log('Service Worker update found');
        });
      })
      .catch((error) => {
        console.error('Service Worker registration failed:', error);
      });
  });
}

// Utility function to check model cache status
async function checkModelCache(): Promise<void> {
  if ('serviceWorker' in navigator && navigator.serviceWorker.controller) {
    const messageChannel = new MessageChannel();
    
    messageChannel.port1.onmessage = (event) => {
      if (event.data.type === 'CACHE_STATUS') {
        const cachedModels = event.data.results.filter((r: any) => r.cached);
        console.log(`Models cached: ${cachedModels.length}/${event.data.results.length}`);
        
        if (cachedModels.length === event.data.results.length) {
          console.log('✅ All models are cached and ready!');
          // You could show a UI indicator here
          localStorage.setItem('rembg-models-cached', 'true');
        } else {
          console.log('⏳ Some models still need to be cached');
          localStorage.setItem('rembg-models-cached', 'false');
        }
      }
    };
    
    navigator.serviceWorker.controller.postMessage(
      { type: 'CHECK_CACHE' },
      [messageChannel.port2]
    );
  }
}

// Make cache checking available globally
(window as any).checkModelCache = checkModelCache;

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
