import React, { useState, useRef } from 'react';
import './App.css';

// Import our rembg-ts library
import { remove, new_session_async } from './rembg-ts';

interface ProcessingStatus {
  loading: boolean;
  error: string | null;
  success: boolean;
}

function App() {
  const [inputImage, setInputImage] = useState<string | null>(null);
  const [outputImage, setOutputImage] = useState<string | null>(null);
  const [status, setStatus] = useState<ProcessingStatus>({
    loading: false,
    error: null,
    success: false
  });
  const [selectedModel, setSelectedModel] = useState<'u2net' | 'u2netp'>('u2net');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadExampleImage = async () => {
    try {
      setStatus({ loading: true, error: null, success: false });
      setInputImage('/images/example.jpg');
      setOutputImage(null);
      setStatus({ loading: false, error: null, success: false });
    } catch (error) {
      setStatus({ loading: false, error: 'Failed to load example image', success: false });
    }
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setInputImage(e.target?.result as string);
        setOutputImage(null);
        setStatus({ loading: false, error: null, success: false });
      };
      reader.readAsDataURL(file);
    }
  };

  const processImage = async () => {
    if (!inputImage) {
      setStatus({ loading: false, error: 'Please select an image first', success: false });
      return;
    }

    setStatus({ loading: true, error: null, success: false });

    try {
      console.log('Starting background removal...');
      
      // Convert data URL to blob
      const response = await fetch(inputImage);
      const blob = await response.blob();
      const arrayBuffer = await blob.arrayBuffer();
      const imageBytes = new Uint8Array(arrayBuffer);

      console.log(`Processing with model: ${selectedModel}`);
      console.log('Image size:', imageBytes.length, 'bytes');

      // Create session for the selected model
      const session = await new_session_async(selectedModel);
      console.log('Session created successfully');      // Process the image
      const resultBytes = await remove(
        imageBytes,
        true,  // alpha_matting - enable to test MODNet model
        240,   // alpha_matting_foreground_threshold
        10,    // alpha_matting_background_threshold
        10,    // alpha_matting_erode_size
        session // Use our session
      ) as Uint8Array;

      console.log('Background removal completed, result size:', resultBytes.length, 'bytes');

      // Create blob URL for the result
      const resultBlob = new Blob([resultBytes], { type: 'image/png' });
      const resultUrl = URL.createObjectURL(resultBlob);
      
      setOutputImage(resultUrl);
      setStatus({ loading: false, error: null, success: true });

    } catch (error) {
      console.error('Error processing image:', error);
      setStatus({ 
        loading: false, 
        error: error instanceof Error ? error.message : 'Failed to process image', 
        success: false 
      });
    }
  };

  const downloadResult = () => {
    if (outputImage) {
      const link = document.createElement('a');
      link.href = outputImage;
      link.download = 'background-removed.png';
      link.click();
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🎭 rembg-ts Demo</h1>
        <p>TypeScript Background Removal in the Browser</p>
      </header>

      <main className="App-main">
        <div className="controls-section">
          <h2>Step 1: Load an Image</h2>
          <div className="button-group">
            <button 
              onClick={loadExampleImage}
              className="btn btn-secondary"
              disabled={status.loading}
            >
              Load Example Image
            </button>
            <button 
              onClick={() => fileInputRef.current?.click()}
              className="btn btn-secondary"
              disabled={status.loading}
            >
              Choose Your Own Image
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              style={{ display: 'none' }}
            />
          </div>

          <h2>Step 2: Select Model</h2>
          <div className="model-selector">
            <label>
              <input
                type="radio"
                value="u2net"
                checked={selectedModel === 'u2net'}
                onChange={(e) => setSelectedModel(e.target.value as 'u2net')}
                disabled={status.loading}
              />
              U2Net (Higher Quality, Slower)
            </label>
            <label>
              <input
                type="radio"
                value="u2netp"
                checked={selectedModel === 'u2netp'}
                onChange={(e) => setSelectedModel(e.target.value as 'u2netp')}
                disabled={status.loading}
              />
              U2NetP (Faster, Lighter)
            </label>
          </div>

          <h2>Step 3: Process</h2>
          <button 
            onClick={processImage}
            className="btn btn-primary"
            disabled={!inputImage || status.loading}
          >
            {status.loading ? 'Processing...' : 'Remove Background'}
          </button>
        </div>

        <div className="status-section">
          {status.loading && (
            <div className="status loading">
              <div className="spinner"></div>
              Processing image with {selectedModel}...
            </div>
          )}
          
          {status.error && (
            <div className="status error">
              ❌ Error: {status.error}
            </div>
          )}
          
          {status.success && (
            <div className="status success">
              ✅ Background removed successfully!
            </div>
          )}
        </div>

        <div className="images-section">
          <div className="image-container">
            <h3>Original Image</h3>
            {inputImage ? (
              <img src={inputImage} alt="Original" className="preview-image" />
            ) : (
              <div className="placeholder">No image selected</div>
            )}
          </div>

          <div className="image-container">
            <h3>Result</h3>
            {outputImage ? (
              <div>
                <img src={outputImage} alt="Background removed" className="preview-image" />
                <button onClick={downloadResult} className="btn btn-success">
                  Download Result
                </button>
              </div>
            ) : (
              <div className="placeholder">Processed image will appear here</div>
            )}
          </div>
        </div>

        <div className="info-section">
          <h2>About This Demo</h2>
          <p>
            This demo showcases the rembg-ts library, a TypeScript port of the popular Python rembg library.
            It performs background removal entirely in the browser using ONNX Runtime Web.
          </p>
          <ul>
            <li><strong>U2Net:</strong> Higher quality results, larger model size</li>
            <li><strong>U2NetP:</strong> Faster processing, smaller model size</li>
          </ul>
          <p>
            <strong>Note:</strong> The first time you use a model, it will be downloaded and cached by your browser.
            This may take a moment depending on your internet connection.
          </p>
        </div>
      </main>
    </div>
  );
}

export default App;
