import React, { useState, useRef } from 'react';
import './App.css';

// Import our rembg-ts library
import { remove, remove_video, new_session_async } from './rembg-ts';

interface ProcessingStatus {
  loading: boolean;
  error: string | null;
  success: boolean;
  progress?: { current: number; total: number };
  downloadProgress?: { loaded: number; total: number };
}

interface CacheStatus {
  modelsAvailable: boolean;
  cachedModels: string[];
  checkingCache: boolean;
}

type FileType = 'image' | 'video';

function App() {
  const [inputFile, setInputFile] = useState<string | null>(null);
  const [inputFileType, setInputFileType] = useState<FileType>('image');
  const [outputFile, setOutputFile] = useState<string | null>(null);
  const [status, setStatus] = useState<ProcessingStatus>({
    loading: false,
    error: null,
    success: false
  });
  const [useAlphaMatting, setUseAlphaMatting] = useState<boolean>(true);
  const [cacheModels, setCacheModels] = useState<boolean>(true);
  const [cacheStatus, setCacheStatus] = useState<CacheStatus>({
    modelsAvailable: false,
    cachedModels: [],
    checkingCache: true
  });
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoInputRef = useRef<HTMLInputElement>(null);
  const [currentFile, setCurrentFile] = useState<File | null>(null);

  // Check cache status on component mount
  React.useEffect(() => {
    checkModelCacheStatus();
  }, []);

  const checkModelCacheStatus = async () => {
    try {
      setCacheStatus(prev => ({ ...prev, checkingCache: true }));
      
      // Check localStorage first
      const modelsAreCached = localStorage.getItem('rembg-models-cached') === 'true';
      
      // Also check if service worker is available to get detailed status
      if ('serviceWorker' in navigator && navigator.serviceWorker.controller) {
        const messageChannel = new MessageChannel();
        
        messageChannel.port1.onmessage = (event) => {
          if (event.data.type === 'CACHE_STATUS') {
            const cachedModels = event.data.results.filter((r: any) => r.cached);
            setCacheStatus({
              modelsAvailable: cachedModels.length === event.data.results.length,
              cachedModels: cachedModels.map((r: any) => r.url),
              checkingCache: false
            });
          }
        };
        
        navigator.serviceWorker.controller.postMessage(
          { type: 'CHECK_CACHE' },
          [messageChannel.port2]
        );
      } else {
        // Fallback when service worker is not available
        setCacheStatus({
          modelsAvailable: modelsAreCached,
          cachedModels: modelsAreCached ? ['u2net'] : [],
          checkingCache: false
        });
      }
    } catch (error) {
      console.error('Error checking cache status:', error);
      setCacheStatus({
        modelsAvailable: false,
        cachedModels: [],
        checkingCache: false
      });
    }
  };
  const loadExampleImage = async () => {
    try {
      setStatus({ loading: true, error: null, success: false });
      setInputFile('/images/example.jpg');
      setInputFileType('image');
      setOutputFile(null);
      setCurrentFile(null);
      setStatus({ loading: false, error: null, success: false });
    } catch (error) {
      setStatus({ loading: false, error: 'Failed to load example image', success: false });
    }
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>, fileType: FileType) => {
    const file = event.target.files?.[0];
    if (file) {
      setCurrentFile(file);
      setInputFileType(fileType);
      
      const reader = new FileReader();
      reader.onload = (e) => {
        setInputFile(e.target?.result as string);
        setOutputFile(null);
        setStatus({ loading: false, error: null, success: false });
      };
      reader.readAsDataURL(file);
    }
  };
  const processFile = async () => {
    if (!inputFile) {
      setStatus({ loading: false, error: 'Please select a file first', success: false });
      return;
    }

    setStatus({ loading: true, error: null, success: false });

    try {
      console.log(`Starting ${inputFileType} processing...`);
      
      // Create session for the u2net model with download progress callback
      // Only show download progress if models aren't already cached
      const session = await new_session_async('u2net', 
        !cacheStatus.modelsAvailable ? (loaded: number, total: number) => {
          setStatus(prev => ({ 
            ...prev, 
            downloadProgress: { loaded, total } 
          }));
        } : undefined
      );
      
      // Clear download progress once session is created
      setStatus(prev => ({ ...prev, downloadProgress: undefined }));
      console.log('Session created successfully');

      if (inputFileType === 'image') {
        // Process image
        const response = await fetch(inputFile);
        const blob = await response.blob();
        const arrayBuffer = await blob.arrayBuffer();
        const imageBytes = new Uint8Array(arrayBuffer);

        console.log('Processing image with U2Net model');
        console.log('Image size:', imageBytes.length, 'bytes');        const resultBytes = await remove(
          imageBytes,
          useAlphaMatting, // alpha_matting
          240,   // alpha_matting_foreground_threshold
          10,    // alpha_matting_background_threshold
          10,    // alpha_matting_erode_size
          session // Use our session
        ) as Uint8Array;

        console.log('Background removal completed, result size:', resultBytes.length, 'bytes');

        // Create blob URL for the result
        const resultBlob = new Blob([resultBytes], { type: 'image/png' });
        const resultUrl = URL.createObjectURL(resultBlob);
        
        setOutputFile(resultUrl);
      } else {
        // Process video
        if (!currentFile) {
          throw new Error('No video file selected');
        }

        console.log('Processing video with U2Net model');
        console.log('Video file:', currentFile.name, currentFile.size, 'bytes');        const resultUrl = await remove_video(
          currentFile,
          useAlphaMatting, // alpha_matting
          240,   // alpha_matting_foreground_threshold
          10,    // alpha_matting_background_threshold
          10,    // alpha_matting_erode_size
          session, // Use our session
          false, // only_mask
          false, // post_process_mask
          (current: number, total: number) => {
            setStatus(prev => ({ 
              ...prev, 
              progress: { current, total } 
            }));
          }
        );

        console.log('Video background removal completed');
        setOutputFile(resultUrl);
      }
      
      setStatus({ loading: false, error: null, success: true });

    } catch (error) {
      console.error(`Error processing ${inputFileType}:`, error);
      setStatus({ 
        loading: false, 
        error: error instanceof Error ? error.message : `Failed to process ${inputFileType}`, 
        success: false 
      });
    }
  };
  const downloadResult = () => {
    if (outputFile) {
      const link = document.createElement('a');
      link.href = outputFile;
      link.download = inputFileType === 'image' ? 'background-removed.png' : 'background-removed.mp4';
      link.click();
    }
  };

  return (
    <div className="App">      <header className="App-header">
        <h1>🎭 rembg-ts Demo</h1>
        <p>TypeScript Background Removal for Images & Videos in the Browser</p>
        
        {/* Cache Status Indicator */}
        <div className="cache-status">
          {cacheStatus.checkingCache ? (
            <div className="cache-indicator checking">
              <span>🔄 Checking cache...</span>
            </div>
          ) : cacheStatus.modelsAvailable ? (
            <div className="cache-indicator cached">
              <span>✅ Models cached - Fast processing ready!</span>
            </div>
          ) : (
            <div className="cache-indicator not-cached">
              <span>⏳ Models will be downloaded and cached on first use</span>
            </div>
          )}
        </div>
      </header>

      <main className="App-main">        <div className="controls-section">
          <h2>Step 1: Load a File</h2>
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
              Choose Image
            </button>
            <button 
              onClick={() => videoInputRef.current?.click()}
              className="btn btn-secondary"
              disabled={status.loading}
            >
              Choose Video
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={(e) => handleFileSelect(e, 'image')}
              style={{ display: 'none' }}
            />
            <input
              ref={videoInputRef}
              type="file"
              accept="video/*"
              onChange={(e) => handleFileSelect(e, 'video')}
              style={{ display: 'none' }}
            />
          </div>

          <h2>Step 2: Processing Options</h2>
          <div className="processing-options">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={useAlphaMatting}
                onChange={(e) => setUseAlphaMatting(e.target.checked)}
                disabled={status.loading}
              />
              Use Alpha Matting (Better edge quality, slower processing)
            </label>
          </div>

          <h2>Step 3: Process</h2>
          <button 
            onClick={processFile}
            className="btn btn-primary"
            disabled={!inputFile || status.loading}
          >
            {status.loading ? `Processing ${inputFileType}...` : `Remove Background from ${inputFileType}`}
          </button>
        </div>        <div className="status-section">          {status.loading && (
            <div className="status loading">
              <div className="spinner"></div>
              {status.downloadProgress ? (
                <div>
                  <div>Downloading U2Net model...</div>
                  <div className="progress-bar">
                    <div 
                      className="progress-fill" 
                      style={{ 
                        width: `${Math.round((status.downloadProgress.loaded / status.downloadProgress.total) * 100)}%` 
                      }}
                    ></div>
                  </div>
                  <div className="progress-text">
                    {Math.round((status.downloadProgress.loaded / status.downloadProgress.total) * 100)}% 
                    ({(status.downloadProgress.loaded / 1024 / 1024).toFixed(1)}MB / {(status.downloadProgress.total / 1024 / 1024).toFixed(1)}MB)
                  </div>
                  {/* Cache Models Checkbox */}
                  <div className="cache-checkbox">
                    <label className="checkbox-label">
                      <input
                        type="checkbox"
                        checked={cacheModels}
                        onChange={(e) => setCacheModels(e.target.checked)}
                      />
                      Cache models for faster future use
                    </label>
                  </div>
                </div>
              ) : (
                <div>
                  {cacheStatus.modelsAvailable ? 
                    `Processing ${inputFileType} with U2Net${useAlphaMatting ? ' + Alpha Matting' : ''}...` :
                    `Loading U2Net model and processing ${inputFileType}${useAlphaMatting ? ' with Alpha Matting' : ''}...`
                  }
                  {status.progress && inputFileType === 'video' && (
                    <div>Frame {status.progress.current} processed</div>
                  )}
                </div>
              )}
            </div>
          )}
          
          {status.error && (
            <div className="status error">
              ❌ Error: {status.error}
            </div>
          )}
          
          {status.success && (
            <div className="status success">
              ✅ Background removed from {inputFileType} successfully!
            </div>
          )}
        </div>

        <div className="images-section">
          <div className="image-container">
            <h3>Original {inputFileType === 'image' ? 'Image' : 'Video'}</h3>
            {inputFile ? (
              inputFileType === 'image' ? (
                <img src={inputFile} alt="Original" className="preview-image" />
              ) : (
                <video src={inputFile} controls className="preview-video" />
              )
            ) : (
              <div className="placeholder">No {inputFileType} selected</div>
            )}
          </div>

          <div className="image-container">
            <h3>Result</h3>
            {outputFile ? (
              <div>
                {inputFileType === 'image' ? (
                  <img src={outputFile} alt="Background removed" className="preview-image" />
                ) : (
                  <video src={outputFile} controls className="preview-video" />
                )}
                <button onClick={downloadResult} className="btn btn-success">
                  Download Result
                </button>
              </div>
            ) : (
              <div className="placeholder">Processed {inputFileType} will appear here</div>
            )}
          </div>
        </div>        <div className="info-section">
          <h2>About This Demo</h2>          <p>
            This demo showcases the rembg-ts library, a TypeScript port of the popular Python rembg library.
            It performs background removal entirely in the browser using ONNX Runtime Web and supports both images and videos using the U2Net model.
          </p>
          <ul>
            <li><strong>U2Net:</strong> High quality background removal model</li>
            <li><strong>Images:</strong> PNG, JPEG, WebP, and other common formats</li>
            <li><strong>Videos:</strong> MP4 and other common video formats (processed frame by frame)</li>
          </ul>
          <p>
            <strong>Note:</strong> The first time you use a model, it will be downloaded and cached by your browser.
            Video processing may take longer as each frame is processed individually. FFmpeg.wasm is used for video handling.
          </p>
        </div>
      </main>
    </div>
  );
}

export default App;
