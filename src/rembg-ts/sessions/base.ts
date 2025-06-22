import * as os from "../libraries/os";
import * as np from "../libraries/numpy";
import type { NumpyArray } from "../libraries/numpy";
import * as ort from "onnxruntime-web";
import { Image } from "../libraries/PIL";
import type { PILImage } from "../libraries/PIL";

export class BaseSession {
  /**
   * This is a base class for managing a session with a machine learning model.
   */

  protected model_name: string;
  protected inner_session: ort.InferenceSession | null = null;
  protected sess_opts: ort.InferenceSession.SessionOptions;
  protected initialized: boolean = false;
  protected downloadProgressCallback?: (loaded: number, total: number) => void;

  constructor(
    model_name: string,
    sess_opts: ort.InferenceSession.SessionOptions,
    downloadProgressCallback?: (loaded: number, total: number) => void,
    ...args: unknown[]
  ) {    /**
     * Initialize an instance of the BaseSession class.
     */
    this.model_name = model_name;
    this.sess_opts = sess_opts;
    this.downloadProgressCallback = downloadProgressCallback;

    // Determine execution providers based on device capabilities
    const providers: string[] = [];

    // For web environment, we'll use CPU or WebGPU
    if (typeof navigator !== "undefined" && "gpu" in navigator) {
      // WebGPU is available
      providers.push("webgpu", "cpu");
    } else {
      // Fallback to CPU
      providers.push("cpu");
    }

    // Set providers in session options
    if (!this.sess_opts.executionProviders) {
      this.sess_opts.executionProviders = providers;
    }
  }

  /**
   * Initialize the ONNX session asynchronously.
   * This method loads the model and creates the inference session.
   */
  async initialize(): Promise<void> {
    if (this.initialized) {
      return;
    }

    try {
      const modelPath = this.getModelPath();

      // Check if model is cached in IndexedDB first
      const cachedModel = await this.getCachedModel(modelPath);
      let modelArrayBuffer: ArrayBuffer;

      if (cachedModel) {
        console.log(`Using cached model: ${modelPath}`);
        modelArrayBuffer = cachedModel;
        // Don't report progress for cached models - they load instantly
      } else {
        console.log(`Downloading model: ${modelPath}`);
        
        // Fetch with strong caching headers
        const response = await fetch(modelPath, {
          cache: 'force-cache', // Use cache if available
          headers: {
            'Cache-Control': 'max-age=31536000', // Cache for 1 year
          }
        });
        
        if (!response.ok) {
          throw new Error(`Failed to fetch model: ${response.statusText}`);
        }

        // Get the total size from Content-Length header
        const contentLength = response.headers.get('Content-Length');
        const totalSize = contentLength ? parseInt(contentLength, 10) : 0;

        if (!response.body) {
          throw new Error('Response body is null');
        }

        // Read the response with progress tracking
        const reader = response.body.getReader();
        const chunks: Uint8Array[] = [];
        let loadedSize = 0;

        while (true) {
          const { done, value } = await reader.read();
          
          if (done) break;
          
          chunks.push(value);
          loadedSize += value.length;

          // Report progress if callback is provided
          if (this.downloadProgressCallback && totalSize > 0) {
            this.downloadProgressCallback(loadedSize, totalSize);
          }
        }

        // Combine all chunks into a single ArrayBuffer
        const totalBytes = chunks.reduce((acc, chunk) => acc + chunk.length, 0);
        const combinedArray = new Uint8Array(totalBytes);
        let offset = 0;
        for (const chunk of chunks) {
          combinedArray.set(chunk, offset);
          offset += chunk.length;
        }
        
        modelArrayBuffer = combinedArray.buffer;
        
        // Cache the model in IndexedDB for future use
        await this.cacheModel(modelPath, modelArrayBuffer);
        console.log(`Model cached: ${modelPath}`);
      }

      // Apply WebGPU MaxPool fix if using WebGPU execution provider
      const executionProviders = this.sess_opts.executionProviders || [];
      const isUsingWebGPU = executionProviders.some((provider) =>
        typeof provider === "string"
          ? provider === "webgpu"
          : provider.name === "webgpu"
      );

      if (isUsingWebGPU) {
        console.log("Applying WebGPU MaxPool fix...");
        modelArrayBuffer = await fixMaxPoolForWebGPU(modelArrayBuffer);
      }

      // Create the ONNX Runtime session
      this.inner_session = await ort.InferenceSession.create(
        modelArrayBuffer,
        this.sess_opts
      );
      this.initialized = true;

      console.log(`Model ${this.model_name} loaded successfully`);
    } catch (error) {
      console.error(
        `Failed to initialize session for ${this.model_name}:`,
        error
      );
      throw error;
    }
  }

  /**
   * Get the model path for fetching from public/models directory
   */
  protected getModelPath(): string {
    return `/models/${this.model_name}.onnx`;
  }

  /**
   * Ensure the session is initialized before use
   */
  protected async ensureInitialized(): Promise<void> {
    if (!this.initialized) {
      await this.initialize();
    }
  }
  normalize(
    img: PILImage,
    mean: [number, number, number],
    std: [number, number, number],
    size: [number, number],
    ..._args: unknown[]
  ): Record<string, NumpyArray> {
    const im = img.convert("RGB").resize(size, Image.Resampling.LANCZOS);

    let im_ary = np.array(im);
    im_ary = im_ary.divide(Math.max(np.max(im_ary), 1e-6));

    const tmpImg = np.zeros([im_ary.shape[0], im_ary.shape[1], 3]);

    // Extract each channel, normalize, and set back
    // tmpImg[:, :, 0] = (im_ary[:, :, 0] - mean[0]) / std[0]
    const channel0 = im_ary.getChannel(0).subtract(mean[0]).divide(std[0]);
    const channel1 = im_ary.getChannel(1).subtract(mean[1]).divide(std[1]);
    const channel2 = im_ary.getChannel(2).subtract(mean[2]).divide(std[2]);

    tmpImg.setChannel(0, channel0);
    tmpImg.setChannel(1, channel1);
    tmpImg.setChannel(2, channel2); // Transpose from HWC to CHW: tmpImg.transpose((2, 0, 1))
    const transposed = tmpImg.transpose([2, 0, 1]);

    // Get input name from session (in ONNX runtime web, use getInputNames() or access from inputs)
    if (!this.inner_session) {
      throw new Error("Session not initialized");
    }
    const inputName = this.inner_session.inputNames[0];

    return {
      [inputName]: np.expand_dims(transposed, 0).astype(np.float32),
    };
  }

  normalize_batch(
    imgs: PILImage[],
    mean: [number, number, number],
    std: [number, number, number],
    size: [number, number],
    ..._args: unknown[]
  ): Record<string, NumpyArray> {
    if (imgs.length === 0) {
      throw new Error("Empty image batch provided");
    }

    // Process all images and create batch tensor
    const processedImages: NumpyArray[] = [];

    for (const img of imgs) {
      const im = img.convert("RGB").resize(size, Image.Resampling.LANCZOS);
      let im_ary = np.array(im);
      im_ary = im_ary.divide(Math.max(np.max(im_ary), 1e-6));

      const tmpImg = np.zeros([im_ary.shape[0], im_ary.shape[1], 3]);

      // Extract each channel, normalize, and set back
      const channel0 = im_ary.getChannel(0).subtract(mean[0]).divide(std[0]);
      const channel1 = im_ary.getChannel(1).subtract(mean[1]).divide(std[1]);
      const channel2 = im_ary.getChannel(2).subtract(mean[2]).divide(std[2]);

      tmpImg.setChannel(0, channel0);
      tmpImg.setChannel(1, channel1);
      tmpImg.setChannel(2, channel2);

      // Transpose from HWC to CHW: tmpImg.transpose((2, 0, 1))
      const transposed = tmpImg.transpose([2, 0, 1]);
      processedImages.push(transposed);
    }

    // Stack all images into a batch: [batch_size, channels, height, width]
    const batchTensor = np.stack(processedImages, 0).astype(np.float32);

    // Get input name from session
    if (!this.inner_session) {
      throw new Error("Session not initialized");
    }
    const inputName = this.inner_session.inputNames[0];

    return {
      [inputName]: batchTensor,
    };
  }

  async predict(_img: PILImage, ..._args: unknown[]): Promise<PILImage[]> {
    throw new Error(
      "NotImplementedError: predict method must be implemented by subclass"
    );
  }

  async predict_batch(
    imgs: PILImage[],
    ..._args: unknown[]
  ): Promise<PILImage[][]> {
    throw new Error(
      "NotImplementedError: predict_batch method must be implemented by subclass"
    );
  }

  static checksum_disabled(..._args: unknown[]): boolean {
    return os.getenv("MODEL_CHECKSUM_DISABLED") !== null;
  }

  static u2net_home(..._args: unknown[]): string {
    return os.path.expanduser(
      os.getenv("U2NET_HOME") ||
        os.path.join(os.getenv("XDG_DATA_HOME") || "~", ".u2net")
    );
  }
  static async download_models(..._args: unknown[]): Promise<string> {
    // In web environment, models are served from /public/models
    // This method can be used to pre-fetch and cache models
    const modelName = this.getModelName(..._args);
    const modelPath = `/models/${modelName}.onnx`;

    try {
      console.log(`Checking model availability: ${modelPath}`);
      const response = await fetch(modelPath, { method: "HEAD" });
      if (!response.ok) {
        throw new Error(`Model not found: ${response.statusText}`);
      }
      console.log(`Model ${modelName} is available`);
      return modelPath;
    } catch (error) {
      console.error(`Failed to verify model ${modelName}:`, error);
      throw error;
    }
  }

  static getModelName(..._args: unknown[]): string {
    throw new Error(
      "NotImplementedError: getModelName method must be implemented by subclass"
    );
  }

  /**
   * Get cached model from IndexedDB
   */
  private async getCachedModel(modelPath: string): Promise<ArrayBuffer | null> {
    try {
      const dbName = 'rembg-models';
      const storeName = 'models';
      const version = 1;

      return new Promise((resolve, reject) => {
        const request = indexedDB.open(dbName, version);
        
        request.onerror = () => {
          console.log('IndexedDB error:', request.error);
          resolve(null);
        };
        
        request.onupgradeneeded = (event) => {
          const db = (event.target as IDBOpenDBRequest).result;
          if (!db.objectStoreNames.contains(storeName)) {
            db.createObjectStore(storeName);
          }
        };
        
        request.onsuccess = () => {
          const db = request.result;
          const transaction = db.transaction([storeName], 'readonly');
          const store = transaction.objectStore(storeName);
          const getRequest = store.get(modelPath);
          
          getRequest.onsuccess = () => {
            const result = getRequest.result;
            if (result && result.data && result.timestamp) {
              // Check if cache is still valid (e.g., not older than 30 days)
              const thirtyDaysAgo = Date.now() - (30 * 24 * 60 * 60 * 1000);
              if (result.timestamp > thirtyDaysAgo) {
                console.log('Found cached model:', modelPath);
                resolve(result.data);
              } else {
                console.log('Cached model expired:', modelPath);
                resolve(null);
              }
            } else {
              resolve(null);
            }
          };
          
          getRequest.onerror = () => {
            console.log('Error retrieving cached model:', getRequest.error);
            resolve(null);
          };
        };
      });
    } catch (error) {
      console.log('Error accessing IndexedDB:', error);
      return null;
    }
  }

  /**
   * Cache model in IndexedDB
   */
  private async cacheModel(modelPath: string, modelData: ArrayBuffer): Promise<void> {
    try {
      const dbName = 'rembg-models';
      const storeName = 'models';
      const version = 1;

      return new Promise((resolve, reject) => {
        const request = indexedDB.open(dbName, version);
        
        request.onerror = () => {
          console.log('IndexedDB error during caching:', request.error);
          resolve(); // Don't fail if caching fails
        };
        
        request.onupgradeneeded = (event) => {
          const db = (event.target as IDBOpenDBRequest).result;
          if (!db.objectStoreNames.contains(storeName)) {
            db.createObjectStore(storeName);
          }
        };
        
        request.onsuccess = () => {
          const db = request.result;
          const transaction = db.transaction([storeName], 'readwrite');
          const store = transaction.objectStore(storeName);
          
          const cacheEntry = {
            data: modelData,
            timestamp: Date.now(),
            size: modelData.byteLength
          };
          
          const putRequest = store.put(cacheEntry, modelPath);
          
          putRequest.onsuccess = () => {
            console.log('Model cached successfully:', modelPath, `(${(modelData.byteLength / 1024 / 1024).toFixed(1)}MB)`);
            resolve();
          };
          
          putRequest.onerror = () => {
            console.log('Error caching model:', putRequest.error);
            resolve(); // Don't fail if caching fails
          };
        };
      });
    } catch (error) {
      console.log('Error caching model:', error);
      // Don't throw - caching failure shouldn't break the app
    }
  }

  /**
   * Clear cached models (utility method)
   */
  static async clearModelCache(): Promise<void> {
    try {
      const dbName = 'rembg-models';
      return new Promise((resolve, reject) => {
        const deleteRequest = indexedDB.deleteDatabase(dbName);
        deleteRequest.onsuccess = () => {
          console.log('Model cache cleared');
          resolve();
        };
        deleteRequest.onerror = () => {
          console.log('Error clearing model cache:', deleteRequest.error);
          resolve();
        };
      });
    } catch (error) {
      console.log('Error clearing model cache:', error);
    }
  }
}

/**
 * Fix MaxPool operations for WebGPU compatibility
 * This function modifies the ONNX model to disable ceil_mode in MaxPool operations
 * which is not yet supported by the WebGPU execution provider
 */
export const fixMaxPoolForWebGPU = async (
  modelBuffer: ArrayBuffer
): Promise<ArrayBuffer> => {
  try {
    const modelBytes = new Uint8Array(modelBuffer);
    const modifiedBytes = new Uint8Array(modelBytes);

    let modificationsCount = 0;
    const ceilModePattern = new TextEncoder().encode("ceil_mode");

    for (
      let i = 0;
      i < modifiedBytes.length - ceilModePattern.length - 10;
      i++
    ) {
      let patternFound = true;
      for (let j = 0; j < ceilModePattern.length; j++) {
        if (modifiedBytes[i + j] !== ceilModePattern[j]) {
          patternFound = false;
          break;
        }
      }

      if (patternFound) {
        // Look for protobuf encoding patterns that indicate integer value 1
        for (
          let k = i + ceilModePattern.length;
          k < i + ceilModePattern.length + 20;
          k++
        ) {
          if (k < modifiedBytes.length - 1) {
            if (
              (modifiedBytes[k] === 0x08 && modifiedBytes[k + 1] === 0x01) ||
              (modifiedBytes[k] === 0x10 && modifiedBytes[k + 1] === 0x01) ||
              (modifiedBytes[k] === 0x18 && modifiedBytes[k + 1] === 0x01)
            ) {
              console.log(`Fixed MaxPool ceil_mode at position ${k}`);
              modifiedBytes[k + 1] = 0x00;
              modificationsCount++;
              break;
            }
          }
        }
      }
    }

    console.log(
      `Modified ${modificationsCount} MaxPool attributes for WebGPU compatibility`
    );
    return modifiedBytes.buffer;
  } catch (error) {
    console.warn(
      "Failed to fix MaxPool operations, using original model:",
      error
    );
    return modelBuffer;
  }
};
