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

  constructor(
    model_name: string,
    sess_opts: ort.InferenceSession.SessionOptions,
    ...args: unknown[]
  ) {
    /**
     * Initialize an instance of the BaseSession class.
     */
    this.model_name = model_name;
    this.sess_opts = sess_opts;

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

      // Download/fetch the model
      console.log(`Loading model: ${modelPath}`);
      const response = await fetch(modelPath);
      if (!response.ok) {
        throw new Error(`Failed to fetch model: ${response.statusText}`);
      }
      let modelArrayBuffer = await response.arrayBuffer();

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
