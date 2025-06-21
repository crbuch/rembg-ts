import * as ort from "onnxruntime-web";
import * as npOpt from "../libraries/numpy_optimized";
import { Image } from "../libraries/PIL";
import type { PILImage } from "../libraries/PIL";

import { BaseSession } from "./base";

export class U2netSession extends BaseSession {
  /**
   * This class represents a U2net session, which is a subclass of BaseSession.
   */ async predict(img: PILImage, ..._args: unknown[]): Promise<PILImage[]> {
    /**
     * Predicts the output masks for the input image using the inner session.
     *
     * Parameters:
     *     img (PILImage): The input image.
     *     ..._args: Additional arguments.
     *
     * Returns:
     *     PILImage[]: The list of output masks.
     */
    await this.ensureInitialized();

    if (!this.inner_session) {
      throw new Error("Session not initialized");
    }

    const inputs = this.normalize(
      img,
      [0.485, 0.456, 0.406],
      [0.229, 0.224, 0.225],
      [320, 320]
    );

    try {
      // Convert NumpyArray to ONNX tensor format
      const inputTensors: Record<string, ort.Tensor> = {};
      for (const [name, npArray] of Object.entries(inputs)) {
        inputTensors[name] = new ort.Tensor(
          "float32",
          npArray.data as Float32Array,
          npArray.shape
        );
      }

      // Run inference
      const ort_outs = await this.inner_session.run(inputTensors);      // Get the output tensor (first output)
      const outputName = Object.keys(ort_outs)[0];
      const pred = ort_outs[outputName];
      
      // Use optimized tensor processing for better memory efficiency
      const maskArray = npOpt.processU2NetOutput(pred, pred.dims[2], pred.dims[3]);

      // Convert to PIL Image (mode "L" for grayscale)
      const mask = Image.fromarray(maskArray);
      const resized_mask = mask.resize(img.size, Image.Resampling.LANCZOS);

      return [resized_mask];
    } catch (error) {
      console.error("Prediction failed:", error);
      throw error;
    }
  }
  static async download_models(...args: unknown[]): Promise<string> {
    /**
     * Downloads the U2net model file from a specific URL and saves it.
     *
     * Parameters:
     *     ...args: Additional arguments.
     *
     * Returns:
     *     string: The path to the downloaded model file.
     */
    const fname = `${this.getModelName(...args)}.onnx`;
    const model_path = `/models/${fname}`;

    try {
      console.log(`Checking U2net model availability: ${model_path}`);
      const response = await fetch(model_path, { method: "HEAD" });
      if (!response.ok) {
        throw new Error(`U2net model not found: ${response.statusText}`);
      }
      console.log(`U2net model is available at: ${model_path}`);
      return model_path;
    } catch (error) {
      console.error(`Failed to verify U2net model:`, error);
      throw error;
    }
  }

  static getModelName(..._args: unknown[]): string {
    /**
     * Returns the name of the U2net session.
     *
     * Parameters:
     *     ..._args: Additional arguments.
     *
     * Returns:
     *     string: The name of the session.
     */
    return "u2net";
  }  async predict_batch(imgs: PILImage[], ..._args: unknown[]): Promise<PILImage[][]> {
    /**
     * Predicts the output masks for a batch of input images using a true pipeline approach.
     * 
     * Pipeline phases:
     * 1. Preprocess ALL images first
     * 2. Run ALL images through U2Net model sequentially, collecting ALL predictions
     * 3. Process ALL predictions into masks
     *
     * Parameters:
     *     imgs (PILImage[]): The batch of input images.
     *     ..._args: Additional arguments.
     *
     * Returns:
     *     PILImage[][]: Array of mask arrays, one for each input image.
     */
    await this.ensureInitialized();

    if (!this.inner_session) {
      throw new Error("Session not initialized");
    }

    if (imgs.length === 0) {
      return [];
    }

    try {      // Phase 1: Preprocess all images first
      const preprocessedInputs: { inputs: Record<string, ort.Tensor>, originalSize: [number, number] }[] = [];
      
      for (const img of imgs) {
        const inputs = this.normalize(
          img,
          [0.485, 0.456, 0.406],
          [0.229, 0.224, 0.225],
          [320, 320]
        );

        // Convert NumpyArray to ONNX tensor format
        const inputTensors: Record<string, ort.Tensor> = {};
        for (const [name, npArray] of Object.entries(inputs)) {
          inputTensors[name] = new ort.Tensor(
            "float32",
            npArray.data as Float32Array,
            npArray.shape
          );
        }

        preprocessedInputs.push({
          inputs: inputTensors,
          originalSize: img.size
        });
      }

      // Phase 2: Run ALL images through the model and collect ALL predictions
      const allPredictions: { pred: ort.Tensor, originalSize: [number, number] }[] = [];
      
      for (let i = 0; i < preprocessedInputs.length; i++) {
        const { inputs, originalSize } = preprocessedInputs[i];
        
        // Run inference for single image
        const ort_outs = await this.inner_session.run(inputs);
        
        // Get the output tensor (first output) and store it
        const outputName = Object.keys(ort_outs)[0];
        const pred = ort_outs[outputName];
        console.log(`Processed frame ${i + 1}`);
        
        allPredictions.push({ pred, originalSize });
      }

      // Phase 3: Process ALL predictions into masks
      const results: PILImage[][] = [];
      
      for (const { pred, originalSize } of allPredictions) {
        // Use optimized tensor processing for better memory efficiency
        const maskArray = npOpt.processU2NetOutput(pred, pred.dims[2], pred.dims[3]);

        // Convert to PIL Image (mode "L" for grayscale)
        const mask = Image.fromarray(maskArray);
        const resized_mask = mask.resize(originalSize, Image.Resampling.LANCZOS);

        results.push([resized_mask]);
      }

      return results;
    } catch (error) {
      console.error("Batch prediction failed:", error);
      throw error;
    }
  }
}
