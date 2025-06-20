import * as io from './libraries/io';
import * as ort from 'onnxruntime-web';

import * as np from './libraries/numpy';
import type { NumpyArray } from './libraries/numpy';
import {
    BORDER_DEFAULT,
    MORPH_ELLIPSE,
    MORPH_OPEN,
    GaussianBlur,
    getStructuringElement,
    morphologyEx,
} from './libraries/cv2';
import { Image } from './libraries/PIL';
import type { PILImage } from './libraries/PIL';
import { estimate_alpha_cf, estimate_foreground_ml, stack_images } from './libraries/pymatting';
import { binary_erosion } from './libraries/scipy.ndimage';

import { new_session } from './session_factory';
import { sessions, sessions_names } from './libraries/sessions';
import type { BaseSession } from './sessions/base';
import { VideoFrameProcessor } from './libraries/video';

ort.env.logLevel = 'error';  // Equivalent to ort.set_default_logger_severity(3)

const kernel = getStructuringElement(MORPH_ELLIPSE, [3, 3]);

enum ReturnType {
    BYTES = 0,
    PILLOW = 1,
    NDARRAY = 2,
}

async function alpha_matting_cutout(
    img: PILImage,
    mask: PILImage,
    foreground_threshold: number,
    background_threshold: number,
    erode_structure_size: number,
): Promise<PILImage> {
    /**
     * Perform alpha matting on an image using a given mask and threshold values.
     *
     * This function takes a PIL image `img` and a PIL image `mask` as input, along with
     * the `foreground_threshold` and `background_threshold` values used to determine
     * foreground and background pixels. The `erode_structure_size` parameter specifies
     * the size of the erosion structure to be applied to the mask.
     *
     * The function returns a PIL image representing the cutout of the foreground object
     * from the original image.
     */
    console.log('alpha_matting_cutout: Starting');
    
    if (img.mode === "RGBA" || img.mode === "CMYK") {
        img = img.convert("RGB");
    }    console.log('alpha_matting_cutout: Converting to arrays');
    const img_array = np.asarray(img);
    const mask_array = np.asarray(mask);
    console.log('alpha_matting_cutout: Image array shape:', img_array.shape);
    console.log('alpha_matting_cutout: Mask array shape:', mask_array.shape);

    // Ensure we have RGB (3-channel) image for MODNet
    let img_array_rgb = img_array;
    if (img_array.shape.length === 3 && img_array.shape[2] === 4) {
        // Convert RGBA to RGB by dropping alpha channel
        console.log('alpha_matting_cutout: Converting RGBA to RGB');
        const [height, width] = img_array.shape;
        const rgbData = new Float32Array(height * width * 3);
        
        // Copy RGB channels, skip alpha
        for (let i = 0; i < height * width; i++) {
            rgbData[i * 3] = img_array.data[i * 4];     // R
            rgbData[i * 3 + 1] = img_array.data[i * 4 + 1]; // G
            rgbData[i * 3 + 2] = img_array.data[i * 4 + 2]; // B
        }
        
        img_array_rgb = new (img_array.constructor as any)(rgbData, [height, width, 3], img_array.dtype);
        console.log('alpha_matting_cutout: RGB Image array shape:', img_array_rgb.shape);
    }

    const is_foreground = mask_array.gt(foreground_threshold);
    const is_background = mask_array.lt(background_threshold);

    let structure: unknown = null;
    if (erode_structure_size > 0) {
        structure = np.ones(
            [erode_structure_size, erode_structure_size],
            { dtype: np.uint8 }
        );
    }

    console.log('alpha_matting_cutout: Applying erosion');
    const is_foreground_eroded = binary_erosion(is_foreground, { structure: structure });
    const is_background_eroded = binary_erosion(is_background, { structure: structure, border_value: 1 });

    console.log('alpha_matting_cutout: Creating trimap');
    const trimap = np.full(mask_array.shape, { dtype: np.uint8, fill_value: 128 });
    trimap.setValues(is_foreground_eroded, 255);
    trimap.setValues(is_background_eroded, 0);    console.log('alpha_matting_cutout: Normalizing images');
    const img_normalized = img_array_rgb.divide(255.0);
    const trimap_normalized = trimap.divide(255.0);

    console.log('alpha_matting_cutout: Calling estimate_alpha_cf');
    const alpha = await estimate_alpha_cf(img_normalized, trimap_normalized);
    console.log('alpha_matting_cutout: Alpha estimation completed, shape:', alpha.shape);
    
    console.log('alpha_matting_cutout: Calling estimate_foreground_ml');
    const foreground = estimate_foreground_ml(img_normalized, alpha) as NumpyArray;
    console.log('alpha_matting_cutout: Foreground estimation completed, shape:', foreground.shape);
    
    console.log('alpha_matting_cutout: Stacking images');
    let cutout = stack_images(foreground, alpha);
    console.log('alpha_matting_cutout: Stack completed, shape:', cutout.shape);

    console.log('alpha_matting_cutout: Converting to uint8');
    cutout = np.clip(cutout.multiply(255), 0, 255).astype(np.uint8);
    const cutoutImage = Image.fromarray(cutout);

    console.log('alpha_matting_cutout: Completed successfully');
    return cutoutImage;
}



function putalpha_cutout(img: PILImage, mask: PILImage): PILImage {
    /**
     * Apply the specified mask to the image as an alpha cutout.
     *
     * Args:
     *     img (PILImage): The image to be modified.
     *     mask (PILImage): The mask to be applied.
     *
     * Returns:
     *     PILImage: The modified image with the alpha cutout applied.
     */
    img.putalpha(mask);
    return img;
}

function get_concat_v_multi(imgs: PILImage[]): PILImage {
    /**
     * Concatenate multiple images vertically.
     *
     * Args:
     *     imgs (PILImage[]): The list of images to be concatenated.
     *
     * Returns:
     *     PILImage: The concatenated image.
     */
    let pivot = imgs.shift();
    if (!pivot) {
        throw new Error("No images provided");
    }
    for (const im of imgs) {
        pivot = get_concat_v(pivot, im);
    }
    return pivot;
}

function get_concat_v(img1: PILImage, img2: PILImage): PILImage {
    /**
     * Concatenate two images vertically.
     *
     * Args:
     *     img1 (PILImage): The first image.
     *     img2 (PILImage): The second image to be concatenated below the first image.
     *
     * Returns:
     *     PILImage: The concatenated image.
     */
    const dst = Image.new("RGBA", [img1.width, img1.height + img2.height]);
    dst.paste(img1, [0, 0]);
    dst.paste(img2, [0, img1.height]);
    return dst;
}

function post_process(mask: NumpyArray): NumpyArray {
    /**
     * Post Process the mask for a smooth boundary by applying Morphological Operations
     * Research based on paper: https://www.sciencedirect.com/science/article/pii/S2352914821000757
     * args:
     *     mask: Binary Numpy Mask
     */
    let processed_mask = morphologyEx(mask, MORPH_OPEN, kernel);
    processed_mask = GaussianBlur(processed_mask, [5, 5], { sigmaX: 2, sigmaY: 2, borderType: BORDER_DEFAULT });
    processed_mask = np.where(processed_mask.lt(127), 0, 255).astype(np.uint8);
    return processed_mask;
}





async function download_models(models: string[]): Promise<void> {
    /**
     * Download models for image processing.
     */
    if (models.length === 0) {
        console.log("No models specified, downloading all models");
        models = [...sessions_names];
    }    for (const model of models) {
        const session_class = sessions[model];
        if (session_class === undefined) {
            console.log(`Error: no model found: ${model}`);
            if (typeof process !== 'undefined' && process.exit) {
                process.exit(1);
            }
            return;
        } else {
            console.log(`Downloading model: ${model}`);
            try {
                await session_class.download_models();
            } catch (e) {
                console.log(`Error downloading model: ${e}`);
            }
        }
    }
}

async function remove(
    data: Uint8Array | PILImage | NumpyArray | (Uint8Array | PILImage | NumpyArray)[],
    alpha_matting = true,
    alpha_matting_foreground_threshold = 240,
    alpha_matting_background_threshold = 10,
    alpha_matting_erode_size = 10,
    session?: BaseSession,
    only_mask = false,
    post_process_mask = false,
    force_return_bytes = false,
    ...args: unknown[]
): Promise<Uint8Array | PILImage | NumpyArray | (Uint8Array | PILImage | NumpyArray)[]> {
    /**
     * Remove the background from an input image or batch of images.
     *
     * This function processes images through segmentation and optionally alpha matting for high-quality background removal.
     * For batch processing, all images are first processed through the segmentation model, then through alpha matting if enabled.
     * This approach is more efficient than processing each image individually through both models.
     * 
     * Parameters:
     *     data (Uint8Array | PILImage | NumpyArray | (Uint8Array | PILImage | NumpyArray)[]): The input image data or batch of images.
     *     alpha_matting (boolean, optional): Whether to use alpha matting for better edge quality. Defaults to true.
     *     alpha_matting_foreground_threshold (number, optional): Foreground threshold for alpha matting. Defaults to 240.
     *     alpha_matting_background_threshold (number, optional): Background threshold for alpha matting. Defaults to 10.
     *     alpha_matting_erode_size (number, optional): Erosion size for alpha matting. Defaults to 10.
     *     session (BaseSession?, optional): A session object for the model. Defaults to undefined.
     *     only_mask (boolean, optional): Flag indicating whether to return only the binary masks. Defaults to false.
     *     post_process_mask (boolean, optional): Flag indicating whether to post-process the masks. Defaults to false.
     *     force_return_bytes (boolean, optional): Flag indicating whether to return the cutout image as bytes. Defaults to false.
     *     ...args (unknown[]): Additional arguments.
     *
     * Returns:
     *     Uint8Array | PILImage | NumpyArray | (Uint8Array | PILImage | NumpyArray)[]: The cutout image(s) with the background removed. Returns an array if input was an array.
     */// Check if input is a batch (array)
    const isBatch = Array.isArray(data);
    const inputArray = isBatch ? data as (Uint8Array | PILImage | NumpyArray)[] : [data as (Uint8Array | PILImage | NumpyArray)];
    
    // Convert all inputs to PIL images and determine return types
    const images: PILImage[] = [];
    const returnTypes: ReturnType[] = [];
    
    for (const item of inputArray) {
        let return_type: ReturnType;
        let img: PILImage;
        
        if (item instanceof Uint8Array || force_return_bytes) {
            return_type = ReturnType.BYTES;
            const bytesIO = new io.BytesIO(item as Uint8Array);
            const blob = bytesIO.toBlob();
            img = await Image.openAsync(blob);
        } else if (item && typeof item === 'object' && 'width' in item && 'height' in item && 'mode' in item) {
            return_type = ReturnType.PILLOW;
            img = item as PILImage;
        } else if (item && typeof item === 'object' && 'shape' in item && 'data' in item) {
            return_type = ReturnType.NDARRAY;
            img = Image.fromarray(item as NumpyArray);
        } else {
            throw new Error(
                `Input type ${typeof item} is not supported. Try using force_return_bytes=true to force bytes output`
            );
        }        
        // Images are now ready for processing
        images.push(img);
        returnTypes.push(return_type);
    }

    // Initialize session if needed
    if (session === undefined) {
        session = new_session("u2net", ...args);
        await session.initialize();
    }

    // BATCH PROCESSING: First pass - get all masks from segmentation model
    console.log(`Processing batch of ${images.length} images through segmentation model...`);
    const allMasks: PILImage[][] = [];
    
    for (const img of images) {
        const masks = await session.predict(img, ...args);
        allMasks.push(masks);
    }
    console.log('Segmentation completed for all images');

    // BATCH PROCESSING: Second pass - process cutouts (including alpha matting if enabled)
    const allCutouts: PILImage[][] = [];
    
    for (let i = 0; i < images.length; i++) {
        const img = images[i];
        const masks = allMasks[i];
        const cutouts: PILImage[] = [];

        for (let mask of masks) {
            if (post_process_mask) {
                mask = Image.fromarray(post_process(np.array(mask)));
            }            let cutout: PILImage;

            if (only_mask) {
                console.log(`Image ${i + 1}: Using only_mask mode`);
                cutout = mask;
            } else if (alpha_matting) {
                console.log(`Image ${i + 1}: Using alpha_matting_cutout`);
                try {
                    cutout = await alpha_matting_cutout(
                        img,
                        mask,
                        alpha_matting_foreground_threshold,
                        alpha_matting_background_threshold,
                        alpha_matting_erode_size,
                    );
                } catch (error) {
                    console.error(`Image ${i + 1}: Alpha matting failed with error:`, error);
                    console.log(`Image ${i + 1}: Alpha matting failed, falling back to putalpha_cutout`);
                    cutout = putalpha_cutout(img, mask);
                }
            } else {
                console.log(`Image ${i + 1}: Using putalpha_cutout (no alpha matting)`);
                cutout = putalpha_cutout(img, mask);
            }

            cutouts.push(cutout);
        }
        
        allCutouts.push(cutouts);
    }

    // Process final results
    const results: (Uint8Array | PILImage | NumpyArray)[] = [];
    
    for (let i = 0; i < images.length; i++) {
        const img = images[i];
        const cutouts = allCutouts[i];
        const return_type = returnTypes[i];
        
        console.log(`Image ${i + 1}: Cutouts processed:`, cutouts.length);
        let cutout = img;
        if (cutouts.length > 0) {
            cutout = get_concat_v_multi(cutouts);
            console.log(`Image ${i + 1}: Final cutout created, size:`, cutout.width, 'x', cutout.height);
        }        // Final cutout is ready

        // Convert to appropriate return type
        if (ReturnType.PILLOW === return_type) {
            results.push(cutout);
        } else if (ReturnType.NDARRAY === return_type) {
            results.push(np.asarray(cutout));
        } else {
            // ReturnType.BYTES
            const bio = new io.BytesIO();
            if (cutout && 'saveAsync' in cutout && typeof cutout.saveAsync === 'function') {
                await cutout.saveAsync(bio, "PNG");
            } else {
                cutout.save(bio, "PNG");
            }
            bio.seek(0);
            results.push(bio.read());
        }
    }    // Return single result or batch based on input
    if (isBatch) {
        return results;
    } else {
        return results[0];
    }
}

async function remove_video(
    data: File | Blob,
    alpha_matting = true,
    alpha_matting_foreground_threshold = 240,
    alpha_matting_background_threshold = 10,
    alpha_matting_erode_size = 10,
    session?: BaseSession,
    only_mask = false,
    post_process_mask = false,
    onProgress?: (current: number, total: number) => void,
    ...args: unknown[]
): Promise<string> {
    /**
     * Remove the background from a video file.
     *
     * This function processes each frame of a video through background removal,
     * then reassembles them into a new video file.
     * 
     * Parameters:
     *     data (File | Blob): The input video file.
     *     alpha_matting (boolean, optional): Whether to use alpha matting for better edge quality. Defaults to true.
     *     alpha_matting_foreground_threshold (number, optional): Foreground threshold for alpha matting. Defaults to 240.
     *     alpha_matting_background_threshold (number, optional): Background threshold for alpha matting. Defaults to 10.
     *     alpha_matting_erode_size (number, optional): Erosion size for alpha matting. Defaults to 10.
     *     session (BaseSession?, optional): A session object for the model. Defaults to undefined.
     *     only_mask (boolean, optional): Flag indicating whether to return only the binary masks. Defaults to false.
     *     post_process_mask (boolean, optional): Flag indicating whether to post-process the masks. Defaults to false.
     *     onProgress (function?, optional): Callback function to report progress. Receives (current, total) frame numbers.
     *     ...args (unknown[]): Additional arguments.
     *
     * Returns:
     *     string: A blob URL pointing to the processed video file.
     */
    console.log('remove_video: Starting video processing');
    
    // Initialize session if needed
    if (session === undefined) {
        session = new_session("u2net", ...args);
        await session.initialize();
    }

    // Create video processor
    const processor = new VideoFrameProcessor(data);
    await processor.init();
    console.log('remove_video: Video processor initialized');

    let frameIndex = 0;
    let frame: Uint8Array | null;

    // Process each frame
    while ((frame = await processor.next()) !== null) {
        frameIndex++;
        console.log(`remove_video: Processing frame ${frameIndex}`);
          // Process frame through background removal
        const processedFrame = await remove(
            frame,
            alpha_matting,
            alpha_matting_foreground_threshold,
            alpha_matting_background_threshold,
            alpha_matting_erode_size,
            session,
            only_mask,
            post_process_mask,
            true // force_return_bytes
        ) as Uint8Array;

        // Add processed frame to video
        await processor.push(processedFrame);
        
        // Report progress if callback provided
        if (onProgress) {
            onProgress(frameIndex, frameIndex); // We don't know total until we finish
        }
    }

    console.log(`remove_video: Processed ${frameIndex} frames, exporting video`);
    
    // Export the final video
    const videoUrl = await processor.exportFinalVideo();
    console.log('remove_video: Video processing completed');
    
    return videoUrl;
}

export {
    ReturnType,
    alpha_matting_cutout,
    putalpha_cutout,
    get_concat_v_multi,
    get_concat_v,
    post_process,
    download_models,
    remove,
    remove_video,
};

// Also export session factory for convenience
export { new_session, new_session_async } from './session_factory';
