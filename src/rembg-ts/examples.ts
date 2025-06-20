/**
 * Examples demonstrating how to use the rembg-ts library
 * 
 * This file contains various examples showing different ways to use
 * the background removal functionality, closely mirroring the Python
 * rembg library usage patterns.
 */

import {
    remove,
    new_session_async,
    Image,
    type PILImage
} from './index';
import type { NumpyArray } from './libraries/numpy';
import * as io from './libraries/io';
import * as np from './libraries/numpy';

/**
 * Example 1: Remove background from input as Uint8Array, output as Uint8Array
 */
async function example_bytes_to_bytes() {
    console.log("Example 1: Bytes to Bytes");

    // Load an image as bytes (in a real application, this might come from a file input)
    const response = await fetch('/path/to/input.png');
    const inputBytes = new Uint8Array(await response.arrayBuffer());    // Remove background (alpha matting is now always enabled)
    const outputBytes = await remove(inputBytes) as Uint8Array;

    // The outputBytes can now be used to create a blob, save to file, etc.
    const blob = new Blob([outputBytes], { type: 'image/png' });
    const url = URL.createObjectURL(blob);
    console.log('Output image URL:', url);
}

/**
 * Example 2: Remove background using PIL-like image objects
 */
async function example_pil_to_pil() {
    console.log("Example 2: PIL to PIL");

    // Load image as PIL Image
    const response = await fetch('/path/to/input.png');
    const blob = await response.blob();
    const inputImage = Image.open(blob);    // Remove background (alpha matting is now always enabled)
    const outputImage = await remove(inputImage) as PILImage;

    // Save the result
    const outputBytes = new io.BytesIO();
    outputImage.save(outputBytes, 'PNG');

    console.log('Processed PIL image:', outputImage);
}

/**
 * Example 3: Remove background using numpy arrays
 */
async function example_numpy_to_numpy() {
    console.log("Example 3: Numpy to Numpy");

    // Load image as numpy array (in practice, this might come from canvas data)
    const response = await fetch('/path/to/input.png');
    const blob = await response.blob();
    const img = Image.open(blob);
    const inputArray = np.asarray(img);    // Remove background (alpha matting is now always enabled)
    const outputArray = await remove(inputArray) as NumpyArray;
    // Convert back to image for display
    const outputImage = Image.fromarray(outputArray);
    console.log('Processed numpy array shape:', outputArray.shape);
    console.log('Converted back to image:', outputImage);
}

/**
 * Example 4: Using alpha matting for better edge quality
 */
async function example_alpha_matting() {
    console.log("Example 4: Alpha Matting");

    const response = await fetch('/path/to/input.png');
    const blob = await response.blob();
    const inputImage = Image.open(blob);    // Remove background with custom alpha matting parameters
    const outputImage = await remove(
        inputImage,
        240,   // alpha_matting_foreground_threshold
        10,    // alpha_matting_background_threshold
        10     // alpha_matting_erode_size
    ) as PILImage;

    console.log('Alpha matting result:', outputImage);
}

/**
 * Example 5: Using a specific model session
 */
async function example_with_custom_session() {
    console.log("Example 5: Custom Session");

    // Create a specific session (u2netp for faster processing)
    const session = await new_session_async("u2netp");

    const response = await fetch('/path/to/input.png');
    const blob = await response.blob();
    const inputImage = Image.open(blob);    // Use the custom session
    const outputImage = await remove(
        inputImage,
        240,   // alpha_matting_foreground_threshold
        10,    // alpha_matting_background_threshold
        10,    // alpha_matting_erode_size
        session // Use our custom session
    ) as PILImage;

    console.log('Custom session result:', outputImage);
}

/**
 * Example 6: Get only the mask without applying it
 */
async function example_mask_only() {
    console.log("Example 6: Mask Only");

    const response = await fetch('/path/to/input.png');
    const blob = await response.blob();
    const inputImage = Image.open(blob);    // Get only the mask
    const mask = await remove(
        inputImage,
        240,   // alpha_matting_foreground_threshold
        10,    // alpha_matting_background_threshold
        10,    // alpha_matting_erode_size
        undefined, // session
        true   // only_mask
    ) as PILImage;

    console.log('Mask result:', mask);
}

/**
 * Example 7: Using post-processing for smoother masks
 */
async function example_with_post_processing() {
    console.log("Example 7: Post Processing");

    const response = await fetch('/path/to/input.png');
    const blob = await response.blob();
    const inputImage = Image.open(blob);

    // Remove background with post-processing enabled for smoother masks
    const outputImage = await remove(
        inputImage,
        240,   // alpha_matting_foreground_threshold
        10,    // alpha_matting_background_threshold
        10,    // alpha_matting_erode_size
        undefined, // session
        false, // only_mask
        true   // post_process_mask - enables morphological operations for smoother edges
    ) as PILImage;

    console.log('Post-processed result:', outputImage);
}

/**
 * Example 8: Force return as bytes regardless of input type
 */
async function example_force_bytes() {
    console.log("Example 8: Force Bytes Output");

    const response = await fetch('/path/to/input.png');
    const blob = await response.blob();
    const inputImage = Image.open(blob);    // Force output as bytes even though input is PIL image
    const outputBytes = await remove(
        inputImage,
        240,   // alpha_matting_foreground_threshold
        10,    // alpha_matting_background_threshold
        10,    // alpha_matting_erode_size
        undefined, // session
        false, // only_mask
        false, // post_process_mask
        true   // force_return_bytes
    ) as Uint8Array;

    console.log('Forced bytes output length:', outputBytes.length);
}

/**
 * Example 9: Batch processing multiple images in one call
 */
async function example_batch_processing() {
    console.log("Example 9: Batch Processing");

    // Load multiple images
    const imagePaths = ['/path/to/image1.png', '/path/to/image2.png', '/path/to/image3.png'];
    const inputImages: PILImage[] = [];

    for (const imagePath of imagePaths) {
        const response = await fetch(imagePath);
        const blob = await response.blob();
        const inputImage = Image.open(blob);
        inputImages.push(inputImage);
    }

    // Process all images in one batch call (more efficient)
    const outputImages = await remove(
        inputImages,
        240,   // alpha_matting_foreground_threshold
        10,    // alpha_matting_background_threshold
        10     // alpha_matting_erode_size
    ) as PILImage[];

    console.log(`Processed ${outputImages.length} images in batch:`, outputImages);
}

/**
 * Example 10: Processing multiple images efficiently with session reuse (individual calls)
 */
async function example_sequential_processing() {
    console.log("Example 10: Sequential Processing with Session Reuse");

    // Create session once for reuse
    const session = await new_session_async("u2net");

    const imagePaths = ['/path/to/image1.png', '/path/to/image2.png', '/path/to/image3.png'];

    for (const imagePath of imagePaths) {
        const response = await fetch(imagePath);
        const blob = await response.blob();
        const inputImage = Image.open(blob);

        // Reuse the same session for all images
        const outputImage = await remove(
            inputImage,
            240,   // alpha_matting_foreground_threshold
            10,    // alpha_matting_background_threshold
            10,    // alpha_matting_erode_size
            session // Reuse session
        ) as PILImage;

        console.log(`Processed ${imagePath}:`, outputImage);
    }
}

/**
 * Example 11: Post-processing the result (custom background, effects, etc.)
 */
async function example_post_processing() {
    console.log("Example 11: Post Processing");

    const response = await fetch('/path/to/input.png');
    const blob = await response.blob();
    const inputImage = Image.open(blob);

    // Remove background first
    const cutoutImage = await remove(
        inputImage,
        240,   // alpha_matting_foreground_threshold
        10,    // alpha_matting_background_threshold
        10     // alpha_matting_erode_size
    ) as PILImage;

    // Post-process: Add a solid color background
    const width = cutoutImage.width;
    const height = cutoutImage.height;
      // Create a new image with a colored background (RGBA for proper compositing)
    const backgroundImage = Image.new("RGBA", [width, height], [255, 0, 0, 255]); // Red background
    
    // Paste the cutout on top of the background
    backgroundImage.paste(cutoutImage, [0, 0]);

    console.log('Post-processed with red background:', backgroundImage);
}
async function example_error_handling() {
    console.log("Example 10: Error Handling");

    try {
        // This will fail if the image doesn't exist
        const response = await fetch('/nonexistent/image.png');
        if (!response.ok) {
            throw new Error('Failed to load image');
        }

        const blob = await response.blob();
        const inputImage = Image.open(blob);

        const outputImage = await remove(inputImage) as PILImage;
        console.log('Success:', outputImage);

    } catch (error) {
        console.error('Error processing image:', error);
    }
}




// Export all examples for use in other modules
export {
    example_bytes_to_bytes,
    example_pil_to_pil,
    example_numpy_to_numpy,
    example_alpha_matting,
    example_with_custom_session,
    example_mask_only,
    example_with_post_processing,
    example_force_bytes,
    example_batch_processing,
    example_sequential_processing,
    example_post_processing,
    example_error_handling
};

// Main function to run all examples
export async function runAllExamples() {
    console.log("=== rembg-ts Examples ===");

    try {
        await example_bytes_to_bytes();
        await example_pil_to_pil();
        await example_numpy_to_numpy();
        await example_alpha_matting();
        await example_with_custom_session();
        await example_mask_only();
        await example_with_post_processing();
        await example_force_bytes();
        await example_batch_processing();
        await example_sequential_processing();
        await example_post_processing();
        await example_error_handling();

        console.log("=== All examples completed ===");
    } catch (error) {
        console.error("Error running examples:", error);
    }
}

// If this file is run directly, execute all examples
if (typeof window !== 'undefined') {
    // Browser environment - you can call runAllExamples() from the console
    (window as any).rembgExamples = {
        runAllExamples,
        example_bytes_to_bytes,
        example_pil_to_pil,
        example_numpy_to_numpy,
        example_alpha_matting,
        example_with_custom_session,
        example_mask_only,
        example_with_post_processing,
        example_force_bytes,
        example_batch_processing,
        example_sequential_processing,
        example_post_processing,
        example_error_handling
    };

    console.log("rembg-ts examples loaded! Call rembgExamples.runAllExamples() to run all examples.");
}
