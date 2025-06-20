/**
 * Main entry point for the web-compatible rembg library
 * 
 * This module exports all the functionality needed to perform background removal
 * in web applications, closely mirroring the Python rembg library structure.
 */

// Main background removal functionality
export {
    remove,
    new_session,
    new_session_async,
    ReturnType,
    alpha_matting_cutout,
    naive_cutout,
    putalpha_cutout,
    get_concat_v_multi,
    get_concat_v,
    post_process,
    apply_background_color,
    fix_image_orientation,
    download_models
} from './bg';

// PIL-like image processing
export {
    Image,
    ImageOps,
    WebPILImage,
    createImageFromPIL,
    createImageFromCanvas,
    isWebNativeImageType,
    isPILImage
} from './libraries/PIL';

export type { PILImage } from './libraries/PIL';


// Session management
export type { BaseSession } from './sessions/base';
export { sessions, sessions_names } from './libraries/sessions';

// Numpy-like operations
export * as np from './libraries/numpy';
export type { NumpyArray } from './libraries/numpy';

// I/O operations
export * as io from './libraries/io';

// OpenCV-like operations  
export * as cv2 from './libraries/cv2';

// Default export for convenience
export { remove as default } from './bg';
