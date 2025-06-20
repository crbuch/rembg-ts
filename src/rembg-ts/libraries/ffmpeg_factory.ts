import { FFmpeg } from "@ffmpeg/ffmpeg";
import { toBlobURL } from "@ffmpeg/util";

const ffmpeg = new FFmpeg();

let isLoaded = false;
export async function initFFmpeg() {
  if (isLoaded) return;

  const baseURL = "https://unpkg.com/@ffmpeg/core@0.12.6/dist/umd";

  // Use toBlobURL() to wrap the CDN files properly for browser loading
  await ffmpeg.load({
    coreURL: await toBlobURL(`${baseURL}/ffmpeg-core.js`, "text/javascript"),
    wasmURL: await toBlobURL(`${baseURL}/ffmpeg-core.wasm`, "application/wasm"),
    // if you need multithreading support:
    // workerURL: await toBlobURL(`${baseURL}/ffmpeg-core.worker.js`, 'text/javascript'),
  });
  isLoaded = true;
}

export function isFFmpegLoaded(): boolean {
  return isLoaded;
}

export default ffmpeg;
