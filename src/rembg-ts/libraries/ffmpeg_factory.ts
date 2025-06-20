import { FFmpeg } from '@ffmpeg/ffmpeg';
import { toBlobURL } from '@ffmpeg/util';

const ffmpeg = new FFmpeg();

let isLoaded = false;
export async function initFFmpeg(){
    if (isLoaded ) return;

    await ffmpeg.load({
      coreURL: await toBlobURL('https://unpkg.com/@ffmpeg/core@0.12.10/dist/ffmpeg-core.js', 'text/javascript'),
      wasmURL: await toBlobURL('https://unpkg.com/@ffmpeg/core@0.12.10/dist/ffmpeg-core.wasm', 'application/wasm'),
    });
    isLoaded = true;
}

export function isFFmpegLoaded(): boolean {
    return isLoaded;
}



export default ffmpeg;