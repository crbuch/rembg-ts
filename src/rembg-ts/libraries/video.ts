import { fetchFile } from "@ffmpeg/util";
import ffmpeg, { initFFmpeg, isFFmpegLoaded } from "./ffmpeg_factory";
import { type FFmpeg } from "@ffmpeg/ffmpeg";

export class VideoFrameProcessor {
  private ffmpeg: FFmpeg;
  private inputVideo: File | Blob;
  private frameIndex = 0;
  private extractedFrameNames: string[] = [];
  private pushedFrameCount = 0;
  private isReady = false;

  /**
   * Creates a new VideoFrameProcessor instance.
   * @param inputVideo A File or Blob containing the video to process.
   */
  constructor(inputVideo: File | Blob) {
    this.inputVideo = inputVideo;
    this.ffmpeg = ffmpeg;
  }

  /**
   * Initializes FFmpeg and extracts all frames from the input video.
   * Must be called before using `next()` or `push()`.
   */
  async init(): Promise<void> {
    if (!isFFmpegLoaded()) {
      console.log("FFmpeg not loaded, initializing...");
      await initFFmpeg();
    }

    const videoData = await fetchFile(this.inputVideo);
    await this.ffmpeg.writeFile("input.mp4", videoData);

    // Extract all frames
    await this.ffmpeg.exec(["-i", "input.mp4", "frame_%04d.png"]);

    // Gather frame names
    const files = await this.ffmpeg.listDir("/");
    this.extractedFrameNames = files
      .filter((f) => f.name.match(/^frame_\d{4}\.png$/))
      .map((f) => f.name)
      .sort();

    this.isReady = true;
  }

  /**
   * Retrieves the next extracted frame from the video as a Uint8Array.
   * Returns `null` when all frames have been consumed.
   * @returns The next video frame or null if finished.
   */
  async next(): Promise<Uint8Array | null> {
    if (!this.isReady)
      throw new Error("FFmpeg not initialized. Call init() first.");

    if (this.frameIndex >= this.extractedFrameNames.length) return null;

    const name = this.extractedFrameNames[this.frameIndex++];
    const frame = await this.ffmpeg.readFile(name, "binary");

    return frame as Uint8Array;
  }

  /**
   * Adds a frame to the output video.
   * Frames should be pushed in sequence and must be valid PNG image data.
   * @param frameData A Uint8Array representing a PNG frame.
   */
  async push(frameData: Uint8Array): Promise<void> {
    if (!this.isReady)
      throw new Error("FFmpeg not initialized. Call init() first.");

    const filename = `outframe_${String(this.pushedFrameCount).padStart(
      4,
      "0"
    )}.png`;
    await this.ffmpeg.writeFile(filename, frameData);
    this.pushedFrameCount++;
  }

  /**
   * Exports the final video composed of all pushed frames.
   * Returns a URL pointing to the generated MP4 file.
   * @returns A blob URL for the resulting video.
   */
  async exportFinalVideo(): Promise<string> {
    if (this.pushedFrameCount === 0) {
      throw new Error("No frames pushed. Use push() before exporting.");
    }

    await this.ffmpeg.exec([
      "-framerate",
      "30",
      "-i",
      "outframe_%04d.png",
      "-c:v",
      "libx264",
      "-pix_fmt",
      "yuv420p",
      "output.mp4",
    ]);

    const output = await this.ffmpeg.readFile("output.mp4", "binary");
    const blob = new Blob([(output as Uint8Array).buffer], {
      type: "video/mp4",
    });
    return URL.createObjectURL(blob);
  }
}