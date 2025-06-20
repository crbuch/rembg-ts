/**
 * IO utilities for web-based image processing
 * Provides file-like operations using Blob and ArrayBuffer, closely mirroring Python's io module
 */

export class BytesIO {
    private buffer: ArrayBuffer;
    private position = 0;
    private view: Uint8Array;
    private closed = false;

    constructor(initialData?: Uint8Array | ArrayBuffer) {
        if (initialData) {
            if (initialData instanceof ArrayBuffer) {
                this.buffer = initialData.slice(0);
                this.view = new Uint8Array(this.buffer);
            } else {
                this.buffer = initialData.buffer.slice(0) as ArrayBuffer;
                this.view = new Uint8Array(this.buffer);
            }
        } else {
            this.buffer = new ArrayBuffer(0);
            this.view = new Uint8Array(this.buffer);
        }
    }

    /**
     * Write data to the BytesIO buffer
     * Supports both synchronous Uint8Array and asynchronous Blob writing
     */
    write(data: Uint8Array | string): number {
        this._checkClosed();
        
        let bytes: Uint8Array;
        if (typeof data === 'string') {
            // Convert string to UTF-8 bytes
            bytes = new TextEncoder().encode(data);
        } else {
            bytes = data;
        }
        
        this.writeBytes(bytes);
        return bytes.length;
    }

    /**
     * Asynchronously write Blob data to the buffer
     */
    async writeBlob(blob: Blob): Promise<number> {
        this._checkClosed();
        const buffer = await blob.arrayBuffer();
        const bytes = new Uint8Array(buffer);
        this.writeBytes(bytes);
        return bytes.length;
    }

    private writeBytes(data: Uint8Array): void {
        const newSize = this.position + data.length;
        if (newSize > this.buffer.byteLength) {
            // Resize buffer with some extra capacity
            const newCapacity = Math.max(newSize, this.buffer.byteLength * 2);
            const newBuffer = new ArrayBuffer(newCapacity);
            const newView = new Uint8Array(newBuffer);
            newView.set(this.view);
            this.buffer = newBuffer;
            this.view = newView;
        }
        
        this.view.set(data, this.position);
        this.position += data.length;
    }

    /**
     * Read specified number of bytes from current position
     */
    read(size?: number): Uint8Array {
        this._checkClosed();
        
        if (size === undefined) {
            // Read all remaining data
            const result = this.view.slice(this.position, this.getSize());
            this.position = this.getSize();
            return result;
        }
        
        const endPos = Math.min(this.position + size, this.getSize());
        const result = this.view.slice(this.position, endPos);
        this.position = endPos;
        return result;
    }

    /**
     * Read one line (until newline or EOF)
     */
    readline(): Uint8Array {
        this._checkClosed();
        
        const newlineCode = 10; // '\n'
        const startPos = this.position;
        
        while (this.position < this.getSize() && this.view[this.position] !== newlineCode) {
            this.position++;
        }
        
        // Include the newline if found
        if (this.position < this.getSize()) {
            this.position++;
        }
        
        return this.view.slice(startPos, this.position);
    }

    /**
     * Read all lines
     */
    readlines(): Uint8Array[] {
        this._checkClosed();
        
        const lines: Uint8Array[] = [];
        while (this.position < this.getSize()) {
            const line = this.readline();
            if (line.length === 0) break;
            lines.push(line);
        }
        return lines;
    }

    /**
     * Seek to a specific position
     */
    seek(position: number, whence = 0): number {
        this._checkClosed();
        
        let newPos: number;
        switch (whence) {
            case 0: // SEEK_SET - absolute position
                newPos = position;
                break;
            case 1: // SEEK_CUR - relative to current position
                newPos = this.position + position;
                break;
            case 2: // SEEK_END - relative to end
                newPos = this.getSize() + position;
                break;
            default:
                throw new Error(`Invalid whence value: ${whence}`);
        }
        
        this.position = Math.max(0, Math.min(newPos, this.getSize()));
        return this.position;
    }

    /**
     * Get current position
     */
    tell(): number {
        this._checkClosed();
        return this.position;
    }

    /**
     * Get the entire buffer contents
     */
    getvalue(): Uint8Array {
        this._checkClosed();
        return this.view.slice(0, this.getSize());
    }

    /**
     * Get current size of the buffer
     */
    getSize(): number {
        return this.view.length;
    }

    /**
     * Truncate the buffer to specified size
     */
    truncate(size?: number): number {
        this._checkClosed();
        
        const newSize = size ?? this.position;
        if (newSize < 0) {
            throw new Error('Truncate size cannot be negative');
        }
        
        if (newSize < this.buffer.byteLength) {
            const newBuffer = new ArrayBuffer(newSize);
            const newView = new Uint8Array(newBuffer);
            newView.set(this.view.slice(0, newSize));
            this.buffer = newBuffer;
            this.view = newView;
        }
        
        this.position = Math.min(this.position, newSize);
        return newSize;
    }

    /**
     * Flush the buffer (no-op for BytesIO but included for compatibility)
     */
    flush(): void {
        this._checkClosed();
        // No-op for BytesIO
    }

    /**
     * Close the BytesIO
     */
    close(): void {
        this.closed = true;
    }

    /**
     * Check if the BytesIO is closed
     */
    get isClosed(): boolean {
        return this.closed;
    }

    /**
     * Convert to Blob
     */
    toBlob(type = 'application/octet-stream'): Blob {
        this._checkClosed();
        return new Blob([this.getvalue()], { type });
    }

    /**
     * Convert to base64 string
     */
    toBase64(): string {
        this._checkClosed();
        const bytes = this.getvalue();
        let binary = '';
        for (let i = 0; i < bytes.length; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
    }

    /**
     * Convert to data URL
     */
    toDataURL(mimeType = 'application/octet-stream'): string {
        return `data:${mimeType};base64,${this.toBase64()}`;
    }    private _checkClosed(): void {
        if (this.closed) {
            throw new Error('I/O operation on closed BytesIO');
        }
    }
}

/**
 * StringIO class for text-based I/O operations
 * Mimics Python's io.StringIO
 */
export class StringIO {
    private buffer = '';
    private position = 0;
    private closed = false;

    constructor(initialValue = '') {
        this.buffer = initialValue;
    }

    write(data: string): number {
        this._checkClosed();
        this.buffer = this.buffer.slice(0, this.position) + data + this.buffer.slice(this.position);
        this.position += data.length;
        return data.length;
    }

    read(size?: number): string {
        this._checkClosed();
        
        if (size === undefined) {
            const result = this.buffer.slice(this.position);
            this.position = this.buffer.length;
            return result;
        }
        
        const endPos = Math.min(this.position + size, this.buffer.length);
        const result = this.buffer.slice(this.position, endPos);
        this.position = endPos;
        return result;
    }

    readline(): string {
        this._checkClosed();
        
        const startPos = this.position;
        const newlineIndex = this.buffer.indexOf('\n', this.position);
        
        if (newlineIndex === -1) {
            this.position = this.buffer.length;
            return this.buffer.slice(startPos);
        } else {
            this.position = newlineIndex + 1;
            return this.buffer.slice(startPos, this.position);
        }
    }

    readlines(): string[] {
        this._checkClosed();
        
        const lines: string[] = [];
        while (this.position < this.buffer.length) {
            const line = this.readline();
            if (line.length === 0) break;
            lines.push(line);
        }
        return lines;
    }

    seek(position: number, whence = 0): number {
        this._checkClosed();
        
        let newPos: number;
        switch (whence) {
            case 0: // SEEK_SET
                newPos = position;
                break;
            case 1: // SEEK_CUR
                newPos = this.position + position;
                break;
            case 2: // SEEK_END
                newPos = this.buffer.length + position;
                break;
            default:
                throw new Error(`Invalid whence value: ${whence}`);
        }
        
        this.position = Math.max(0, Math.min(newPos, this.buffer.length));
        return this.position;
    }

    tell(): number {
        this._checkClosed();
        return this.position;
    }

    getvalue(): string {
        this._checkClosed();
        return this.buffer;
    }

    truncate(size?: number): number {
        this._checkClosed();
        
        const newSize = size ?? this.position;
        if (newSize < 0) {
            throw new Error('Truncate size cannot be negative');
        }
        
        this.buffer = this.buffer.slice(0, newSize);
        this.position = Math.min(this.position, newSize);
        return newSize;
    }

    flush(): void {
        this._checkClosed();
        // No-op for StringIO
    }

    close(): void {
        this.closed = true;
    }

    get isClosed(): boolean {
        return this.closed;
    }

    private _checkClosed(): void {
        if (this.closed) {
            throw new Error('I/O operation on closed StringIO');
        }
    }
}

/**
 * Web File I/O utilities for handling browser File objects and downloads
 */
export class WebFileIO {
    /**
     * Read a File object as text
     */
    static async readText(file: File, encoding = 'utf-8'): Promise<string> {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result as string);
            reader.onerror = () => reject(reader.error);
            reader.readAsText(file, encoding);
        });
    }

    /**
     * Read a File object as ArrayBuffer
     */
    static async readArrayBuffer(file: File): Promise<ArrayBuffer> {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result as ArrayBuffer);
            reader.onerror = () => reject(reader.error);
            reader.readAsArrayBuffer(file);
        });
    }

    /**
     * Read a File object as Uint8Array
     */
    static async readBytes(file: File): Promise<Uint8Array> {
        const buffer = await this.readArrayBuffer(file);
        return new Uint8Array(buffer);
    }

    /**
     * Read a File object as data URL
     */
    static async readDataURL(file: File): Promise<string> {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result as string);
            reader.onerror = () => reject(reader.error);
            reader.readAsDataURL(file);
        });
    }

    /**
     * Create a downloadable file from data
     */
    static downloadFile(data: Uint8Array | string | Blob, filename: string, mimeType?: string): void {
        let blob: Blob;
        
        if (data instanceof Blob) {
            blob = data;
        } else if (typeof data === 'string') {
            blob = new Blob([data], { type: mimeType || 'text/plain' });
        } else {
            blob = new Blob([data], { type: mimeType || 'application/octet-stream' });
        }
        
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }

    /**
     * Save BytesIO to file download
     */
    static downloadBytesIO(bytesIO: BytesIO, filename: string, mimeType?: string): void {
        const data = bytesIO.getvalue();
        this.downloadFile(data, filename, mimeType);
    }

    /**
     * Save StringIO to file download
     */
    static downloadStringIO(stringIO: StringIO, filename: string): void {
        const data = stringIO.getvalue();
        this.downloadFile(data, filename, 'text/plain');
    }

    /**
     * Create a File object from data
     */
    static createFile(data: Uint8Array | string | Blob, filename: string, options?: FilePropertyBag): File {
        let content: Uint8Array | string | Blob;
        
        if (data instanceof Blob) {
            content = data;
        } else {
            content = data;
        }
        
        return new File([content], filename, options);
    }
}

/**
 * Image I/O utilities specifically for web image processing
 */
export class ImageIO {
    /**
     * Convert HTMLImageElement to BytesIO
     */
    static async imageElementToBytesIO(img: HTMLImageElement, format = 'image/png'): Promise<BytesIO> {
        const canvas = document.createElement('canvas');
        canvas.width = img.width || img.naturalWidth;
        canvas.height = img.height || img.naturalHeight;
        
        const ctx = canvas.getContext('2d');
        if (!ctx) throw new Error('Could not get 2D context');
        
        ctx.drawImage(img, 0, 0);
        
        const blob = await new Promise<Blob>((resolve, reject) => {
            canvas.toBlob((blob) => {
                if (blob) resolve(blob);
                else reject(new Error('Failed to convert canvas to blob'));
            }, format);
        });
        
        const arrayBuffer = await blob.arrayBuffer();
        const uint8Array = new Uint8Array(arrayBuffer);
        return new BytesIO(uint8Array);
    }

    /**
     * Convert Canvas to BytesIO
     */
    static async canvasToBytesIO(canvas: HTMLCanvasElement, format = 'image/png'): Promise<BytesIO> {
        const blob = await new Promise<Blob>((resolve, reject) => {
            canvas.toBlob((blob) => {
                if (blob) resolve(blob);
                else reject(new Error('Failed to convert canvas to blob'));
            }, format);
        });
        
        const arrayBuffer = await blob.arrayBuffer();
        const uint8Array = new Uint8Array(arrayBuffer);
        return new BytesIO(uint8Array);
    }

    /**
     * Convert BytesIO to HTMLImageElement
     */
    static async bytesIOToImageElement(bytesIO: BytesIO): Promise<HTMLImageElement> {
        const blob = bytesIO.toBlob('image/png');
        const url = URL.createObjectURL(blob);
        
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                URL.revokeObjectURL(url);
                resolve(img);
            };
            img.onerror = () => {
                URL.revokeObjectURL(url);
                reject(new Error('Failed to load image'));
            };
            img.src = url;
        });
    }

    /**
     * Convert File to HTMLImageElement
     */
    static async fileToImageElement(file: File): Promise<HTMLImageElement> {
        const dataURL = await WebFileIO.readDataURL(file);
        
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = () => reject(new Error('Failed to load image from file'));
            img.src = dataURL;
        });
    }
}

/**
 * Seek constants (matching Python's io module)
 */
export const SEEK_SET = 0;
export const SEEK_CUR = 1;
export const SEEK_END = 2;

/**
 * Helper functions for common I/O operations
 */
export function open_bytes(initialData?: Uint8Array | ArrayBuffer): BytesIO {
    return new BytesIO(initialData);
}

export function open_string(initialValue?: string): StringIO {
    return new StringIO(initialValue);
}

/**
 * Convert various input types to BytesIO for uniform processing
 */
export async function to_bytes_io(input: File | Blob | Uint8Array | ArrayBuffer | string): Promise<BytesIO> {
    if (input instanceof File || input instanceof Blob) {
        const arrayBuffer = await input.arrayBuffer();
        return new BytesIO(new Uint8Array(arrayBuffer));
    } else if (input instanceof Uint8Array) {
        return new BytesIO(input);
    } else if (input instanceof ArrayBuffer) {
        return new BytesIO(input);
    } else if (typeof input === 'string') {
        const encoder = new TextEncoder();
        const bytes = encoder.encode(input);
        return new BytesIO(bytes);
    } else {
        throw new Error(`Unsupported input type: ${typeof input}`);
    }
}

/**
 * Convert BytesIO back to various output formats
 */
export function from_bytes_io(bytesIO: BytesIO, outputType: 'blob'): Blob;
export function from_bytes_io(bytesIO: BytesIO, outputType: 'uint8array'): Uint8Array;
export function from_bytes_io(bytesIO: BytesIO, outputType: 'arraybuffer'): ArrayBuffer;
export function from_bytes_io(bytesIO: BytesIO, outputType: 'base64'): string;
export function from_bytes_io(bytesIO: BytesIO, outputType: 'dataurl', mimeType?: string): string;
export function from_bytes_io(bytesIO: BytesIO, outputType: string, mimeType?: string): Blob | Uint8Array | ArrayBuffer | string {
    switch (outputType) {
        case 'blob':
            return bytesIO.toBlob();
        case 'uint8array':
            return bytesIO.getvalue();        case 'arraybuffer':
            return bytesIO.getvalue().buffer.slice(0) as ArrayBuffer;
        case 'base64':
            return bytesIO.toBase64();
        case 'dataurl':
            return bytesIO.toDataURL(mimeType);
        default:
            throw new Error(`Unsupported output type: ${outputType}`);
    }
}

const ioModule = {
    BytesIO,
    StringIO,
    WebFileIO,
    ImageIO,
    SEEK_SET,
    SEEK_CUR,
    SEEK_END,
    open_bytes,
    open_string,
    to_bytes_io,
    from_bytes_io
};

export default ioModule;
