import * as os from './libraries/os';
import * as ort from 'onnxruntime-web';

import { sessions_class } from './libraries/sessions';
import { BaseSession } from './sessions/base';

export function new_session(
    model_name = "u2net", 
    downloadProgressCallback?: (loaded: number, total: number) => void,
    ...args: unknown[]
): BaseSession {
    /**
     * Create a new session object based on the specified model name.
     *
     * This function searches for the session class based on the model name in the 'sessions_class' list.
     * It then creates an instance of the session class with the provided arguments.
     * The 'sess_opts' object is created using the 'ort.InferenceSession.SessionOptions()' constructor.
     * If the 'OMP_NUM_THREADS' environment variable is set, the 'inter_op_num_threads' option of 'sess_opts' is set to its value.
     *
     * Note: The session is created but not initialized. Call session.initialize() before using predict().
     *
     * Parameters:
     *     model_name (string): The name of the model.
     *     ...args: Additional arguments.
     *
     * Raises:
     *     Error: If no session class with the given `model_name` is found.
     *
     * Returns:
     *     BaseSession: The created session object.
     */
    let session_class: typeof BaseSession | null = null;    for (const sc of sessions_class) {
        if (sc.getModelName() === model_name) {
            session_class = sc;
            break;
        }
    }

    if (session_class === null) {
        throw new Error(`No session class found for model '${model_name}'`);
    }

    const sess_opts: ort.InferenceSession.SessionOptions = {};

    if (os.getenv("OMP_NUM_THREADS") !== null) {
        const threads = parseInt(os.getenv("OMP_NUM_THREADS")!);
        sess_opts.interOpNumThreads = threads;
        sess_opts.intraOpNumThreads = threads;
    }

    return new session_class(model_name, sess_opts, downloadProgressCallback, ...args);
}

/**
 * Create and initialize a session asynchronously
 */
export async function new_session_async(
    model_name = "u2net", 
    downloadProgressCallback?: (loaded: number, total: number) => void,
    ...args: unknown[]
): Promise<BaseSession> {
    const session = new_session(model_name, downloadProgressCallback, ...args);
    await session.initialize();
    return session;
}
