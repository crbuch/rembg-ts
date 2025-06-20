/**
 * System utilities for web environment
 * Provides basic sys-like functionality
 */

export function exit(code: number): never {
    // In web environment, we can't actually exit the process
    // Instead, throw an error or handle gracefully
    throw new Error(`System exit requested with code: ${code}`);
}

const sys = {
    exit
};

export default sys;
