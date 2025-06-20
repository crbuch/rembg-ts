/**
 * Type casting utility for TypeScript
 * Provides runtime type assertion functionality
 */

export function cast<T>(value: unknown): T {
    return value as T;
}

export default cast;
