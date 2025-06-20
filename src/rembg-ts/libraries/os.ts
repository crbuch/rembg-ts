/**
 * OS-like utilities for web environment
 * Implements a subset of Python os functionality needed for rembg
 */

export function getenv(name: string, defaultValue?: string | null): string | null {
    /**
     * Get an environment variable value.
     * In web environment, this could check localStorage, sessionStorage, or predefined config.
     * 
     * Parameters:
     *     name: Environment variable name
     *     defaultValue: Default value if not found
     * 
     * Returns:
     *     The environment variable value or default
     */
    
    // In a web environment, we might check:
    // 1. Predefined configuration object
    // 2. localStorage
    // 3. URL parameters
    // 4. Build-time environment variables
    
    // For now, return defaults for known variables
    const envVars: Record<string, string> = {
        'MODEL_CHECKSUM_DISABLED': '', // Empty string means not set
        'XDG_DATA_HOME': '', // Empty string means use default
        'U2NET_HOME': '', // Empty string means use default
    };
    
    const value = envVars[name];
    if (value !== undefined && value !== '') {
        return value;
    }
    
    return defaultValue ?? null;
}

export const path = {
    join(...paths: string[]): string {
        /**
         * Join path components using forward slashes (web-compatible).
         * 
         * Parameters:
         *     ...paths: Path components to join
         * 
         * Returns:
         *     Joined path string
         */
        return paths
            .filter(p => p && p.length > 0)
            .map(p => p.replace(/[/\\]+$/, '')) // Remove trailing slashes
            .join('/');
    },
    
    expanduser(path: string): string {
        /**
         * Expand ~ to user home directory.
         * In web environment, this might map to a default storage location.
         * 
         * Parameters:
         *     path: Path that may contain ~
         * 
         * Returns:
         *     Expanded path
         */
        if (path.startsWith('~')) {
            // In web environment, we can't access actual user home
            // Map to a default location or use browser storage path
            const homeReplacement = '/app-data'; // Virtual home directory
            return path.replace(/^~/, homeReplacement);
        }
        return path;
    }
};

const os = {
    getenv,
    path
};

export default os;
