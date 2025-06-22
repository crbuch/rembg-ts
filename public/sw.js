/**
 * Service Worker for caching ONNX models
 * This ensures models are cached persistently across browser sessions
 */

const CACHE_NAME = 'rembg-models-v1';
const MODEL_URLS = [
  '/models/u2net.onnx',
  '/models/modnet_photographic_portrait_matting.onnx'
];

// Install event - pre-cache models
self.addEventListener('install', (event) => {
  console.log('Service Worker: Installing and pre-caching models...');
  
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('Service Worker: Caching models...');
        // Pre-cache all models during service worker installation
        return cache.addAll(MODEL_URLS.map(url => new Request(url, {
          cache: 'no-store' // Ensure we get fresh copies during install
        })));
      })
      .then(() => {
        console.log('Service Worker: All models cached successfully');
        // Force activation of new service worker
        return self.skipWaiting();
      })
      .catch((error) => {
        console.error('Service Worker: Error caching models:', error);
      })
  );
});

// Activate event
self.addEventListener('activate', (event) => {
  console.log('Service Worker: Activating...');
  
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          // Delete old caches
          if (cacheName !== CACHE_NAME) {
            console.log('Service Worker: Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => {
      // Take control of all clients immediately
      return self.clients.claim();
    })
  );
});

// Fetch event - serve models from cache
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  
  // Only handle model file requests
  if (url.pathname.startsWith('/models/') && url.pathname.endsWith('.onnx')) {
    console.log('Service Worker: Intercepting model request:', url.pathname);
    
    event.respondWith(
      caches.open(CACHE_NAME)
        .then((cache) => {
          return cache.match(event.request);
        })
        .then((cachedResponse) => {
          if (cachedResponse) {
            console.log('Service Worker: Serving model from cache:', url.pathname);
            return cachedResponse;
          }
          
          // If not in cache, fetch and cache it
          console.log('Service Worker: Model not in cache, fetching:', url.pathname);
          return fetch(event.request)
            .then((response) => {
              // Only cache successful responses
              if (response.status === 200) {
                const responseClone = response.clone();
                caches.open(CACHE_NAME)
                  .then((cache) => {
                    cache.put(event.request, responseClone);
                    console.log('Service Worker: Cached new model:', url.pathname);
                  });
              }
              return response;
            });
        })
        .catch((error) => {
          console.error('Service Worker: Error serving model:', error);
          // Fallback to network
          return fetch(event.request);
        })
    );
  }
});

// Message handling for manual cache operations
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'CACHE_MODELS') {
    console.log('Service Worker: Manual model caching requested');
    
    event.waitUntil(
      caches.open(CACHE_NAME)
        .then((cache) => {
          return cache.addAll(MODEL_URLS);
        })
        .then(() => {
          event.ports[0].postMessage({ success: true });
        })
        .catch((error) => {
          console.error('Service Worker: Error in manual caching:', error);
          event.ports[0].postMessage({ success: false, error: error.message });
        })
    );
  }
  
  if (event.data && event.data.type === 'CHECK_CACHE') {
    caches.open(CACHE_NAME)
      .then((cache) => {
        return Promise.all(
          MODEL_URLS.map(url => cache.match(url).then(response => ({
            url,
            cached: !!response
          })))
        );
      })
      .then((results) => {
        event.ports[0].postMessage({ type: 'CACHE_STATUS', results });
      });
  }
});
