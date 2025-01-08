import threading
import time
from typing import Optional, List

class RateLimiter:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(RateLimiter, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize rate limiter settings."""
        self._request_times: List[float] = []
        self._request_lock = threading.Lock()
        self._max_requests = 3  # Configurable
        self._window_seconds = 60  # Configurable
        self._min_delay = 1.5  # Minimum delay between requests
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        return cls()
    
    def wait_if_needed(self, timeout: Optional[float] = None) -> bool:
        """Thread-safe rate limiting with timeout and backoff."""
        start_time = time.time()
        
        with self._request_lock:
            now = time.time()
            
            # Clean old requests
            self._request_times = [t for t in self._request_times 
                                 if now - t < self._window_seconds]
            
            # Check if we need to wait
            if len(self._request_times) >= self._max_requests:
                wait_time = self._window_seconds - (now - self._request_times[0])
                
                # Check timeout
                if timeout and wait_time > timeout:
                    return False
                
                # Implement exponential backoff
                backoff = min(wait_time * 1.5, self._window_seconds)
                time.sleep(backoff)
            
            # Ensure minimum delay between requests
            if self._request_times and now - self._request_times[-1] < self._min_delay:
                time.sleep(self._min_delay)
            
            self._request_times.append(time.time())
            return True
    
    def get_remaining_quota(self) -> int:
        """Get remaining requests in current window."""
        with self._request_lock:
            now = time.time()
            self._request_times = [t for t in self._request_times 
                                 if now - t < self._window_seconds]
            return max(0, self._max_requests - len(self._request_times))
    
    def reset(self):
        """Reset rate limiter state."""
        with self._request_lock:
            self._request_times = [] 