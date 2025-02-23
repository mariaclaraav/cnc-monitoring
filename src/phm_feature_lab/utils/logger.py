import logging
import sys

class Logger:
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self) -> None:
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            # Create a console line handler
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.INFO)
            
            # Set formatter
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            handler.setFormatter(formatter)
            
            # Add handler to logger
            self.logger.addHandler(handler)
        
    def get_logger(self) -> logging.Logger:
        return self.logger
    
    def add_event(self, status: str = "SUCESS", message: str = "") -> None:
        """Log an event."""
        log_message = f"Message: {message}"
        if status == "SUCCESS":
            self.logger.info(log_message)
        elif status == "ERROR":
            self.logger.error(log_message)
        else:
            self.logger.debug(log_message)
            