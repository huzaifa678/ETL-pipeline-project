from src import logger
import logging

def error_message_detail(error):
    import sys
    _, _, exc_tb = sys.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno
    return f"Error occurred in file [{file_name}] at line [{line_no}]: {error}"
    
class CustomException(Exception):
    def __init__(self, error):
        super().__init__(error)
        self.error_message = error_message_detail(error)
        logging.error(self.error_message)
        
    def __str__(self):
        return self.error_message
    
    
if __name__ == "__main__":
    try:
        a = 1/0
    except Exception as e:
        raise CustomException(e)