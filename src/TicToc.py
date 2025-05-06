import time

class TicToc:
    def __init__(self):
        self.start_time = None

    def tic(self):
        self.start_time = time.time()

    def toc(self, print_time=True):
        elapsed = time.time() - self.start_time
        if print_time:
            print(f"Elapsed time: {elapsed:.6f} seconds")
        return elapsed