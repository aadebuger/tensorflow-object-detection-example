import threading
import cv2


class VideoCaptureThreading:
    def __init__(self, src=0, width=640, height=480):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def start(self):
        if self.started:
            print('[!] Threaded video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self): 
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()
import time
def runvideo(n_frames=500, width=1280, height=720, with_threading=False):
        if with_threading:
            cap = VideoCaptureThreading(1)
        else:
            cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if with_threading:
            cap.start()
        t0 = time.time()
        i = 0
        while i < n_frames:
            _, frame = cap.read()
#            cv2.imshow('Frame', frame)
#            cv2.waitKey(1) & 0xFF
            i += 1
        print('[i] Frames per second: {:.2f}, with_threading={}'.format(n_frames / (time.time() - t0), with_threading))
        if with_threading:
            cap.stop()
#        cv2.destroyAllWindows()
runvideo(with_threading=True)
