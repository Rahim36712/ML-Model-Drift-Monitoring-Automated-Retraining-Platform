"""OS-released process locks for single-owner API and worker services."""
from contextlib import contextmanager
import os


@contextmanager
def process_lock(path):
    stream = path.open("a+b")
    stream.seek(0)
    stream.write(b"0")
    stream.flush()
    stream.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        stream.close()
        raise RuntimeError(f"Another service already owns {path.name}. Use one serving worker.") from exc
    try:
        yield
    finally:
        stream.close()
