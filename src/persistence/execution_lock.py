"""Local worker exclusion released by the OS when the owning process exits."""

import os
from contextlib import contextmanager
from pathlib import Path


class ExecutionLockedError(RuntimeError):
    pass


@contextmanager
def exclusive_execution(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        try:
            if os.name == "nt":
                import msvcrt

                if not os.fstat(fd).st_size:
                    os.write(fd, b"0")
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, PermissionError) as exc:
            raise ExecutionLockedError("Another worker owns this checkpoint") from exc
        yield
    finally:
        # Do not unlink: another process may already be waiting on this inode.
        os.close(fd)
