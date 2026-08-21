import fcntl
import os

def get_next_slot(max_concurrency=16):
    counter_file = "/tmp/gcsfs_alts_counter.txt"
    fd = os.open(counter_file, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        val_bytes = os.read(fd, 32)
        if not val_bytes:
            val = 0
        else:
            try:
                val = int(val_bytes.decode('utf-8').strip())
            except ValueError:
                val = 0
        next_val = (val + 1) % max_concurrency
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, str(next_val).encode('utf-8'))
        return val
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

for i in range(20):
    print(get_next_slot())
