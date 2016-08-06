#!/usr/bin/env python3

import os
import sys
from contextlib import contextmanager
import time


def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


@contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    if stdout is None:
        stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    # NOTE: `copied` is inheritable on Windows when duplicating a standard
    # stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout  # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            # NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

# class Timer:
#     def __init__(self):
#         self.reset()
#
#     def update(self, train_time, wait_time):
#         self.cycle += 1
#         self.cycle_t = time.time() - self.prev_time
#         self.accum_time += self.cycle_t
#         self.eta = self.accum_time/self.cycle*100000/3600
#         self.prev_time = time.time()
#
#         self.accum_train_time += train_time
#         self.accum_wait_time += wait_time
#
#     def reset(self):
#         self.prev_time = time.time()
#         self.accum_time = 0
#         self.accum_train_time = 0
#         self.accum_wait_time = 0
#         self.cycle_t = 0
#         self.cycle = 0
#         self.eta = 0
#
#     def print_stat(self):
#         print("Average cycle time: %fs" % (self.accum_time/self.cycle))
#         print("Average training time: %fs" % (self.accum_train_time/self.cycle))
#         print("Average waiting time: %fs" % (self.accum_wait_time/self.cycle))
#         print("ETA: %f hrs" % (self.eta))
