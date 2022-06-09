# -*- coding:utf-8 -*-

import tracemalloc
import os
import linecache
import time


class MallocTrace(object):
    def __init__(self, out) -> None:
        self.out = open(out, "w")
        self.prev_snapshot = None
        self.curr_snapshot = None

    def __del__(self) -> None:
        self.out.close()

    @staticmethod
    def start(nframe=1):
        tracemalloc.start(nframe)

    @staticmethod
    def stop():
        tracemalloc.stop()

    def take_snapshot(self):
        self.prev_snapshot = self.curr_snapshot
        self.curr_snapshot = tracemalloc.take_snapshot()
        self.curr_snapshot = self.curr_snapshot.filter_traces(
            (
                tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
                tracemalloc.Filter(False, "<unknown>"),
                tracemalloc.Filter(False, tracemalloc.__file__),
                tracemalloc.Filter(False, linecache.__file__),
            )
        )

    def display_snapshot(self, key_type="lineno", limit=10):
        top_stats = self.curr_snapshot.statistics(key_type)[:limit]

        self.out.write(
            "##%s: Top %d Stats\n"
            % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), limit)
        )
        for index, stat in enumerate(top_stats, 1):
            frame = stat.traceback[0]
            filename = os.sep.join(frame.filename.split(os.sep))
            self.out.write(
                "#%s: %s:%s: %.1f KiB\n"
                % (index, filename, frame.lineno, stat.size / 1024)
            )
            for frame in stat.traceback:
                line = linecache.getline(frame.filename, frame.lineno).strip()
                if line:
                    self.out.write("    %s\n" % line)

        other = top_stats[limit:]
        if other:
            size = sum(stat.size for stat in other)
            self.out.write("%s other: %.1f KiB\n" % (len(other), size / 1024))
        total = sum(stat.size for stat in top_stats)
        self.out.write("Total allocated size: %.1f KiB\n" % (total / 1024))
        self.out.write("\n")
        self.out.flush()

    def compare_snapshot(self, limit=10):
        if not self.prev_snapshot or not self.curr_snapshot:
            return

        top_stats = self.curr_snapshot.compare_to(
            self.prev_snapshot, "lineno", cumulative=True
        )[:limit]
        self.out.write(
            "##%s: Top %d Stats\n"
            % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), limit)
        )
        for index, stat in enumerate(top_stats, 1):
            frame = stat.traceback[0]
            filename = os.sep.join(frame.filename.split(os.sep))
            self.out.write(
                "#%s: %s:%s: %.1f KiB (+%.1f Kib)\n"
                % (
                    index,
                    filename,
                    frame.lineno,
                    stat.size / 1024,
                    stat.size_diff / 1024,
                )
            )
            for frame in stat.traceback:
                line = linecache.getline(frame.filename, frame.lineno).strip()
                if line:
                    self.out.write("    %s\n" % line)

        other = top_stats[limit:]
        if other:
            size = sum(stat.size for stat in other)
            self.out.write("%s other: %.1f KiB\n" % (len(other), size / 1024))
        total = sum(stat.size for stat in top_stats)
        self.out.write("Total allocated size: %.1f KiB\n" % (total / 1024))
        self.out.write("\n")
        self.out.flush()
