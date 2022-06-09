# -*- coding:utf-8 -*-

import unittest
import tempfile

from rl_framework.common.utils.trace_malloc import MallocTrace


class MallocTraceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.out = tempfile.mktemp()
        self.malloc_tracer = MallocTrace(out=self.out)

    def test_malloc_trace(self):
        self.malloc_tracer.start()

        self.malloc_tracer.take_snapshot()
        self.malloc_tracer.display_snapshot()

        # run your code
        __ = [dict(zip("abc", (1, 2, 3, 4, 5))) for _ in range(10000)]

        self.malloc_tracer.take_snapshot()
        self.malloc_tracer.display_snapshot()

        self.malloc_tracer.stop()

        with open(self.out) as out:
            read_lines = out.readlines()
            # print(read_lines)
            self.assertTrue(len(read_lines) > 0)


if __name__ == "__main__":
    unittest.main()
