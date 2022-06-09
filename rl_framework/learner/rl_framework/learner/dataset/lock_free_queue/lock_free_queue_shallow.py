# -*- coding:utf-8 -*-
import multiprocessing
import multiprocessing.sharedctypes
import copy
import numpy as np


# 用condition实现的安全队列
class SafeQueue:
    """multiprocessing.Queue serialises python objects and stuffs them into a Pipe object.
    This serialisation happens in a background thread, and it not tied to the put() call.
    As such, no guarantee can be made about the order in while objects are put in the
    queue during threaded access.
    This poses a problem for the ordered mode, as it requires this guarantee.
    This class implements a very simple bounded queue that can guarantee ordering
    of inserted items."""

    def __init__(self, size):
        # The size of the queue is increased by one to give space for a QueueClosed signal.
        size += 1
        """
        The condition variable is used to both lock access to the internal resources
        and signal new items are ready.
        """
        self.cvar = multiprocessing.Condition()
        # A shared array is used to store items in the queue
        sary = multiprocessing.sharedctypes.RawArray("b", 8 * size)
        self.vals = np.frombuffer(sary, dtype=np.int64, count=size)
        self.vals[:] = -1
        # tail is the next item to be read from the queue
        self.tail = multiprocessing.sharedctypes.RawValue("l", 0)
        # size is the current number of items in the queue. head = tail + size
        self.size = multiprocessing.sharedctypes.RawValue("l", 0)

    def put(self, new_value):
        """
        Put an unsigned integer into the queue. This method always assumes that there is space in the queue.
        ( In the circular buffer, this is guaranteed by the implementation )
        :param new_value: The item to insert. Must be >= 0, as -2 is used to signal a queue close.
        :return:
        """
        assert new_value >= 0
        with self.cvar:
            assert self.size.value < len(self.vals)
            head = (self.tail.value + self.size.value) % len(self.vals)
            self.vals[head] = new_value
            self.size.value += 1
            self.cvar.notify()

    def get(self):
        """
        Fetch the next item in the queue. Blocks until an item is ready.
        :return: The next unsigned integer in the queue.
        """
        with self.cvar:
            while True:
                if self.size.value > 0:
                    rval = self.vals[self.tail.value]
                    self.tail.value = (self.tail.value + 1) % len(self.vals)
                    self.size.value -= 1
                    assert rval >= 0
                    return rval
                self.cvar.wait()

    def get_circ_size(self):
        return self.size.value


class SharedCircBuf:
    """A circular buffer for numpy arrays that uses shared memory for
    inter-process communication."""

    def __init__(self, queue_size, ary_template, keys):
        """
        Create the circular buffer. An array template must be passed to determine the
        size of the buffer elements.

        :param queue_size: Number of arrays to use as buffer elements.
        :param ary_template: Buffer elements match this array in shape and data-type.
        """
        import multiprocessing.sharedctypes

        # The buffer uses two queues to synchonise access to the buffer.
        # Element indices are put and fetched from these queues.
        # Elements that are ready to be written to go into the write_queue.
        # Elements that are ready to be read go into the read_queue.
        # This is essentially a token passing process. Tokens are taken out of queues
        # and are not put back until
        # operations are complete.
        self.read_queue = SafeQueue(queue_size)
        self.write_queue = SafeQueue(queue_size)
        self.write_idx = -1

        self.arys = []
        ###按照元素创建了共享内存，按照元素的共享内存数组
        for i in range(queue_size):
            data = []
            for skey in range(len(keys)):
                key = keys[skey]
                elem_n_bytes = ary_template[key].nbytes
                elem_dtype = ary_template[key].dtype
                elem_size = ary_template[key].size
                elem_shape = ary_template[key].shape
                sarray = multiprocessing.sharedctypes.RawArray("b", elem_n_bytes)
                data.append(
                    np.frombuffer(sarray, dtype=elem_dtype, count=elem_size).reshape(
                        elem_shape
                    )
                )
            self.arys.append(data)
            self.write_queue.put(i)

    class Guard:
        """with statement guard object for synchronisation of access to the buffer."""

        def __init__(self, out_queue, arys, idx_op):
            """
            The guard object returns ary, and once the guard ends the value of idx is put into queue.
            Used to put an element index representing a token back into the buffer queues
            once operations on the element are complete.

            :param queue: The queue to be populated when the guard ends.
            :param idx: The value to put into the queue.
            :param ary: Value to return in the with statement.
            """
            self.out_queue = out_queue
            self.arys = arys
            self.idx_op = idx_op

        def __enter__(self):
            self.idx = self.idx_op()
            return self.arys[self.idx]

        def __exit__(self, *args):
            self.out_queue.put(self.idx)

    def put(self, in_ary):
        """
        Convenience method to put in_ary into the buffer.
        Blocks until there is room to write into the buffer.

        :param in_ary: The array to place into the buffer.
        :return:
        """
        with self.put_direct() as ary:
            for i in range(len(ary)):
                ary[i][:] = in_ary[i]

    def __put_idx(self):
        write_idx = self.write_queue.get()

        return write_idx

    def put_direct(self):
        """
        Allows direct access to the buffer element.
        Blocks until there is room to write into the buffer.

        :return: A guard object that returns the buffer element.
        """
        # Once the guard is released, write_idx will be placed into read_queue.
        return self.Guard(self.read_queue, self.arys, self.__put_idx)

    def get(self):
        """
        Convenience method to get a copy of an array in the buffer.
        Blocks until there is data to be read.

        :return: A copy of the next available array.
        """
        if self.write_idx >= 0:
            self.write_queue.put(self.write_idx)
        read_idx = self.read_queue.get()
        result = copy.copy(self.arys[read_idx])
        self.write_idx = read_idx

        return result

    def get_size(self):
        return self.read_queue.get_circ_size()
