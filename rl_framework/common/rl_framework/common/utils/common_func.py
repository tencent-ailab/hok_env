# -*- coding:utf-8 -*-

import threading


# Singleton
class Singleton(object):
    _instance_lock = threading.Lock()

    def __init__(self, cls):
        self._cls = cls
        self._instance = {}

    def __call__(self, *args, **kwargs):
        if self._cls not in self._instance:
            with Singleton._instance_lock:
                if self._cls not in self._instance:
                    self._instance[self._cls] = self._cls(*args, **kwargs)
        return self._instance[self._cls]
