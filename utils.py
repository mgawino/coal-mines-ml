# -*- coding: utf-8 -*-
import os
import time

import click

PROJECT_PATH = os.path.dirname(__file__)
RESULTS_PATH = os.path.join(PROJECT_PATH, 'results')
MODEL_CACHE_PATH = os.path.join(PROJECT_PATH, 'model_cache')


def measure_time(fun):
    def inner(self, *args, **kwargs):
        start_time = time.process_time()
        result = fun(self, *args, **kwargs)
        elapsed_sec = round(time.process_time() - start_time, 2)
        msg = self.function.__name__ if hasattr(self, 'function') else self.__class__.__name__
        click.secho('Finished {} in {} sec'.format(msg, elapsed_sec), fg='yellow')
        return result
    return inner


def timeit(fun, *args, **kwargs):
    start_time = time.process_time()
    fun(*args, **kwargs)
    duration = time.process_time() - start_time
    return duration