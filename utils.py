# -*- coding: utf-8 -*-
import time

import click


def measure_time(context=None):
    def outer(fun):
        def inner(self, *args, **kwargs):
            nonlocal context
            start_time = time.process_time()
            result = fun(self, *args, **kwargs)
            elapsed_sec = round(time.process_time() - start_time, 2)
            msg = self.function.__name__ if context is None else context
            click.secho('Finished {} in {} sec'.format(msg, elapsed_sec), fg='yellow')
            return result
        return inner
    return outer