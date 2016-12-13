# -*- coding: utf-8 -*-
import time

import click


def measure_time(context=None, log_start=True):
    def outer(fun):
        def inner(self, *args, **kwargs):
            nonlocal context
            if context is None and hasattr(self, 'function'):
                context = self.function.__name__
            if log_start:
                click.secho('Started {}... '.format(context), fg='yellow')
            start_time = time.process_time()
            result = fun(self, *args, **kwargs)
            elapsed_sec = round(time.process_time() - start_time, 2)
            click.secho('Finished {} in {} sec'.format(context, elapsed_sec), fg='yellow')
            return result
        return inner
    return outer
