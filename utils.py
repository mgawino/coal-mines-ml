# -*- coding: utf-8 -*-


def grouped(iterable, count):
    """ Group @iterable into lists of length @count """
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == count:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk
