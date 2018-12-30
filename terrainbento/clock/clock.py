""""""

import yaml


class Clock(object):
    """terrainbento clock."""

    @classmethod
    def from_file(cls, filename):
        """
        clock = Clock.from_file(yaml-file-like)
        """
        with open(filename, 'r') as f:
            dict = yaml.load(f)
        return cls.from_dict(dict)

    @classmethod
    def from_dict(cls, param):
        """
        clock = Clock.from_dict(dict-like)
        """
        return cls(**param)

    def __init__(self, start=0., step=10., stop=100.):
        """
        Parameters
        ----------
        start : float, optional
            Model start time. Default is 0.
        stop : float, optional
            Model stop time. Default is 100.
        step : float, optional
            Model time step. Default is 10.

        Examples
        --------
        >>> from terrainbento import Clock
        >>> clock = Clock()
        >>> clock.start
        0.0
        >>> clock.stop
        100.0
        >>> clock.step
        10.0
        """
        try:
            self.start = float(start)
        except ValueError:
            msg = ("Clock: Required parameter *start* is "
                   "not compatible with type float.")
            raise ValueError(msg)

        try:
            self.step = float(step)
        except ValueError:
            msg = ("Clock: Required parameter *step* is "
                   "not compatible with type float.")
            raise ValueError(msg)

        try:
            self.stop = float(stop)
        except ValueError:
            msg = ("Clock: Required parameter *stop* is "
                   "not compatible with type float.")
            raise ValueError(msg)

        if self.start > self.stop:
            msg = "Clock: *start* is larger than *stop*."
            raise ValueError(msg)
