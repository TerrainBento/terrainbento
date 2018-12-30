""""""

import yaml


class Clock(object):
    """terrainbento clock."""

    @classmethod
    def from_file(cls, filename):
        """
        clock = Clock.from_file("file.yaml")
        """
        dict = yaml.load(filename)
        return cls.from_dict(dict)

    @classmethod
    def from_dict(cls, dictionary, outputwriters=None):
        """
        clock = Clock.from_dict(dict-like)
        """
        params = yaml.load(filename)
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
            msg = "Required parameter *start* is not " "compatible with type float."

        try:
            self.step = float(step)
        except ValueError:
            msg = "Required parameter *step* is not " "compatible with type float."

        try:
            self.stop = float(stop)
        except ValueError:
            msg = "Required parameter *stop* is not " "compatible with type float."
