"""Clock sets the run duration and timestep in terrainbento model runs."""

import yaml


class Clock(object):
    """terrainbento clock."""

    @classmethod
    def from_file(cls, filelike):
        """Construct a Clock from a yaml file.

        Parameters
        ----------
        filelike : file-like

        Examples
        --------
        >>> from io import StringIO
        >>> from terrainbento import Clock
        >>> filelike = StringIO('''
        ... start: 0
        ... step: 10
        ... stop: 100
        ... ''')
        >>> clock = Clock.from_file(filelike)
        >>> clock.start
        0.0
        >>> clock.stop
        100.0
        >>> clock.step
        10.0
        """
        try:
            with open(filelike, "r") as f:
                params = yaml.safe_load(f)
        except TypeError:
            params = yaml.safe_load(filelike)
        return cls.from_dict(params)

    @classmethod
    def from_dict(cls, params):
        """Construct a Clock from a dictionary.

        Parameters
        ----------
        param : dict-like

        Examples
        --------
        >>> from terrainbento import Clock
        >>> params = {"start": 0, "step": 10, "stop": 100}
        >>> clock = Clock.from_dict(params)
        >>> clock.start
        0.0
        >>> clock.stop
        100.0
        >>> clock.step
        10.0
        """
        return cls(**params)

    def __init__(self, start=0.0, step=10.0, stop=100.0):
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

        The follow constructs the default clock.

        >>> clock = Clock()
        >>> clock.start
        0.0
        >>> clock.stop
        100.0
        >>> clock.step
        10.0

        User specified parameters may be provided.

        >>> clock = Clock(start=0, step=200, stop=2400)
        >>> clock.start
        0.0
        >>> clock.stop
        2400.0
        >>> clock.step
        200.0
        """
        try:
            self.start = float(start)
        except ValueError:
            msg = (
                "Clock: Required parameter *start* is "
                "not compatible with type float."
            )
            raise ValueError(msg)

        try:
            self.step = float(step)
        except ValueError:
            msg = (
                "Clock: Required parameter *step* is "
                "not compatible with type float."
            )
            raise ValueError(msg)

        try:
            self.stop = float(stop)
        except ValueError:
            msg = (
                "Clock: Required parameter *stop* is "
                "not compatible with type float."
            )
            raise ValueError(msg)

        if self.start > self.stop:
            msg = "Clock: *start* is larger than *stop*."
            raise ValueError(msg)
