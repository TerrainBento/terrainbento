"""Clock sets the run duration and timestep in terrainbento model runs."""

import yaml

# import cfunits


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
        >>> from six import StringIO
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

    def __init__(self, start=0., step=10., stop=100., units="day"):
        """
        Parameters
        ----------
        start : float, optional
            Model start time. Default is 0.
        stop : float, optional
            Model stop time. Default is 100.
        step : float, optional
            Model time step. Default is 10.
        units : str, optional
            Default is "day".
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

        There are two ways to advance the model time stored in `clock.time`.

        >>> clock.time
        0.0

        First, to advance by the step size provided, use `advance`:

        >>> clock.advance()
        >>> clock.time
        200.0

        To  advance by an arbitrary time pass the `dt` value to `advance`:

        >>> clock.advance(2.)
        >>> clock.time
        202.0

        It is also possible to change the timestep.

        >>> clock.step = 18.
        >>> clock.advance()
        >>> clock.time
        220.0

        And to change the stop time.

        >>> clock.stop = 1000.
        >>> clock.stop
        1000.
        """
        # verify that unit is a valid CFUNITS
        # raise ValueError()

        try:
            self._start = float(start)
        except ValueError:
            msg = (
                "Clock: Required parameter *start* is "
                "not compatible with type float."
            )
            raise ValueError(msg)

        try:
            self._step = float(step)
        except ValueError:
            msg = (
                "Clock: Required parameter *step* is "
                "not compatible with type float."
            )
            raise ValueError(msg)

        try:
            self._stop = float(stop)
        except ValueError:
            msg = (
                "Clock: Required parameter *stop* is "
                "not compatible with type float."
            )
            raise ValueError(msg)

        if self.start > self.stop:
            msg = "Clock: *start* is larger than *stop*."
            raise ValueError(msg)

        self._time = 0.0

    def advance(self, dt=None):
        """Advance the time stepper by one time step or a provided value.

        Parameters
        ----------
        dt : float, optional
            Model time step. Default is to use the step provided at
            instantiation.
        """
        step = dt or self._step
        self._time += step
        if self._stop is not None and self._time > self._stop:
            raise StopIteration()

    @property
    def time(self):
        """Current time."""
        return self._time

    @property
    def start(self):
        """Start time."""
        return self._start

    @property
    def stop(self):
        """Stop time."""
        return self._stop

    @stop.setter
    def stop(self, new_val):
        """Change the stop time."""
        if self._time > new_val:
            raise ValueError("")
        self._stop = new_val

    @property
    def step(self):
        """Time Step."""
        return self._step

    @step.setter
    def step(self, new_val):
        """Change the time step."""
        self._step = new_val
