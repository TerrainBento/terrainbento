#!/usr/bin/env python3

from terrainbento.output_writers.static_interval_writer import (
    StaticIntervalOutputWriter,
)


class StaticIntervalOutputClassAdapter(StaticIntervalOutputWriter):
    def __init__(
        self,
        model,
        output_interval,
        ow_class,
        name="class-output-writer",
        **static_interval_kwargs,
    ):
        """A simple output writer which converts old style 'class' output
        writers to a new style static interval writer.

        Parameters
        ----------
        model : a terrainbento ErosionModel instance

        output_interval : float, int
            The model defined output interval.

        ow_class : uninstantiated output class
            An uninstantiated class that writes output in a **run_one_step**
            function.

        name : string, optional
            The name of the output writer to use when generating output
            filenames. Defaults to 'class-output-writer'

        static_interval_kwargs : keyword args, optional
            Keyword arguments that will be passed directly to
            StaticIntervalOutputWriter. These include:

                * save_first_timestep : bool, defaults to False
                * save_last_timestep : bool, defaults to True
                * output_dir : string, defaults to './output'

            Please see :py:class:`StaticIntervalOutputWriter` and
            :py:class:`GenericOutputWriter` for more detail. Note: `add_id` is
            automatically included as True.

        Returns
        -------
        StaticIntervalOutputClassAdapter: object

        """
        static_interval_kwargs["add_id"] = True  # Force add_id to be True
        super().__init__(
            model,
            name=name,
            intervals=output_interval,
            intervals_repeat=False,
            times=None,
            **static_interval_kwargs,
        )
        self.ow_class = ow_class(model)

    def run_one_step(self):
        """Call the old-style class's output function."""
        self.ow_class.run_one_step()


class StaticIntervalOutputFunctionAdapter(StaticIntervalOutputWriter):
    def __init__(
        self,
        model,
        output_interval,
        ow_function,
        name="function-output-writer",
        **static_interval_kwargs,
    ):
        """A simple output writer which converts old style 'function' output
        writers to a new style static interval writer.

        Parameters
        ----------
        model : a terrainbento ErosionModel instance

        output_interval : float, int
            The model defined output interval.

        ow_function : output function
            A function that can write output. This function must accept the
            model as its only argument.

        name : string, optional
            The name of the output writer to use when generating output
            filenames. Defaults to 'function-output-writer'

        static_interval_kwargs : keyword args, optional
            Keyword arguments that will be passed directly to
            StaticIntervalOutputWriter. These include:

                * save_first_timestep : bool, defaults to False
                * save_last_timestep : bool, defaults to True
                * output_dir : string, defaults to './output'

            Please see :py:class:`StaticIntervalOutputWriter` and
            :py:class:`GenericOutputWriter` for more detail. Note: `add_id` is automatically included as True.

        Returns
        -------
        StaticIntervalOutputFunctionAdapter: object

        """
        static_interval_kwargs["add_id"] = True  # Force add_id to be True
        super().__init__(
            model,
            name=name,
            intervals=output_interval,
            intervals_repeat=False,
            times=None,
            **static_interval_kwargs,
        )
        self.ow_function = ow_function

    def run_one_step(self):
        """Call the old-style output function."""
        self.ow_function(self.model)
