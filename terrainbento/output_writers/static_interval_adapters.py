#!/usr/bin/env python3

class StaticIntervalOutputClassAdapter(StaticIntervalOutputWriter):
    def __init__(self,
            model,
            output_interval,
            ow_class,
            save_first_timestep=False,
            ):
        self.ow_class = ow_class(model)
        super().__init__(model,
                name="class-output-writer",
                intervals=output_interval,
                add_id=True,
                save_first_timestep=save_first_timestep,
                )

    def run_one_step(self):
        self.ow_class.run_one_step()

class StaticIntervalOutputFunctionAdapter(StaticIntervalOutputWriter):
    def __init__(self,
            model,
            output_interval,
            ow_function,
            save_first_timestep=False,
            ):
        self.ow_function = ow_function
        super().__init__(model,
                name="function-output-writer",
                intervals=output_interval,
                add_id=True,
                save_first_timestep=save_first_timestep,
                )

    def run_one_step(self):
        self.ow_function()
