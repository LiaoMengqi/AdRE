class TemperatureScheduler:
    """
    Temperature Exponential Decay
    """

    def __init__(self,
                 init_temperature: float,
                 final_temperature: float,
                 num_steps: int):
        self.init_temperature = init_temperature
        self.final_temperature = final_temperature
        self.num_steps = num_steps
        self.current_temperature = init_temperature

    def step(self):
        self.current_temperature = self.current_temperature * ((self.final_temperature / self.init_temperature) ** (
                1 / self.num_steps))

    def get_current_temperature(self):
        return self.current_temperature
