"""Dynamic step-dependent parameters"""

from typing import Union

from tensorflow.keras.optimizers import schedules
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class DynamicParameter:
    """Interface for learning rate schedule wrappers as dynamic-parameters"""
    def __init__(self):
        self.value = 0
        self.step = 0

    @staticmethod
    def create(value: Union[float, LearningRateSchedule], **kwargs):
        """Converts a floating or LearningRateSchedule `value` into a DynamicParameter object"""
        if isinstance(value, float):
            return ConstantParameter(value)

        if isinstance(value, LearningRateSchedule):
            return ScheduleWrapper(schedule=value, **kwargs)

        # already DynamicParameter (or, DynamicParameter only!)
        assert isinstance(value, DynamicParameter) or isinstance(value, ScheduleWrapper)
        return value

    def __call__(self, *args, **kwargs):
        return self.value

    def serialize(self) -> dict:
        return dict(step=int(self.step))

    def on_episode(self):
        self.step += 1

    def load(self, config: dict):
        self.step = config.get('step', 0)

    def get_config(self) -> dict:
        return {}


# TODO: decay on new episode (optional)
class ScheduleWrapper(LearningRateSchedule, DynamicParameter):
    """A wrapper for built-in tf.keras' learning rate schedules"""
    def __init__(self, schedule: LearningRateSchedule, min_value=1e-4):
        super().__init__()
        self.schedule = schedule
        self.min_value = min_value

    def __call__(self, *args, **kwargs):
        # self.step += 1
        self.value = max(self.min_value, self.schedule.__call__(self.step))
        return self.value

    def get_config(self) -> dict:
        return self.schedule.get_config()


class ConstantParameter(DynamicParameter):
    """A constant learning rate schedule that wraps a constant float learning rate value"""
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def __call__(self, *args, **kwargs):
        return self.value

    def serialize(self) -> dict:
        return {}


class ExponentialDecay(ScheduleWrapper):
    def __init__(self, initial_value: float, decay_steps: int, decay_rate: float, staircase=False, min_value=0.0):
        super().__init__(schedule=schedules.ExponentialDecay(initial_learning_rate=initial_value,
                                                             decay_steps=decay_steps, decay_rate=decay_rate,
                                                             staircase=staircase),
                         min_value=min_value)


class StepDecay(ScheduleWrapper):
    def __init__(self, initial_value: float, decay_steps: int, decay_rate: float, min_value=1e-4):
        super().__init__(schedule=schedules.ExponentialDecay(initial_value, decay_steps, decay_rate, staircase=True),
                         min_value=min_value)


class PolynomialDecay(ScheduleWrapper):
    def __init__(self, initial_value: float, end_value: float, decay_steps: int, power=1.0, cycle=False):
        super().__init__(schedule=schedules.PolynomialDecay(initial_learning_rate=initial_value,
                                                            decay_steps=decay_steps, end_learning_rate=end_value,
                                                            power=power, cycle=cycle))
