from enum import Enum


class Task(Enum):
    """Task."""
    ATE = "ate"
    ACD = "acd"
    E2E = "e2e"
    ACSA = "acsa"
    ACTE = "acte"
    TASD = "tasd"
    MULTI_TASK = "multi_task"

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)


def convert_value_to_task(value: str) -> Task:
    """
    Convert value to Task.

    :param value: value
    :return: Task
    """
    for task in Task:
        if value == task.value:
            return task
    raise ValueError(f"Task with value {value} does not exist.")
