import numpy as np
from dm_control import suite, viewer

environment = suite.load(domain_name="humanoid", task_name="stand")
# Define a uniform random policy.
spec = environment.action_spec()


def random_policy(time_step):
    return np.random.uniform(spec.minimum, spec.maximum, spec.shape)


# Launch the viewer application.
viewer.launch(environment, policy=random_policy)
