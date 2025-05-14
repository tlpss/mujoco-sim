from dm_control import composer, mjcf
from dm_control.composer.observation import observable

NUM_SUBSTEPS = 25  # The number of physics substeps per control timestep.


class Switch(composer.Entity):
    """A Switch (bistable button) Entity. The switch is pressed when a force is applied to it, the state of the switch flips on the rising edge when pressed."""

    def __init__(self, *args, **kwargs):
        self.box_size = 0.05
        self.box_height = 0.05
        self._is_active = False
        self._is_pressed = False
        super().__init__(*args, **kwargs)

    def _build(self, target_force_range=(5, 200)):
        self._min_force, self._max_force = target_force_range

        self._mjcf_model = mjcf.RootElement()

        # create switch that consist of a cube and a cylinder as a button

        self._cube_geom = self._mjcf_model.worldbody.add(
            "geom",
            type="box",
            size=[self.box_size / 2, self.box_size / 2, self.box_height / 2],
            rgba=[1, 1, 1, 1],
            pos=[0, 0, self.box_height / 2],
        )
        self._button_geom = self._mjcf_model.worldbody.add(
            "geom",
            type="cylinder",
            size=[self.box_size / 2.5, self.box_size / 2.5, 0.01],
            rgba=[1, 0, 0, 1],
            pos=[0, 0, self.box_height],
        )
        self._button_site = self._mjcf_model.worldbody.add(
            "site", type="cylinder", size=self._button_geom.size * 1.01, rgba=[1, 0, 0, 0], pos=[0, 0, self.box_height]
        )
        self._sensor = self._mjcf_model.sensor.add("touch", site=self._button_site)

    def _build_observables(self):
        return SwitchObservables(self)

    @property
    def mjcf_model(self):
        return self._mjcf_model

    # Update the activation (and colour) if the desired force is applied.
    def _update_activation(self, physics):
        current_force = physics.bind(self.touch_sensor).sensordata[0]
        self.was_pressed = self._is_pressed
        self._is_pressed = current_force >= self._min_force and current_force <= self._max_force

 
        # Debounce logic: prevent reactivation for a certain number of steps
        if not hasattr(self, "_debounce_counter"):
            self._debounce_counter = 0

        if self._debounce_counter > 0:
            self._debounce_counter -= 1
            return
    
        # flip state on rising edge
        if self._is_pressed and not self.was_pressed:
            self._is_active = not self._is_active
            self._debounce_counter = 50  # Set debounce duration
        
        physics.bind(self._button_geom).rgba = [0, 1, 0, 1] if self._is_active else [1, 0, 0, 1]
        self._num_pressed_steps += int(self._is_pressed)

    def initialize_episode(self, physics, random_state):
        self._num_pressed_steps = 0
        self._is_active = False
        self._update_activation(physics)

    def deactivate(self, physics):
        self._is_active = False
        physics.bind(self._button_geom).rgba = [1, 0, 0, 1]

    def after_substep(self, physics, random_state):
        self._update_activation(physics)

    @property
    def touch_sensor(self):
        return self._sensor

    @property
    def num_activated_steps(self):
        return self._num_pressed_steps

    @property
    def is_active(self):
        return self._is_active

    def get_position(self, physics):
        return physics.bind(self._button_geom).xpos + 0.5 * self._button_geom.size[1]


class SwitchObservables(composer.Observables):
    """A touch sensor which averages contact force over physics substeps."""

    @composer.observable
    def touch_force(self):
        return observable.MJCFFeature(
            "sensordata", self._entity.touch_sensor, buffer_size=NUM_SUBSTEPS, aggregator="mean"
        )

    @composer.observable
    def position(self):
        return observable.Generic(self._entity.get_position)

    @composer.observable
    def active(self):
        return observable.Generic(self._entity.is_active)


if __name__ == "__main__":

    block = Switch()
    model = block.mjcf_model
    physics = mjcf.Physics.from_mjcf_model(model)

    # visualize mujoco scene
    print(block.get_position(physics))

    # visualize mujoco scene
    import mujoco
    from mujoco import viewer

    m = mujoco.MjModel.from_xml_string(model.to_xml_string())
    viewer.launch(m)
