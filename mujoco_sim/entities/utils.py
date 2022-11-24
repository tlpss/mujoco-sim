from __future__ import annotations

from pathlib import Path

from dm_control import mjcf


def get_assets_root_folder() -> Path:
    return Path(__file__).parents[1] / "mjcf"


def write_xml(model: mjcf.RootElement):
    mjcf.export_with_assets(model, "mjcf")


def create_dummy_arena() -> mjcf.RootElement():
    arena = mjcf.RootElement()
    checker = arena.asset.add(
        "texture", type="2d", builtin="checker", width=300, height=300, rgb1=[0.2, 0.3, 0.4], rgb2=[0.3, 0.4, 0.5]
    )
    grid = arena.asset.add("material", name="grid", texture=checker, texrepeat=[5, 5], reflectance=0.2)
    arena.worldbody.add("geom", type="plane", size=[2, 2, 0.1], material=grid)
    for x in [-2, 2]:
        arena.worldbody.add("light", pos=[x, -1, 3], dir=[-x, 1, -2])
    return arena


if __name__ == "__main__":
    print(get_assets_root_folder())


def build_mocap(model: mjcf.RootElement, name: str) -> mjcf.Element:
    # mocap -> 'teleporting bodies'
    # https://github.com/deepmind/mujoco/issues/433
    # https://mujoco.readthedocs.io/en/latest/modeling.html?highlight=mocap#mocap-bodies
    mocap = model.worldbody.add("body", name=name, pos=[0.0, 0.0, 0.0], mocap=True)

    # use site to make the mocap geometry object 'visual only' (no collisions)
    # https://mujoco.readthedocs.io/en/latest/XMLreference.html?highlight=site#body-site
    mocap.add("site", type="sphere", size=[0.005])
    return mocap
