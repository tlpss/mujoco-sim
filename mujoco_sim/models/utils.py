from pathlib import Path

from dm_control import mjcf


def get_assets_root_folder() -> Path:
    return Path(__file__).parents[1] / "assets"


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
