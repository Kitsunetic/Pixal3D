from pathlib import Path


def test_all_boundary_fits_finish_before_final_renders():
    # Given: the production Blender renderer source.
    repository = Path(__file__).resolve().parents[2]
    source = (repository / "data_toolkit/blender_script/render_cond.py").read_text()
    main_source = source[source.index("def main(arg):") : source.index("if __name__")]

    # When: the two render phases and persistent-scene setting are located.
    fit_loop = main_source.index("for i, view in enumerate(views):")
    final_engine = main_source.index("scene.render.engine = arg.engine")
    final_loop = main_source.index("for fitted_view in fitted_views:")
    persistent_data = main_source.index("scene.render.use_persistent_data = True")

    # Then: every fit precedes the single engine switch and every final render.
    assert persistent_data < fit_loop < final_engine < final_loop
    assert main_source.count("scene.render.engine = arg.engine") == 1


def test_each_views_lighting_rng_is_replayed_for_final_render():
    # Given: the production Blender renderer source.
    repository = Path(__file__).resolve().parents[2]
    source = (repository / "data_toolkit/blender_script/render_cond.py").read_text()
    main_source = source[source.index("def main(arg):") : source.index("if __name__")]

    # When: fit-state capture and final-state replay are located.
    capture = main_source.index("lighting_rng_state = np.random.get_state()")
    fit_lighting = main_source.index("init_random_lighting(cam_dir)", capture)
    replay = main_source.index("np.random.set_state(fitted_view[\"lighting_rng_state\"])")
    final_lighting = main_source.index("init_random_lighting(cam_dir)", replay)
    post_fit_capture = main_source.index("post_fit_rng_state = np.random.get_state()")
    post_final_restore = main_source.index(
        "np.random.set_state(post_fit_rng_state)", final_lighting
    )

    # Then: each final render replays lighting without advancing global RNG state.
    assert capture < fit_lighting < post_fit_capture < replay < final_lighting
    assert final_lighting < post_final_restore
