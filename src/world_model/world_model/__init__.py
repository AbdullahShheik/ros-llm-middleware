"""Shared world-context derivation for the ros-llm-middleware stack.

Two modules, kept separate by dependency weight:

  scene_tracking     -- stdlib XML only. Classifies world entities into
                        objects / zones / robots by parsing the world SDF.
  build_environment  -- needs yaml + numpy + PIL. Fuses the occupancy map,
                        the SDF spawn poses, and live /object_map readings
                        into the "=== ENVIRONMENT CONTEXT ===" prompt block.

Nothing is re-exported at package level on purpose: importing
``world_model.scene_tracking`` must not drag in build_environment's heavier
dependencies, which an eager re-export here would do. Import the module you
need directly.
"""
