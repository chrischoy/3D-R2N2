#!/usr/bin/env python
"""
render_mesh.py provides functions for rendering meshes using OSGRenderer package
These are all wrapper functions so that users don't have to import PyRenderer
"""
import numpy as np
import sys

renderer_root = "../OSGRenderer/"
sys.path.append(renderer_root)

from PyRenderer import Renderer

def init_render(viewport_size_x, viewport_size_y, mesh_paths):
  """
  initialize renderer using the given parameters

  Parameters:
    viewport_size_x - horizontal length of rendering in pixels
    viewport_size_y - vertical length of rendering in pixels

  Return:
    renderer - renderer object
  """
  renderer = Renderer()
  renderer.initialize(mesh_paths, viewport_size_x, viewport_size_y)
  return renderer


def render_views(renderer, mesh_id, azimuths, elevations):
  """
  render multiple views of the object
  """
  assert(len(azimuths) == len(elevations))
  renderer.setModelIndex(mesh_id)
  renderings = []
  for view in zip(azimuths, elevations):
    renderer.setViewpoint(view[0], view[1], 0, 1, 25)
    rendering, depth = renderer.render()
    renderings.append(rendering.transpose((2, 1, 0)))

  return renderings
