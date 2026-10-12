# One-click ParaView setup for the polycrystal cooling output.
#
# Run it from ParaView:  Tools -> Python Shell -> "Run Script...", pick this file.
# (Or from a terminal:  pvpython paraview_setup.py  -- but the GUI is the point here.)
#
# It builds the view described in README.md:
#   cells coloured by damage_driving_ratio (>1 == damage growing right now)
#   one black line glyph per grain, along that grain's basal-plane normal (c_axis)
#
# WHY THE TWO EXTRA FILTERS: c_axis is written at all 8 corners of every cell -- that is what
# keeps values from being averaged across grain boundaries -- so glyphing the raw points would
# give 8 glyphs per grain, sitting at its corners. PointDataToCellData collapses the 8 identical
# corner values to one per cell (exact: they are bit-identical), and CellCenters turns that into
# a single point at the centroid.
#
# WHY A LINE AND NOT AN ARROW: the c-axis is a DIRECTOR. c and -c describe the same crystal, and
# the sign stored in the file is an artefact of how the direction was sampled. An arrow would
# show a polarity the material does not have.

import os
from paraview.simple import *

HERE = os.path.dirname(os.path.abspath(__file__))
PVD = os.path.join(HERE, "build", "polycrystalCooling.pvd")
if not os.path.exists(PVD):
    PVD = os.path.join(HERE, "polycrystalCooling.pvd")

CELL_SIZE = 0.1          # LengthOfTheBody / cells per side
COLOR_BY = "damage_driving_ratio"
COLOR_RANGE = (0.0, 1.2)  # centred on 1.0 = the damage threshold

reader = PVDReader(FileName=PVD)
view = GetActiveViewOrCreate("RenderView")

# --- the grains, coloured by how hard damage is being driven ---
grains = Show(reader, view)
grains.Representation = "Surface With Edges"
ColorBy(grains, ("POINTS", COLOR_BY))
lut = GetColorTransferFunction(COLOR_BY)
lut.ApplyPreset("Cool to Warm (Extended)", True)
lut.RescaleTransferFunction(*COLOR_RANGE)
grains.SetScalarBarVisibility(view, True)
GetScalarBar(lut, view).Title = "sigma_33 / threshold"

# --- one c-axis glyph per grain ---
cellData = PointDatatoCellData(Input=reader)
centers = CellCenters(Input=cellData)
glyph = Glyph(Input=centers, GlyphType="Line")
glyph.OrientationArray = ["POINTS", "c_axis"]
glyph.ScaleArray = ["POINTS", "No scale array"]
glyph.ScaleFactor = 0.8 * CELL_SIZE
glyph.GlyphMode = "All Points"

g = Show(glyph, view)
g.Representation = "Surface"
ColorBy(g, None)                 # solid colour, so the lines read as geometry not data
g.AmbientColor = [0.0, 0.0, 0.0]
g.DiffuseColor = [0.0, 0.0, 0.0]
g.LineWidth = 2.0

# --- last timestep, camera on the whole RVE ---
scene = GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()
scene.AnimationTime = reader.TimestepValues[-1]
view.ResetCamera()
view.CameraPosition = [2.6, -2.2, 1.9]
view.CameraFocalPoint = [0.5, 0.5, 0.5]
view.CameraViewUp = [0.0, 0.0, 1.0]
Render()

print("Loaded %d timesteps (animation clock = accumulated cooling in K, 0 -> %g)."
      % (len(reader.TimestepValues), reader.TimestepValues[-1]))
print("Coloured by %s; black lines are each grain's basal-plane normal." % COLOR_BY)
print("Press Play to watch stress build, cross the threshold, then damage follow.")
