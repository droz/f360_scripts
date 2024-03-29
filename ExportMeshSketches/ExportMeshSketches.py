""" This is a python script for Fusion 360 that exports all the sketches
    that represent meshes in a model to a json file.
    The exported coordinates will be in meters."""

import sys
import os
import adsk.core, adsk.fusion, adsk.cam, traceback
import adsk

def run(context):
    with open('/tmp/dbg.txt', 'w') as out_file:
        sys.stdout = out_file

        ui = None
        try:
            app = adsk.core.Application.get()
            ui  = app.userInterface

            product = app.activeProduct
            design = adsk.fusion.Design.cast(product)
            units = design.unitsManager

            title = 'Mesh export'
            if not design:
                ui.messageBox('No active design', title)
                return
            # Start generating a json string.
            # The json library is not available in the Fusion 360 python environment, so
            # we have to generate the json manually.
            json = '{\n'
            json += '  "meshes" : [\n'
            # Find all the mesh sketches 
            root = design.rootComponent
            occs = root.allOccurrences
            for occ in occs:
                for sketch in occ.component.sketches:
                    if not sketch.name.lower().startswith('mesh'):
                        continue
                    part_name = sketch.name[5:]
                    json += '    { "name" : "' + part_name + '",\n'
                    json += '      "edges" : [\n'
                    for edge in sketch.sketchCurves.sketchLines:
                        # When we export the coordinates, we switch to a Z up coordinate frame.
                        # Fusion uses cm as internal units, so we convert to meters.
                        x_start_m = edge.geometry.startPoint.z * 0.01
                        y_start_m = edge.geometry.startPoint.x * 0.01
                        z_start_m = edge.geometry.startPoint.y * 0.01
                        x_end_m = edge.geometry.endPoint.z * 0.01
                        y_end_m = edge.geometry.endPoint.x * 0.01
                        z_end_m = edge.geometry.endPoint.y * 0.01
                        json += '        { "start" : [%f, %f, %f], "end" : [%f, %f, %f] },\n' % (
                                 x_start_m, y_start_m, z_start_m,
                                 x_end_m, y_end_m, z_end_m)
                    json = json[:-2] + '\n      ]\n'
                    json += '    },\n'

            json = json[:-2] + '\n  ]\n}\n'

            with open('/tmp/mesh_export.json', 'w') as json_file:
                json_file.write(json)

        except:
            if ui:
                ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))
