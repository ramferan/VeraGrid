from VeraGridEngine.api import *
from VeraGridEngine.Topology.VoltageLevels import common_functions

print('Creating grid...')

# declare a circuit object
grid = MultiCircuit()

country = Country('Spain')
grid.add_country(country)

# subs_vic = substation_wizards.simple_bar('Vic', grid, 2, 1, 220, 41.956664, 2.282089, country=country)

sub = Substation()
grid.add_substation(sub)

subs_centelles, conn_buses, all_buses, x_off, y_off = common_functions.create_single_bar(name='Centelles',
                                                                                         grid=grid,
                                                                                         n_bays=4,
                                                                                         v_nom=220,
                                                                                         substation=sub,
                                                                                         country=country,
                                                                                         include_disconnectors=True)

print()

print('Saving grid...')
save_file(grid, 'Test_substations_types_Alex.gridcal')
