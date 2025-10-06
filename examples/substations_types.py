from VeraGridEngine.api import *
from VeraGridEngine.Topology import substation_wizards

print('Creating grid...')

# declare a circuit object
grid = MultiCircuit()

country = Country('Spain')
grid.add_country(country)

# subs_vic = substation_wizards.simple_bar('Vic', grid, 2, 1, 220, 41.956664, 2.282089, country=country)

subs_centelles = substation_wizards.create_single_bar(name='Centelles', grid=grid, n_bays=4, v_nom=220,
                                                      country=country, include_disconnectors=True)

print()

print('Saving grid...')
save_file(grid, 'Test_substations_types_Alex.gridcal')
