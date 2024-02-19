# mpai-spg-ai-training

The file GamesDatabased.txt stores all the information about the different games played.
All the code used for extrapolating the data and creating the input for the training is in the file Generator.py.
As described in the notebook NotebookTraining.ipynb, The database contains the following columns:

1. Players ID
2. Car's Global Position x-axis
3. Car's Global Position z-axis
4. Car's Velocity x-axis
5. Car's Velocity z-axis
6. Car's Rotation y-axis
7. Car's Angular Velocity y-axis
8. Car's Acceleration x-axis
9. Car's Acceleration z-axis
10. Tile ID
11. Tile Ranking (e.g. first, second, last tile of the track)
12. Car's Relative Position x-axis concerning the Tile's Center
13. Car's Relative Position z-axis concerning the Tile's Center
14. Time passed from the start of the race

Whenever the "Player ID" has the '_' character at the start of the name, it means a new race has started.
The "Tile ID" field specifies the type of tile (turn left, turn right etc.). Each unique tile is associated with a different integer number that creates the "Tile ID".
The race track can be seen as an array of tiles. "Tile Ranking" correspond to the index of the race track array associated with the tile.
