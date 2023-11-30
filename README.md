# CCST9017
The code implemented in our group project is shown here, apology for not having enough time to write some annotation.  
It try to approximate the winning strategy of SIM.  

Using nueral work to solve mathematical problems is a relatively new subject, but the author believe that it will someday become extremely popular.


To play with out AI, first use command "pip install -r requirements.txt" to install dependency, then run "python play_game.py" and follow the instruction.

## SIM
SIM is a game that two players play sequentially to draw lines between six points, the first one whose own lines form a triangluar with three of the six points as the corner will lose the game.  
It is known that player two must have a winning strategy, since it is a combinatorial game that cannot end in a draw. But the winning strategy itself is not elegant at all.  
Hence, we try to approximate the winning strategy with deep neural network.  
This work can be easily transferred to other game whose winning strategies have not yet to be discoverred, since we barely made any assumption about the game.   