
# coding: utf-8

# In[2]:


class commands:
    def __init__(self):
        self.basic_actions = [
           'go north',
           'go south',
           'go west',
           'go east',
           'go northeast',
           'go northwest',
           'go southeast',
           'go southwest',
           'go down',
           'go up']

        self.directions = ['north', 'south', 'east', 'west', 'northwest', 'northeast', 'southwest', 'southeast', 'up', 'down']

        self.action_space = set()

        for action in self.basic_actions:
            self.action_space.add(action)

        self.command1_actions = [
         'open OBJ',
         'get OBJ',
         'eat OBJ',
         'ask OBJ',
         'make OBJ',
         'wear OBJ',
         'move OBJ',
         'kick OBJ',
         'find OBJ',
         'play OBJ',
         'feel OBJ',
         'read OBJ',
         'fill OBJ',
         'pick OBJ',
         'pour OBJ',
         'pull OBJ',
         'leave OBJ',
         'break OBJ',
         'enter OBJ',
         'shake OBJ',
         'banish OBJ',
         'read OBJ',
         'enchant OBJ',
         'feel OBJ',
         'pour OBJ']
                  
        self.command2_actions = [
        'pour OBJ on DCT',
        'hide OBJ in DCT',
        'pour OBJ in DCT',
        'move OBJ in DCT',
        'hide OBJ on DCT',
        'flip OBJ for DCT',
        'fix OBJ with DCT',
        'spray OBJ on DCT',
        'dig OBJ with DCT',
        'cut OBJ with DCT',
        'pick OBJ with DCT',
        'pour OBJ from DCT',
        'fill OBJ with DCT',
        'burn OBJ with DCT',
        'flip OBJ with DCT',
        'read OBJ with DCT',
        'hide OBJ under DCT',
        'carry OBJ from DCT',
        'inflate OBJ with DCT',
        'unlock OBJ with DCT',
        'give OBJ to DCT', 
        'carry OBJ to DCT',
        'spray OBJ with DCT']
                    

        self.filtered_tokens = ['Score', 'Moves']

