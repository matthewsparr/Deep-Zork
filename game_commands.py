
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
           'd',
           'u']

        self.directions = ['north', 'south', 'east', 'west', 'northwest', 'northeast', 'southwest', 'southeast']

        self.action_space = set()

        for action in self.basic_actions:
            self.action_space.add(action)

        self.command1_actions = [
         'open OBJ',
         'get OBJ',
         'set OBJ',
         'hit OBJ',
         'eat OBJ',
         'put OBJ',
         'cut OBJ',
         'dig OBJ',
         'ask OBJ',
         'fix OBJ',
         'make OBJ',
         'wear OBJ',
         'move OBJ',
         'kick OBJ',
         'kill OBJ',
         'find OBJ',
         'play OBJ',
         'feel OBJ',
         'hide OBJ',
         'read OBJ',
         'fill OBJ',
         'flip OBJ',
         'burn OBJ',
         'pick OBJ',
         'pour OBJ',
         'pull OBJ',
         'apply OBJ',
         'leave OBJ',
         'ask OBJ',
         'break OBJ',
         'enter OBJ',
         'curse OBJ',
         'shake OBJ',
         'burn OBJ',
         'inflate OBJ',
         'brandish OBJ',
         'donate OBJ',
         'squeeze OBJ',
         'attach OBJ',
         'find OBJ',
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
        'squeeze OBJ on DCT',
        'pour OBJ from DCT',
        'fill OBJ with DCT',
        'brandish OBJ at DCT',
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

