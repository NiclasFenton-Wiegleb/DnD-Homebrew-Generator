
def system_prompt(output_template):
    '''This function takes a json output_template and integrates it into the system prompt.'''
    system_prompt = f'''### Context ###
            You are an AI assisstant that extracts the locations, characters, key actions and clues of a narrative.

            ### Input ###
            You receive a narrative which contains different locations, key characters, actions and clues which move the narrative forward.

            ### Output ###

            Only include information vital to the plot. Ignore information that does not advance the narrative.

            locations:  list of locations where the narrative takes place in the order the locations are visited.
            characters: list of characters that appear in the narrative and the location where they first appear.
            actions: list of actions that advance the narrative. This includes the characters taking the action and the location where the action occurs.
            clues: list of clues that are observed by the protagonist that move the narrative forward. This includes the location of the clue and a summary of the significance of the clue.

            ### Output Template ###

            json:
            {output_template}
            '''
    return system_prompt