"""
File to allow for print statements to be silenced in code
https://github.com/pb-aj/un-spot-able
"""

def spotable(message, dp=True):
   """
    Determines if print statement should be displayed or not.

    Arguments
    ---------
    message: string
        message to print

    dp: boolean (optional)
        Displayes message only if dispaly == True (default)

    Returns
    -------
    None
    """

   if dp:
        print(message)