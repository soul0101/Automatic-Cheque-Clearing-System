def check_valid_micr(micr_string):
    """
    Checks if a string is a valid MICR code.
    A valid MICR code is a string of the form:
        ⑈<6 digits>⑈<8 digits>⑆<5 digits>⑈31
    Where ⑈ and ⑆ are special characters.

    Parameters
    ----------
    micr_string: str
        The string to check.

    Returns
    -------
    bool
        True if the string is a valid MICR code, False otherwise.
    """
    return (micr_string[0] == "⑈" and 
            micr_string[1:6].isdigit() and
            micr_string[7] == "⑈" and
            micr_string[8:16].isdigit() and
            micr_string[17] == "⑆" and 
            micr_string[18:23].isdigit() and 
            micr_string[24] == "⑈" and 
            micr_string[25:27] == "31")

def get_digits(s):
    """
    Returns a string containing all the digits in the input string s.

    Parameters
    ----------
    s : str
        The input string.

    Returns
    -------
    str
        A string containing all the digits in the input string s.
    """
    return ''.join(ch for ch in s if ch.isdigit())

def get_alnum(s):
    """
    Returns a string containing only the alphanumeric characters in the input string.
    
    Parameters
    ----------
    s : str
        The input string.
    
    Returns
    -------
    str
        The output string.
    """
    return ''.join(ch for ch in s if ch.isalnum())

def get_alpha(s):
    """
    Returns a string containing only the alphabetic characters in the input string.
    
    Parameters
    ----------
    s : str
        The input string.
    
    Returns
    -------
    str
        The output string.
    """
    return ''.join(ch for ch in s if ch.isalpha())

def is_empty(a):
    """
    Returns True if the given array or string is empty, False otherwise.
    
    Parameters
    ----------
    a : list or string
        The array or string to check.
        
    Returns
    -------
    bool
        True if the given array or string is empty, False otherwise.
        """
    return len(a) == 0

if __name__ == "__main__":
    test = []
    print(is_empty(test))