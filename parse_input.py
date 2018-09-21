'''
General things to keep in mind:
* Individuals are represented as arrays of literals with a binary 0 or 1 
    (F or T) value. For example, an individual in a problem with 3 literals 
    might look like [0, 0, 1] where x0 = F, x1 = F, x2 = T.
* Clauses are represented as arrays indices specifying which indexes in an
    individual to test. For example, clause [1, 0] would translate as 'x1 | x0'
    and would be false for the example individual above.
'''

# ATTN DUNC AND DUSTIN: I'm doing this in an IDE on a PC, so chances are you'll
#   have to change this line to accommodate whatever your local file format is. 
#   ofc we'll eventually allow the filename to specified as input, but for now 
#   this is how it is.
# NOTE: this should stay the same as long as we're all working from a clone of 
#   the repo!


"""
    STYLE: 
        - 80 chars max per line
        - lets use underscores instead of camel case
        - function docstrings inside functions
        - module docstring at top of each file with 
            - authors, purpose, bugs, etc.
"""

#Initialize the variables we're going to need to keep track of the length of
#   our array of literals (vars x1,...,xn) and Clauses (TF statements that 
#   we're trying to MAXSAT), Technically you don't have to initialize anything
#   in Python, but I like the structure it lends the program.

num_literals = 0  # To be set upon file Parsing
num_clauses = 0  # To be set upon file Parsing
clause_list = [] # Initializing a variable

def get_num_clauses(filename):
    '''
    Function to define how many clauses and literals we are dealing with.
    Format: F(str) -> [numLits, num_clauses]
    '''
    with open(filename, 'r+') as cnf_file:
        for line in cnf_file.readlines():
            # Lines starting with P define the number of literals and 
            #   clauses we are dealing with
            if line.startswith('p'):  
                return line.split()[2:4]



def make_array_of_clauses(filename):
    '''
    Take a cnf file and return an list containing all the clauses from the 
    file. The clause array is formatted as an array of arrays, with each 
    subarray being a clause. You can think of each sub array as consisting
    of N truth functional vars represented as indices for our literals list.
    For example, subarray [0, 1, 2] would translate as 'x0 | x1 | x2' whereas
    [0, 1, -2] would translate as 'x0 & x1 & ~x2'. This translation will happen
    when 
    '''
    with open(filename, 'r+') as cnf_file:
        clauses = []
        for line in cnf_file.readlines():
            # Lines that don't start with c or p contain a clause:
            if not line[0].isalpha(): 
                clause_as_strings = line.replace("0", "").strip().split()
                clause_as_ints = []
                for literal in clause_as_strings:
                    clause_as_ints.append(int(literal)) 
                clauses.append(clause_as_ints)
        print(clauses)
        return clauses


def return_problem(file_name):
    """
    Purpose:
    Parameters:
    Return:
    """
    lit_and_clause_nums = get_num_clauses(file_name)
    num_literals = lit_and_clause_nums[0]
    num_clauses = lit_and_clause_nums[1]
    clause_list = make_array_of_clauses(file_name)
    problem = {
        "num_literals": int(num_literals),
        "num_clauses": int(num_clauses),
        "clauses": clause_list
    }
    return problem
