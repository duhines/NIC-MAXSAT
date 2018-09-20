'''
General things to keep in mind
* Individuals are represented as arrays of literals with a binary 0 or 1 (F or T) value. For example, an individual
in a problem with 3 literals might look like [0, 0, 1] where x0 = F, x1 = F, x2 = T.
* Clauses are represented as arrays indices specifying which indexes in an individual to test. For example, clause
[1, 0] would translate as 'x1 | x0' and would be false for the example individual above.
'''

# ATTN DUNC AND DUSTIN: I'm doing this in an IDE on a PC, so chances are you'll have to change this line to accommodate
# whatever your local file format is. ofc we'll eventually allow the filename to specified as input, but for now this is
# how it is.
test_file = r"maxsat-problems/maxsat-crafted/MAXCUT/DIMACS_MOD/brock200_1.clq.cnf"

# Initialize the variables we're going to need to keep track of the length of our array of literals (vars x1,...,xn)
# and Clauses (TF statements that we're trying to MAXSAT), Technically you don't have to initialize anything in Python,
# but I like the structure it lends the program.
numLiterals = 0  # To be set upon file Parsing
numClauses = 0  # To be set upon file Parsing
clauseList = [] # Initializing a variable
'''
Function to define how many clauses and literals we are dealing with.
Format: F(str) -> [numLits, numClauses]
'''
def get_num_clauses(filename):
    with open(filename, 'r+') as cnf_file:
        for l in cnf_file.readlines():
            if l.startswith('p'):  # Lines starting with P define the number of literals and clauses we are dealing with
                return l.split()[2:4]


'''
Take a cnf file and return an list containing all the clauses from the file. The clause array is formatted as an array
of arrays, with each subarray being a clause. You can think of each sub array as consisting of N truth functional vars
represented as indices for our literals list. For example, subarray [0, 1, 2] would translate as 'x0 | x1 | x2' whereas
[0, 1, -2] would translate as 'x0 & x1 & ~x2'. This translation will happen when 
'''
def make_array_of_clauses(filename):
    with open(filename, 'r+') as cnf_file:
        clauses = []
        for l in cnf_file.readlines():
            if not l[0].isalpha():  # Lines that don't start with c or p contain a clause:
                clauses.append(l.strip().split())
        return clauses


def main():
    litAndClauseNums = get_num_clauses(test_file)
    numLiterals = litAndClauseNums[0]
    numClauses = litAndClauseNums[1]
    print(numLiterals, numClauses)
    clauseList = make_array_of_clauses(test_file)
    print(clauseList)


main()