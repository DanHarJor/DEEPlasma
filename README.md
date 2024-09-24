# DEEPlasma
A repo for GENE surrogate model activities in the DEEPlasma Group

# Submodules
For detailed info on submodules https://git-scm.com/book/en/v2/Git-Tools-Submodules
This repo has a submodule GENE_ML. To clone this repo you must use the command:

git clone --recurse-submodules https://yourtoken@github.com/DanHarJor/DEEPlasma/

# Common Errors:

Each variable name must be followed by a space character, eg:
yes
coll = 0.1
no
coll=0.1
This would cause a problem with parser.write_input_file(), it is to avoid coll being confused with collision_op

