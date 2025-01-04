# DEEPlasma
A repo for GENE surrogate model activities in the DEEPlasma Group

# Submodules
For detailed info on submodules https://git-scm.com/book/en/v2/Git-Tools-Submodules
This repo has a submodule GENE_ML. To clone this repo you must use the command:

HTTPS:
git clone --recurse-submodules https://yourtoken@github.com/DanHarJor/DEEPlasma/

SSH:
git clone --recurse-submodules git@github.com:DanHarJor/DEEPlasma.git

# SSH Security Tips for LUMI
If you have permission issues can can add this to your  /users/<username>/.bashrc

**export GIT_SSH_COMMAND="ssh -i ~/.ssh/<private_key>"**

Now when git tries to ssh during the clone process it will always have access to your private key.
# Common Errors:

Each variable name must be followed by a space character, eg:
yes
coll = 0.1
no
coll=0.1
This would cause a problem with parser.write_input_file(), it is to avoid coll being confused with collision_op

# Version Management Guidelines

## DEEPlasma
No specific version management instructions as this mainly holds notebooks and the real code is in GENE_ML

Just only edit your own notebooks and we shouldn't have merge issues.

## GENE_ML 
main is for stable code only  
  
Develop tests for your code as your write it, I use a notebook for this.  
  
Development branch is for merging multiple feature branches together and testing before merging to main branch.    

Make a new feature branch for each feature **IMPORTANT Make sure the files being edited are different for each feature. If two features must share the same files then either focus on one before starting the other or use one branch for both features**    
  
## Push to remote often to keep it safe online  
