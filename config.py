class Config():
    def __init__(self):
        ##Parser
        #The parsers main function is write_input_file
        # wite_input_file takes a parameters file from base_params_path and a set of points in the form of a dict {param1:[point1,point2...], param2:[point1,point2...]...} 
        #  It will then create a parameters file that can scan over all the points.
        self.base_params_path = "/home/djdaniel/DEEPlasma/parameters_base_dependancies_fixed_dp" 
        self.remote_save_base_dir=f'/scratch/project_462000451/gene_out/gene_auto/'
        self.save_dir = "/home/djdaniel/DEEPlasma/temp/"


        ## Runner
        #The Runner is responsible for actually running a parameters file on lumi. Its main function is code_run.
        # code_run will take the set of points named samples and parse them into a parameters file.
        #  It then uses ssh to run GENE with this parametres file and a passed sbatch script.
        self.host = 'lumi' #needs to be configured in /home/<user>/.ssh/config
        self.sbatch_base_path = "/home/djdaniel/DEEPlasma/sbatch_base_dp" 
        # single_run_timelim = 200 # a guess for the number of seconds it takes to run one sample.
        self.remote_run_dir = '/project/project_462000451/gene/'
        self.local_run_files_dir = "/home/djdaniel/DEEPlasma/run_files"
        self.local_username = 'djdaniel' # used to identify if a directory is local or on remote.

        #Paramiko
        #Some SSH is done with bash commands using the .ssh/config host given above
        #Other SSH is done with paramiko, aka what I should have used from the start.
        #___________________
        import paramiko
        p_host = 'lumi.csc.fi'
        p_user = 'danieljordan'
        p_IdentityFile = '/home/djdaniel/.ssh/lumi-key'
        with open(p_IdentityFile, 'r') as key_file:
            pkey = paramiko.RSAKey.from_private_key(key_file)        
        # pkey = paramiko.RSAKey.from_private_key_file(p_IdentityFile, password=p_password)
        self.paramiko_ssh_client = paramiko.SSHClient()
        policy = paramiko.AutoAddPolicy()
        self.paramiko_ssh_client.set_missing_host_key_policy(policy)
        self.paramiko_ssh_client.connect(p_host, username=p_user, pkey=pkey)
        self.paramiko_sftp_client = self.paramiko_ssh_client.open_sftp()

        remote_params_dummy_file = self.paramiko_sftp_client.open('.bashrc') # any dummy file would serve this purpose
        self.paramiko_file_type = type(remote_params_dummy_file) 

        import os
        import sys
        print('Appending to path:', os.path.join(os.getcwd(),'config'))
        sys.path.append(os.path.join(os.getcwd(),'config'))

config = Config()

if __name__ == '__main__':
    config = Config()
    # print(config.save_dir)


    


    