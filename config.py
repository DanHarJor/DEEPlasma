class Config():
    def __init__(self):
        ##Parser
        #The parsers main function is write_input_file
        # wite_input_file takes a parameters file from base_params_path and a set of points in the form of a dict {param1:[point1,point2...], param2:[point1,point2...]...} 
        #  It will then create a parameters file that can scan over all the points.
        self.base_params_path = "/home/djdaniel/DEEPlasma/parameters_base_dp" 
        self.remote_save_base_dir=f'/scratch/project_462000451/gene_out/gene_auto/'
        self.save_dir = "/home/djdaniel/DEEPlasma/temp/"


        ## Runner
        #The Runner is responsible for actually running a parameters file on lumi. Its main function is code_run.
        # code_run will take the set of points named samples and parse them into a parameters file.
        #  It then uses ssh to run GENE with this parametres file and a passed sbatch script.
        self.host = 'lumi1' #needs to be configured in /home/<user>/.ssh/config
        self.sbatch_base_path = "/home/djdaniel/DEEPlasma/sbatch_base_dp" 
        # guess_sample_wallseconds = 200 # a guess for the number of seconds it takes to run one sample.
        self.remote_run_dir = '/project/project_462000451/gene/'
        self.local_run_files_dir = "/home/djdaniel/DEEPlasma/run_files"

config = Config()

if __name__ == '__main__':
    config = Config()
    print(config.save_dir)


    


    