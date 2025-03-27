from dask_jobqueue import SLURMCluster

# custom_worker_command = "bash -c 'export PATH=/project/project_462000451/enchanted_container_lumi3/bin:$PATH && python -m distributed.cli.dask_worker'"

prologue_commands = [
    # "module load your_module_name",
    "echo 'checking squeue --me'",
    'squeue --me',
    
    "echo 'loading cray-python and activating enviroment'",
    "module load cray-python",
    "source /project/project_462000451/venv_lumi_cray_python/bin/activate",
    
    # "echo 'exporting path for container'",
    # "export PATH='/project/project_462000451/enchanted_container_lumi3/bin:$PATH'",
    # "echo 'checking dask can be imported'",
    # "python3 /project/project_462000451/DEEPlasma/check_dask_imported.py",
    # "python3 -m distributed.cli.dask_worker"
]
cluster = SLURMCluster(
    queue='standard',
    account="project_462000451",
    cores=128,
    memory="200GB",
    interface= "lo",
    # death_timeout=2,
    # job_extra=['--preload', 'worker_setup.py']
    # worker_command = custom_worker_command
    job_script_prologue=prologue_commands
)

#['lo', 'nmn0', 'hsn1', 'hsn0', 'can0', 'bond0', 'net1', 'net2', 'net3']
cluster.scale(jobs=3)  # ask for 10 jobs

from dask.distributed import Client
client = Client(cluster)
print(client)