import json
import plac
from mpi4py import MPI

TAG_CMD = 1
TAG_RES = 2
TAG_DATA = 3

def job_years(cfg, job):
    for year in job['years']:

        if cfg['nanhandling']['method'] == 'drop':
            unit = CSVPreprocessorUnit(year=year, dropnan=True)
        else:
            unit = CSVPreprocessorUnit(year=year)

        chunk_idx = 0

        path = "%s/%s_%d.csv" % (cfg['input_path'], cfg['input_prefix'], year)

        for chunk in pd.read_csv(path, chunksize=job['chunksize'], low_memory=False):

            chunk = unit.pipe(chunk)
            chunk_idx += 1

            for idx, site in enumerate(cfg['sites']):
                keys = list(chunk.keys())
                work = chunk[chunk['AQS_Code'] == site].values

                comm.isend({'keys': keys, 'input': work}, dest=2+idx, tag=TAG_DATA)

def job_site(cfg, job):
    pass

@plac.annotations(
    config_path=("Path to the json configuration file to use for the run", 'positional', None, str)
)
def main(config_path: str):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_procs = comm.Get_size()

    cfg = json.loads(open(config_path, 'r').read())

    if rank == 0:

        # rank 1 is the years parser
        comm.isend({'type': 'years', 'years': cfg['years']}, dest=1, tag=TAG_CMD)

        # the remaining ranks are the site processors
        mpi_dst = 2
        for site in cfg['sites']:
            comm.isend({'type': 'site', 'site': site}, dest=mpi_dst, tag=TAG_CMD)
            mpi_dst += 1

        # wait for everyone to finish
        while mpi_dst > 1:
            req = comm.irecv(source=0, tag=TAG_RES)
            job = req.wait()
            mpi_dst -= 1

        # issue the shutdown
        for mpi_dst in range(1, n_procs):
            comm.isend({'type': 'shutdown'}, dest=mpi_dst, tag=TAG_CMD)

    else:
        # wait for jobs
        while True:
            req = comm.irecv(source=0, tag=TAG_CMD)
            job = req.wait()

            if job['type'] == 'years':
                cmd_years(cfg, job)
            elif job['type'] == 'site':
                cmd_site(cfg, job)
            elif job['type'] == 'shutdown':
                break

            comm.isend({'type': 'response', 'rank': rank}, dest=0, tag=TAG_RES)


if __name__ == '__main__':
    plac.call(main)
