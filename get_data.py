import s3fs
import pathlib
import tarfile

data_local_path = pathlib.Path('data')

remote_paths = ["s3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_train.tar.gz",
                "s3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_test_public.tar.gz"]
local_paths = [data_local_path / 'SN7_buildings_train.tar.gz', data_local_path / 'SN7_buildings_test_public.tar.gz']

fs = s3fs.S3FileSystem(anon=True)
for local, remote in zip(local_paths, remote_paths):
    if not local.exists():
        print(f'Downloading zip {remote} ...')
        fs.download(remote, local.as_posix())

for fname in local_paths:
    print(f'Unzippping {fname} ...')
    with tarfile.open(fname, "r:gz") as tar:
        tar.extractall(data_local_path)
