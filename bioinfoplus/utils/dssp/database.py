"""
The DSSP code(for field STRUCTURE in output dataframes):
    H = alpha helix
    B = residue in isolated beta-bridge
    E = extended strand, participates in beta ladder
    G = 3-helix (3/10 helix)
    I = 5 helix (pi helix)
    T = hydrogen bonded turn
S = bend
"""
import urllib.request
import sys
import re
import gzip
import io
import os
from wisepy.talking import Talking

talking = Talking()


@talking
def do(o: 'download path' = 'data',
       *,
       page_url:
       'page url' = 'ftp://ftp.cbi.pku.edu.cn/pub/database/DSSP/20130820'):
    if os.path.exists(o) and not os.path.isdir(o):
        raise ValueError("Invalid output path.")

    if not os.path.exists(o):
        os.makedirs(o)

    directory = urllib.request.urlopen(page_url)
    matcher = re.compile("([\w]+\.dssp\.gz)")
    file_names = directory.read().decode("utf-8")
    file_names = matcher.findall(file_names)

    def store_file(filename: str):
        url = f"{page_url}/{filename}"
        try:
            content = urllib.request.urlopen(url).read()
        except Exception as e:
            print(
                f"error {e} find: in storing processing. filename : {filename}"
            )
            sys.exit(1)

        with io.BytesIO(content) as f:
            unzipped = gzip.GzipFile(fileobj=f)
            with open(f'{o}/{filename}.dssp', 'wb') as f:
                f.write(unzipped.read())
        print(f'data `{filename}` downloaded.')

    for each in file_names:
        store_file(each)


if __name__ == '__main__':
    talking.on()
