import qmpy_rester as qr
import json
import time
import os
import argparse
import sys

PAGE_LIMIT = 50
def download_by_batch(batch_num):
    t1 = time.time()
    with qr.QMPYRester() as q:
        kwargs = {'limit':PAGE_LIMIT, 
                  'offset': batch_num*PAGE_LIMIT,
                  #'_oqmd_band_gap': '<2.2',         # band gap less than 2.2
                  #'_oqmd_band_gap': '>=2.2',         # band gap greater than or equal to 2.2
                  '_oqmd_natoms': '>9',             # more than 9 atoms
                  }

        tries = 0
        while tries < 4:
            tries += 1
            try:
                data = q.get_optimade_structures(verbose=False, **kwargs)
                if data is None:
                    raise Exception('no data received')
                break
            except:
                print(sys.exc_info())

    t2 = time.time()

    if batch_num == 0:
        print('Size of query dataset is %d.'%data['meta']['data_available'])
    
    with open('query_files9/'+str(batch_num)+'.json', 'w') as json_file:
        json.dump(data['data'], json_file, indent=2)
    
    print('Loading Batch %d time %.3f seconds'%(batch_num, t2-t1))

    if data['links']['next']:
        return True
    else:
        return False

if __name__ == "__main__":
    if not os.path.exists('query_files9'):
        os.mkdir('query_files9')

    batch_num = 233
    while download_by_batch(batch_num):
        batch_num = batch_num + 1