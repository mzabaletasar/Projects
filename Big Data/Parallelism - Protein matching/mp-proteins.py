import time    
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp


# Functions used
def read_and_find(i, patron):
    df = pd.read_csv('proteins.csv', 
                 skiprows=i, nrows=int(500000/num_workers), header=None)
    df = df.rename(columns={0:'structureId', 1:'sequence'})
    
    df['count'] = df['sequence'].str.count(patron)
    occs = df[df['count'] > 0].sort_values('count', ascending=False)
    return occs

def collect_result(result):
    global results
    results.append(result)

results = []
num_workers = mp.cpu_count()


def main():
    pattern = input('Input the pattern to be searched: ')
    pattern = pattern.upper()    
    
    # Serial part to compute execution time
    st_serial = time.time()   
    df = pd.read_csv('proteins.csv')
    df['count'] = df['sequence'].str.count(pattern)
    occs = df[df['count'] > 0].sort_values(['count', 'structureId'], ascending=[False, True])
    et_serial = time.time()
    
    # Multiprocessing part                                               
    st_multiproc = time.time()
    pool = mp.Pool(mp.cpu_count())
            
    for k in [int(j*500000/num_workers)+1 for j in range(num_workers)]:
        pool.apply_async(read_and_find, args=(k, pattern), callback=collect_result)
                
    pool.close()   
    pool.join()
    et_multiproc = time.time()  
            
    occs_multiproc = pd.concat(results).sort_values(['count', 'structureId'], ascending=[False, True])
       
    # Create barplot with multiprocessing results (and print results)
    # to be sure it is the same as the serial
    if occs_multiproc.shape[0] > 0:
        labels = occs_multiproc[:10].sort_values('structureId', ascending=True)
        plt.rcParams.update({'font.size':8})
        n = labels.shape[0]
        plt.bar(range(n), height=labels['count'], tick_label=labels['structureId'])
        plt.title('10 proteins with more occurrences')
        plt.xlabel('Protein id')
        plt.ylabel('Number of occurrences')
            
        max_occ_multiproc = occs_multiproc[occs_multiproc['count'] == occs_multiproc['count'].values[0]]
        print('\nProteins with max num of occurrences (%d occurrences %d times):' % (occs_multiproc['count'].values[0],
                                                                                       max_occ_multiproc.shape[0]))
        for i in range(len(max_occ_multiproc.values)):
            print('Structure Id:', max_occ_multiproc.sort_values('structureId',ascending=True).values[i][0], ';',
                      ' Protein: ', max_occ_multiproc.sort_values('structureId',ascending=True).values[i][1])
    else:
        print('No occurrences found')
    
    # Print execution time in multiprocessing and speedup
    print('Execution time multiproc, ', num_workers, 'workers:', et_multiproc-st_multiproc)
    print('\nSpeedup: ', (et_serial-st_serial)/(et_multiproc-st_multiproc))

if __name__ == '__main__':
    main()
        