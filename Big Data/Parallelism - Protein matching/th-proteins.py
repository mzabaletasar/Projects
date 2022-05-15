import threading as th
import pandas as pd
import time 
import matplotlib.pyplot as plt
import multiprocessing as mp


# Function used
def read_and_find(i, patron):
    df = pd.read_csv('proteins.csv', 
                         skiprows=i, nrows=int(500000/num_threads), header=None)
    df = df.rename(columns={0:'structureId', 1:'sequence'})
    df['count'] = df['sequence'].str.count(patron)
    occs = df[df['count'] > 0].sort_values('count', ascending=False) 
    results.append(occs)      

results = []                                        
num_threads = mp.cpu_count() - 1  

    
def main():
    pattern = input('Input the pattern to be searched: ')
    pattern = pattern.upper()
     
    # Serial part to compute execution time
    st_serial = time.time()
    df = pd.read_csv('proteins.csv')
    df['count'] = df['sequence'].str.count(pattern)
    occs = df[df['count'] > 0].sort_values(['count', 'structureId'], ascending=[False, True])
    et_serial = time.time()

    # Threads part
    st_threads = time.time()
    threads = []
    
    for k in range(num_threads):
        t = th.Thread(target=read_and_find, args = ((k*int(500000/num_threads))+1, pattern))
        t.start()
        threads.append(t)
        
    for thread in threads:
        thread.join()
            
    et_threads = time.time()
    
    occs_threads = pd.concat(results)
    occs_threads = occs_threads.sort_values(['count', 'structureId'], ascending=[False, True])
    
    # Create barplot with threads results (and print results)
    # to be sure it is the same as the serial
    if occs_threads.shape[0] > 0:
        labels = occs_threads[:10].sort_values('structureId', ascending=True)
        plt.rcParams.update({'font.size':8})
        n = labels.shape[0]
        plt.bar(range(n), height=labels['count'], tick_label=labels['structureId'])
        plt.title('10 proteins with more occurrences')
        plt.xlabel('Protein id')
        plt.ylabel('Number of occurrences')
        
        max_occ_threads = occs_threads[occs_threads['count'] == occs_threads['count'].values[0]]
        print('\nProteins with max num of occurrences (%d occurrences %d times):' % (occs_threads['count'].values[0],
                                                                                   max_occ_threads.shape[0]))
        for i in range(len(max_occ_threads.values)):
            print('Structure Id:', max_occ_threads.sort_values('structureId',ascending=True).values[i][0], ';',
                  ' Protein: ', max_occ_threads.sort_values('structureId',ascending=True).values[i][1])
    else:
        print('No occurrences found')
    
    # Print execution time in threads and speedup
    print('Execution time,', num_threads, 'threads: ', et_threads-st_threads)
    print('\nSpeedup: ', (et_serial-st_serial)/(et_threads-st_threads))
        
if __name__ == '__main__':
    main()