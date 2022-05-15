import time    
import pandas as pd
import matplotlib.pyplot as plt


pattern = input('Input the pattern to be searched: ')
pattern = pattern.upper()

st = time.time()   

# Read file and find occurrences
df = pd.read_csv('proteins.csv')
df['count'] = df['sequence'].str.count(pattern)
occs = df[df['count'] > 0].sort_values(['count', 'structureId'], ascending=[False, True])

et = time.time()

if occs.shape[0] > 0:
    
    # Get labels for barplot and plot it
    labels = occs[:10].sort_values('structureId', ascending=True)
    
    plt.rcParams.update({'font.size':8})
    n = labels.shape[0]
    plt.bar(range(n), height=labels['count'], tick_label=labels['structureId'])
    plt.title('10 proteins with more occurrences')
    plt.xlabel('Protein id')
    plt.ylabel('Number of occurrences')
    
    # Get proteins with max number of occurrences and print them
    max_occ = occs[occs['count']==occs['count'].values[0]]

    print('Proteins with max num of occurrences (%d occurrences %d times):' % (occs['count'].values[0],
                                                                                   max_occ.shape[0]))
    for i in range(len(max_occ.values)):
        print('Structure Id:', max_occ.sort_values('structureId',ascending=True).values[i][0], ';',
                  ' Protein: ', max_occ.sort_values('structureId',ascending=True).values[i][1])

    # Print execution time
    print('Execution time: ', et-st)

else:
    print('No occurrences found.')