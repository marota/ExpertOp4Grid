# Run function as a process
import multiprocessing
import time
import sys
import cProfile
import pstats
import io
import os
import datetime
from threading import Timer
from alphadeesp_process import alphadeesp_process

# p = multiprocessing.Process(target=alphadeesp_process, name="AlphadeespProcess", args=())
# p.start()
#

#
# # Wait amount of seconds for process
# time.sleep(seconds)
#
# # Terminate process
# p.terminate()

# Cleanup
#p.join()




### RUNTIME
seconds = 20

### INIT PROFILER
pr = cProfile.Profile()
pr.enable()

### EXIT PROCESS when time is over
def exitfunc():
    print("Interrupted after : "+str(seconds)+" seconds")
    pr.disable()
    result = io.StringIO()
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=result).sort_stats(sortby)
    ps.print_stats()
    result = result.getvalue()

    # Write pstats file
    filename = "runtime_wcci_"+str(seconds)+"s.pstats"
    filename_csv = filename.replace(".pstats",".csv")
    filename_png = filename.replace(".pstats", ".png")
    pr.dump_stats(filename)

    # Write dot graph file
    os.system('gprof2dot -f pstats '+filename+' | dot -Tpng -o '+filename_png)

    # Write csv file
    # chop the string into a csv-like buffer
    result = 'ncalls' + result.split('ncalls')[-1]
    result = '\n'.join([';'.join(line.rstrip().split(None, 5)) for line in result.split('\n')])
    # save it to disk

    with open(filename_csv, 'w+') as f:
        # f=open(result.rsplit('.')[0]+'.csv','w')
        f.write(result)
        f.close()
    os._exit(0)

### Proceed with TIMER
Timer(seconds, exitfunc).start()
alphadeesp_process()



