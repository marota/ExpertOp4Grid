# Run function as a process
import multiprocessing
import time
if __name__ == '__main__':
    # Start foo as a process
    from alphadeesp_process import alphadeesp_process

    p = multiprocessing.Process(target=alphadeesp_process, name="AlphadeespProcess", args=())
    p.start()

    seconds = 300

    # Wait amount of seconds for process
    time.sleep(seconds)

    # Terminate process
    p.terminate()

    # Cleanup
    p.join()
    print("Interrupted after : "+str(seconds)+" seconds")


