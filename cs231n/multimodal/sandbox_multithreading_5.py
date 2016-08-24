import threading
import Queue
import time


def run_experiment(exp):
    print "running {}".format(exp)
    time.sleep(7)  # simulate that the task takes 7 seconds

    return


def worker():
    while True:
        item = q.get()
        run_experiment(item)
        q.task_done()
        # write to database that this item has finished
        print "finished item {}".format(item)


def main():
    # get list of experiment conditions (from database)
    exp_configs = []
    for i in range(10):
        exp_configs.append('exp_{}'.format(i))

    # set number of worker threads
    num_worker_threads = 4

    # initialize the queue
    global q
    q = Queue.Queue()
    # initialize the worker threads
    for i in range(num_worker_threads):
        print "start thread {}".format(i)
        t = threading.Thread(target=worker)
        t.daemon = True
        t.start()

    # put experiments in the queue
    for exp in exp_configs:
        q.put(exp)

    q.join()       # block until all tasks are done
    return

main()