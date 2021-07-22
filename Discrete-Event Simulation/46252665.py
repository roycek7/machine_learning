import heapq
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('ggplot')  # ggplot style

dataframe = pd.read_csv('data.csv')  # load dataset containing inter-arrival and service time

T = 3000  # simulation time
service_desk = 7  # final service desk
N = 50  # number of batches
selected_portion = 70  # discarding first 30% of sequence
BUSY = -1  # representative value indicates all busy service desk workers
ARRIVAL = 'Arrival'  # name of customer arrival event
DEPARTURE = 'Departure'  # name of customer departure event
events = []  # event list to generate sequence
free_time = []  # time spent by service desk available


# service desk class object containing all information within simulation process
class ServiceDesk:
    def __init__(self, _id, arrive):
        self.id = _id  # customer identification number
        self.arrival_time = arrive  # time of customer arrival
        self.event_type = 'Arrival'  # customer status, updates according to simulation
        self.service_start = -1  # time when a customer is served
        self.service_end = -1  # time when a customer is finished serving
        self.queue = -1  # queue information of customer attendance

    def pretty_print(self):
        print(f'customer_id: {self.id}, state: {self.event_type.upper()}, arrival_time: {self.arrival_time}, '
              f'queue: {self.queue}, service_start: {self.service_start}, service_end: {self.service_end}')


# arrival time for the next customer based on the inter-arrival time in the given dataset
def arrival_time():
    return random.choices(dataframe.inter_arrival_time.tolist(), k=1)[0]


# service time for the next customer based on the service time in the given dataset
def service_time():
    return random.choices(dataframe.service_time.tolist(), k=1)[0]


# check whether any service desk is free and return index, otherwise all busy status
def check_service_desk(desk_status):
    for desk_id, desk_available in enumerate(desk_status):
        if desk_available:
            return desk_id
    return BUSY


# set the service desk worker to available and return all the status of the service desks
def free_desk(desk_status, worker):
    desk_status[worker] = True
    return desk_status


# check the length of all service desks, and return the shortest one based on being served or not, otherwise
# assign to the min queue length
def shortest_queue(queue_size, status):
    queue_length = [len(i) for i in queue_size]
    empty_queue = check_service_desk(status)
    if empty_queue != BUSY:
        return empty_queue
    return queue_length.index(min(queue_length))


# simulation of the whole process of air secure from time 0 to T and generates sequence of events from all
# customers while updating the variables in the service desk class
def simulate():
    desk_status = [True] * service_desk  # availability of service desks, True indicates free to serve
    event_queue = []  # stores all upcoming events to be processed
    regular_queue = [[] for i in range(service_desk)]  # customers waiting in the queue of the service desk
    queue_service_time = [0] * service_desk  # the service end time of the desk

    _time = 0  # time variable which changes with respect to the simulation process
    _id = 1  # customer id initialised

    # first arrival of customer
    first_arrival_time = arrival_time()
    service = ServiceDesk(_id, first_arrival_time)
    heapq.heappush(event_queue, (first_arrival_time, service))

    # simulation process
    while _time < T:
        objects = heapq.heappop(event_queue)
        _time = objects[0]
        service = objects[1]

        events.append(service)  # store events to list for creating dataframe

        # if the current event is 'arrival', generate next arrival time for the new customer and serve if there
        # exist and available service desk worker. If the workers are busy put them in the shortest queue prioritising
        # them to be served in an empty queue
        if service.event_type == ARRIVAL:
            _id += 1
            new_arrival_time = _time + arrival_time()
            new_service = ServiceDesk(_id, new_arrival_time)
            heapq.heappush(event_queue, (new_arrival_time, new_service))
            worker = check_service_desk(desk_status)
            if worker != BUSY:
                # service desk worker currently serving, not available
                desk_status[worker] = False

                # updating service desk class attributes
                service.event_type = DEPARTURE
                service.service_start = _time
                service.service_end = _time + service_time()
                service.queue = worker + 1

                # if the event already exist in the fifo queue, resist from pushing again
                if service.service_end not in [i[0] for i in event_queue]:
                    heapq.heappush(event_queue, (service.service_end, service))
            else:
                wait_queue = shortest_queue(regular_queue, desk_status)
                regular_queue[wait_queue].append(service)
            continue

        # if there is are customer's event as 'departure' in the service desk queue, check if the service desk worker
        # is already serving some customer, otherwise serve
        if service.event_type == DEPARTURE:
            for idx, waiting_queue in enumerate(regular_queue):
                if len(waiting_queue) > 0:
                    if _time >= queue_service_time[idx]:
                        # free service desk workers
                        worker = regular_queue.index(waiting_queue)
                        desk_status = free_desk(desk_status, worker)

                        # get event
                        service_complete = waiting_queue.pop(0)

                        # updating service desk class attributes
                        service_complete.service_start = _time
                        service_complete.service_end = _time + service_time()
                        service_complete.event_type = DEPARTURE
                        service_complete.queue = worker + 1

                        # update desk latest service time
                        queue_service_time[worker] = service_complete.service_end

                        # if the event already exist in the fifo queue, resist from pushing again
                        if service_complete.service_end not in [i[0] for i in event_queue]:
                            heapq.heappush(event_queue, (service_complete.service_end, service_complete))

                        # service desk worker currently serving, not available
                        desk_status[worker] = False

            continue


# main simulation function
simulate()

# create list for dataframe columns
arrival_time_list, service_start_list, service_end_list, queue = [], [], [], []
for ev in events:
    if ev.event_type == DEPARTURE:
        service_start_list.append(ev.service_start), arrival_time_list.append(ev.arrival_time), service_end_list. \
            append(ev.service_end), queue.append(ev.queue)

# dataframe for the sequence of events
waiting_dataframe = pd.DataFrame(
    {'arrival_time': arrival_time_list,
     'service_start': service_start_list,
     'service_end': service_end_list,
     'queue': queue
     })

# create column wait time, service_start - arrival_time
waiting_dataframe = waiting_dataframe.drop_duplicates()
waiting_dataframe = waiting_dataframe.reset_index(drop=True)
waiting_dataframe['wait_time'] = waiting_dataframe['service_start'] - waiting_dataframe['arrival_time']

print('----------------------------------------------OUTPUT--------------------------------------------------')

waiting_dataframe = waiting_dataframe.assign(
    next_customer_service=waiting_dataframe.groupby('queue').service_start.shift(-1))

# Discarding first 30% of the simulation
waiting_dataframe = waiting_dataframe.tail(int(len(waiting_dataframe) * (selected_portion / 100)))
print(f'Sequence of Events: \n{waiting_dataframe}')

# 90% of user waiting less than 8 minutes
wait_times = waiting_dataframe['wait_time'].tolist()
count = 0
for i in wait_times:
    if i < 8:
        count += 1
print(f'Percentage of customers waiting less than 8 minutes: {count * 100 / len(wait_times)}')
waiting_mean = np.mean(waiting_dataframe['wait_time'].to_numpy())
standard_deviation = np.std(waiting_dataframe['wait_time'].to_numpy()) / \
                     np.sqrt(len(waiting_dataframe['wait_time'].to_numpy()))

print('-----------------------------------------WAITING TIME-------------------------------------------------')

print(f"90% Customer's waiting time with {service_desk} service desk: {waiting_mean}")
print(f"95% CI for total time in the system: ({waiting_mean - 1.96 * standard_deviation}, "
      f"{waiting_mean + 1.96 * standard_deviation})")

# Probabilities of 50 Batches
batches = int(len(wait_times) / N)
probabilities = []
for i in range(0, len(wait_times), batches):
    if len(probabilities) < N:
        count = 0
        for j in wait_times[i: i + batches]:
            if int(j) < 8:
                count += 1
        probabilities.append(count / batches)

print('----------------------------------------PROBABILITIES-------------------------------------------------')
print(f'Probability that a customer waits less than 8 minutes before entering the service for {N} batches: \n'
      f'{probabilities}')

# plot probability vs batches
_fig = plt.figure()
plt.plot(range(N), probabilities)
plt.xlabel('batches')
plt.ylabel('probability')
_fig.savefig('probabilities.png')

print('----------------------------------------95% CI PER QUEUE----------------------------------------------')

# queue information
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
fig.subplots_adjust(hspace=0.25, wspace=0.15)

axs = axs.ravel()
for i in range(1, service_desk + 1):
    queue = waiting_dataframe.loc[waiting_dataframe['queue'] == i]

    # subplots of queues waiting time
    axs[i - 1].plot(range(queue.shape[0]), queue['wait_time'].tolist())
    axs[i - 1].set_title(f'Queue {i}')

    # queue waiting time 95% confidence interval
    queue_mean = np.mean(queue['wait_time'].to_numpy())
    queue_standard_deviation = np.std(queue['wait_time'].to_numpy()) / \
                               np.sqrt(len(queue['wait_time'].to_numpy()))
    print(f"95% CI for total time in queue {i}: ({queue_mean - 1.96 * queue_standard_deviation}, "
          f"{queue_mean + 1.96 * queue_standard_deviation})")

    # insert time available
    free_time.append((queue['next_customer_service'] - queue['service_end']).sum())

# x-axis, y-axis and saving figure
fig.text(0.5, 0.04, 'customer', ha='center')
fig.text(0.04, 0.5, 'wait time', va='center', rotation='vertical')
fig.savefig(f'queues.png')

# plot availability of service desk by time
fig_ = plt.figure()
plt.plot(range(1, service_desk + 1), free_time)
plt.bar(range(1, service_desk + 1), free_time)
plt.ylabel('free_time')
plt.xlabel('service desk')
fig_.savefig(f'free_time.png')
print('------------------------------------------------------------------------------------------------------')
