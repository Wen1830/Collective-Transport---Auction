import numpy as np
import random
from matplotlib import pyplot as plt
import itertools
import time as ct




def RND_Walk(n,m,location,speed,deadline,q,C,e,ee,D,timemax,ferry,dist):
    
    avail_task = [[] for i in range(m)]
    for i in range(m):
        avail_task[i] = list(range(0,n))
    
    # List 'unavail' is for recording the tasks that cannot be completed
    unavail = [[] for i in range(m)]
    
    
    # The list 'load' is for recording the tasks assigned to each robot
    # For example load=[[4],[3]] represents task 4 is assigned to robot 1 and
    # task 3 is assigned to robot 2 
    load = [[] for i in range(m)]
    
    # List 'd' is for recording the distance travelled for each robot
    d = np.empty(m)
    
    # Record makespan
    finish_time = np.empty(m)
    
    
    
    nearest_task = dist[-1,0:n].copy()
    for i in range(m):
        nearest_index = np.argmin(nearest_task)
        load[i].append(2*n)
        load[i].append(nearest_index)
        
        
        d[i] = (dist[-1,load[i][1]])
            
        nearest_task[nearest_index] = 2*D
        
        
        e[nearest_index] = e[nearest_index] - 1
        avail_task[i].append(nearest_index+n)
        if e[nearest_index] <= 0:
            for ii in range(m):
                avail_task[ii].remove(nearest_index)
    
    
    # Update robots state
    capacity = np.empty(m)
    time = np.empty(m)
    travel_dist = np.empty(m)
    for i in range(m):
        capacity[i] = q[load[i][-1]]
        time[i] = d[i]/speed
        finish_time[i] = d[i]/speed
        travel_dist[i] = dist[load[i][0],load[i][1]]
    
    def disttravel(tour,dist,speed,n,load,robot_index,time):
        # The input is the tour with task index and the distance matrix
        tour = list(tour)
        tour.append(2*n)
        tour.insert(0, load[robot_index][-1])
        l = len(tour)
        t = time[robot_index]
        T = [time[robot_index]]
        for i in range(l-1):
            t = t + dist[tour[i],tour[i+1]]/speed
            T.append(dist[tour[i],tour[i+1]]/speed)
        T = np.cumsum(T)
        t = t/speed
        return t,T
        
    
    def range_check(next_task,robot_index,load,time,active_task,deadline,dist):
        '''Check range constraint and ensure ending points can be gauranteed
        to be completed.'''
        
        aim_task = active_task.copy()
        aim_task = [i for i in aim_task if i >= n]
        
        
        if len(aim_task) == 0:
            distance = dist[load[robot_index][-1],next_task] + dist[next_task,next_task+n] + dist[next_task+n,2*n]
            distance = distance + d[robot_index]
            t1 = time[robot_index] + dist[load[robot_index][-1],next_task]/speed
            t2 = t1 + dist[next_task,next_task+n]/speed
            checktime = deadline[[next_task,next_task+n]] - [t1,t2]
            if distance < ferry and any(ii > 0 for ii in checktime):
                return 1
            else:
                return 0
        
        elif len(aim_task) > 0:
            if next_task >= n:
                return 1
            else:
                aim_task.append(next_task+n)
                routes = itertools.permutations(aim_task)
                routes = list(routes)
                nr = len(routes)
                tt = []
                TT = [[] for i in range(nr)]
                for i in range(nr):
                    route = list(routes[i])
                    route.insert(0,next_task)
                    t,T = disttravel(route,dist,speed,n,load,robot_index,time)
                    tt.append(t)
                    TT[i] = T
                shortest_index = np.argmin(tt)
                worst_index = np.argmax(tt)
                shortest_route = routes[shortest_index]
                shortest_route = list(shortest_route)
                worst_route = routes[worst_index]
                worst_route = list(worst_route)
                
                worst_TT = TT[worst_index]
                worst_TT = deadline[worst_route] - TT[worst_index][2:-1]
                worst_time = tt[worst_index] #+ d[robot_index]/speed +dist[load[robot_index][-1],next_task]/speed
                
                optimal_time = tt[shortest_index] #+ d[robot_index]/speed +dist[load[robot_index][-1],next_task]/speed
                optimal_TT = TT[shortest_index]
                optimal_TT = deadline[shortest_route] - optimal_TT[2:-1]
                
                if any(ii > 0 for ii in worst_TT) and worst_time/speed < ferry:
                    return 1
                else:
                    if any(ii > 0 for ii in optimal_TT) and optimal_time/speed < ferry:
                        return 1
                    else:
                        return 0
                    
                    
    def feasibleTask(robot_index,n,m,load,active_task,avail_task,unavail,dist,capacity,deadline,time):
        
        
        'Find all available task'
        point = avail_task[robot_index].copy()
        # for i in [j for j in avail_task[robot_index] if j < n]:
        #     point.append(i)
        # ending_active = [i for i in active_task if i >= n]
        # for i in range(len(ending_active)):
        #     point.append(ending_active[i])
        
        
        'Exclude any task that is already expired for the auction'
        expire = np.argwhere(deadline[point]<time[robot_index])
        if np.size(expire) > 0:
            expire = expire.tolist()
            expire_task = []
            for i in range(len(expire)):
                expire_task.append(point[expire[i][0]])
            for i in range(len(expire)):
                point.remove(expire_task[i])
                avail_task[robot_index].remove(expire_task[i])
                unavail[robot_index].append(expire_task[i])
                if expire_task[i] < n:
                    #point.remove(expire_task[i]+n)
                    unavail[robot_index].append(expire_task[i]+n)
        
        cant_reach = point.copy()
        for i in range(len(point)):
            if deadline[cant_reach[i]] - (dist[load[robot_index][-1],cant_reach[i]]/speed+time[robot_index]) <= 0:
                point.remove(cant_reach[i])
                avail_task[robot_index].remove(cant_reach[i])
                unavail[robot_index].append(cant_reach[i])
                if cant_reach[i] < n:
                    #point.remove(cant_reach[i]+n)
                    unavail[robot_index].append(cant_reach[i]+n)
        
                
        'Exclude any task that exceeds the capacity of current robot for auction'
        for i in [ii for ii in point if ii < n]:
            if capacity[robot_index] + q[i] > C:
                point.remove(i)
                
        
        'Bid Calculation'
        nn = len(point)
        if nn == 0:
            next_task = 2*n
            load[robot_index].append(next_task)
            finish_time[robot_index] = time[robot_index] + dist[load[robot_index][-1],next_task]/speed
            time[robot_index] = 9999
            return next_task,avail_task,unavail,time,capacity,load
        
        
        
        index = random.randint(0,nn-1)
        next_task = point[index]
        
        
        'Range Check'
        rangecheck = range_check(next_task,robot_index,load,time,active_task,deadline,dist)
        if rangecheck == 1:
            pass
        else:
            nearest = dist[robot_index,active_task]
            nearest_index = np.argmin(nearest)
            next_task = active_task[nearest_index]
        
        
        
        'Update state information'
        if e[next_task] <= 1 and next_task < n:
            for ii in range(m):
                if avail_task[ii].count(next_task) > 0:
                    avail_task[ii].remove(next_task)
            avail_task[robot_index].append(next_task+n)
            e[next_task] = e[next_task] - 1
        elif e[next_task] <= 1 and next_task >= n:
            e[next_task] = e[next_task] - 1
            avail_task[robot_index].remove(next_task)
        elif e[next_task] > 1 and next_task < n:
            e[next_task] = e[next_task] - 1
            avail_task[robot_index].append(next_task+n)
        elif e[next_task] > 1 and next_task >= n:
            e[next_task] = e[next_task] - 1
            avail_task[robot_index].remove(next_task)
            
        
        time[robot_index] = time[robot_index] + dist[load[robot_index][-1],next_task]/speed
        finish_time[robot_index] = finish_time[robot_index] + dist[load[robot_index][-1],next_task]/speed
        d[robot_index] = d[robot_index] + dist[load[robot_index][-1],next_task]
        travel_dist[robot_index] = travel_dist[robot_index] + dist[load[robot_index][-1],next_task]
        capacity[robot_index] = capacity[robot_index] + q[next_task]
    
        
        load[robot_index].append(next_task)
    
        return next_task,avail_task,unavail,time,capacity,load
    
    
    
    def forcefinish(robot_index,n,m,load,active_task,avail_task,unavail,dist,capacity,deadline,time):
        '''After robots reach 75% of the maximum travel range, force robots to finish
        the remaining tasks and then return to the depot.'''
        
        
        active_task = [i for i in active_task if i > n]
        nn = len(active_task)
        routes = itertools.permutations(active_task)
        routes = list(routes)
        nr = len(routes)
        tt = []
        for i in range(nr):
            t,T = disttravel(routes[i],dist,speed,n,load,robot_index,time)
            tt.append(t)
        shortest_index = np.argmin(tt)
        shortest_route = routes[shortest_index]
        shortest_route = list(shortest_route)
        
        
        #robot_list = list(range(m))
        #exclude_robot = [i for i in robot_list if i != robot_index]
        for i in range(nn):
            next_task = shortest_route[i]
            capacity[robot_index] = capacity[robot_index] + q[next_task]
            load[robot_index].append(next_task)
            # if e[next_task] <= 1 and next_task < n:
            #     for ii in range(m):
            #         if avail_task[ii].count(next_task) > 0:
            #             avail_task[ii].remove(next_task)
            #     for jj in exclude_robot:
            #         if avail_task[jj].count(next_task) > 0:
            #             avail_task[jj].remove(next_task+n)
            #     e[next_task] = e[next_task] - 1
            # elif e[next_task] <= 1 and next_task >= n:
            #     e[next_task] = e[next_task] - 1
            #     avail_task[robot_index].remove(next_task)
            # elif e[next_task] > 1:
            #     e[next_task] = e[next_task] - 1
            #     #avail_task[robot_index].remove(next_task)
            
            if e[next_task] <= 1 and next_task < n:
                for ii in range(m):
                    if avail_task[ii].count(next_task) > 0:
                        avail_task[ii].remove(next_task)
                avail_task[robot_index].append(next_task+n)
                e[next_task] = e[next_task] - 1
            elif e[next_task] <= 1 and next_task >= n:
                e[next_task] = e[next_task] - 1
                avail_task[robot_index].remove(next_task)
            elif e[next_task] > 1 and next_task < n:
                e[next_task] = e[next_task] - 1
                avail_task[robot_index].append(next_task+n)
            elif e[next_task] > 1 and next_task >= n:
                e[next_task] = e[next_task] - 1
                avail_task[robot_index].remove(next_task)
            
        
        time[robot_index] = time[robot_index] + tt[shortest_index]
        travel_dist[robot_index] = travel_dist[robot_index] + tt[shortest_index]*speed
        finish_time[robot_index] = finish_time[robot_index] + tt[shortest_index]
        d[robot_index] = 0
        
        return avail_task,unavail,time,capacity,load
            
    
    
    start = ct.time()
    while np.size(avail_task) > 0:
        
        
        robot_index = np.argwhere(time==min(time))
        robot_index = robot_index[0][0]
        robot_index = robot_index.tolist()
        
        # active_task = activetask(robot_index,load)
        active_task = avail_task[robot_index]
        
        if d[robot_index] >= 0.75*ferry:
            if active_task == []:
                d[robot_index] = 0
                time[robot_index] = time[robot_index] + dist[load[robot_index][-1],2*n]/speed
                finish_time[robot_index] = finish_time[robot_index] + dist[load[robot_index][-1],2*n]/speed
                travel_dist[robot_index] = travel_dist[robot_index] + dist[load[robot_index][-1],2*n]
                load[robot_index].append(2*n)
                continue
            else:
                avail_task,unavail,time,capacity,load = forcefinish(robot_index,n,m,load,active_task,avail_task,unavail,dist,capacity,deadline,time)
                continue
        
        
        next_task,avail_task,unavail,time,capacity,load = feasibleTask(robot_index,n,m,load,active_task,avail_task,unavail,dist,capacity,deadline,time)
    
    
    for i in range(m):
        if load[i][-1] != 2*n:
            load[i].append(2*n)
            finish_time[robot_index] = finish_time[robot_index] + dist[load[robot_index][-1],2*n]/speed
            travel_dist[robot_index] = travel_dist[robot_index] + dist[load[robot_index][-1],2*n]
    
    compl = len([i for i in e if i == 0])
    compl = int(compl/2)
    complete_rate = compl/n*100
    # uncompl = len([i for i in e if i > 0])
    # uncompl = np.ceil(uncompl/2)
    # complete_rate = (n-uncompl)/n*100
    computing_time = ct.time() - start
    #print('Completion Rate is {} %'.format(complete_rate))
    #print('Computation Time %s seconds' % (ct.time() - start))
    return complete_rate,travel_dist,finish_time,load,computing_time