import numpy as np
import random
from matplotlib import pyplot as plt
import itertools
import time as ct
from Auction_CT_NonSim_noBox import Auction
from Feas_Random_Walk import RND_Walk
import scipy.io
#import os


# dist = np.delete(dist,0,0)
# dist = np.delete(dist,0,1)
# lend = np.shape(dist)
# for i in range(lend[0]):
#     dist[i,i] = 0

# output_dir = "Results_feas_rnd"
# # If folder doesn't exist, then create it.
# if not os.path.isdir(output_dir):
#     os.makedirs(output_dir)
    
case_num = 1

complete = [[] for i in range(case_num)]
computetime = [[] for i in range(case_num)]
travel = [[] for i in range(case_num)]
finish = [[] for i in range(case_num)]

complete2 = [[] for i in range(case_num)]
computetime2 = [[] for i in range(case_num)]
travel2 = [[] for i in range(case_num)]
finish2 = [[] for i in range(case_num)]

modelName = '200T_2RT_run_' # Partial name of the MATLAB data file

for iii in range(case_num):
    
    iCase = iii+1
    
    data = scipy.io.loadmat("CT Data/"+modelName+str(iCase)+".mat")
    taskData = data['taskdata']
    taskLocation = taskData[:,:2]
    X = taskData[:,0]
    Y = taskData[:,1]
    q = taskData[:,2]
    e = taskData[:,3]
    deadline = taskData[:,-1]
    depot = data['depot']
    X = np.append(X,depot[0])
    Y = np.append(Y,depot[1])

    
    # Number of Tasks
    n = 100 # This number needs to divided by 2 based on the first number in the MATLAB data file name. Ex: '200T_2RT_run_1' will have n=200/2=100
    
    # Number of Robots
    m = 20
    
    # Robot's moving speed (2.778)
    speed = 2.778
    
    
    # Index for starting points and ending points 
    pickup = list(range(n))
    delivery = list(range(n,2*n))
    
    # Deadlines for each tasks
    # deadline = np.zeros(2*n)
    # deadline[0:n] = np.array([(random.randint(1000,2000)) for i in range(n)])  #510 1700
    # deadline[n:2*n] = np.array([(random.randint(3000,3600)) for i in range(n)]) #2200 3600
    timemax = 3600
    
    # Robots' ferry range
    ferry = 4000
    
    # Max number of tasks for a robot
    #L = int(n/m+1)
    
    # Max capacity of robot
    C = 4
    
    # Workload of task
    # q = [random.randint(1,3) for i in range(int(n))]
    # e = []
    # for i in range(len(q)):
    #     q.append(-q[i])
    #     if q[i] <= 2: #2
    #         e.append(1) #1
    #     else:
    #         e.append(2)    
    
    # for i in range(len(e)):
    #     e.append(e[i])
        
    ee = e.copy()
    
    
    # Get random locations
    # X = [(random.randint(0,1000)) for i in range(2*n+1)] 
    # Y = [(random.randint(0,1000)) for i in range(2*n+1)]
    
    # Diagonal distance (corner to corner)
    D = 1000*2**0.5 # 1000*2**0.5
    
    location = np.column_stack((X,Y))
    
    # Get Euclidean distance
    depot, tasks = location[-1, :], location[0:2*n, :]
    
    dist = np.empty((2*n+1,2*n+1))
    for i in range(2*n+1):
        for j in range(2*n+1):
            dist[i,j] = np.sqrt((X[i]-X[j])**2 + (Y[i]-Y[j])**2)
            
    # Record robots' finish time
    finish_time = np.empty(m)
    
    # Available tasks
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

    
    
    complete[iii],travel[iii],finish[iii],load,computetime[iii] = Auction(n,m,location,speed,deadline,q,C,e,ee,D,timemax,ferry,dist)
    complete2[iii],travel2[iii],finish2[iii],load2,computetime2[iii] = RND_Walk(n,m,location,speed,deadline,q,C,e,ee,D,timemax,ferry,dist)
    
    
average_complete = sum(complete)/case_num
average_computetime = sum(computetime)/case_num
avg_completetime = sum(np.max(finish,axis=1))/case_num
#avg_dist = sum(np.max(travel,axis=1))/case_num
avg_dist = np.sum(travel)/case_num
std_complete = np.std(complete)
std_computetime = np.std(computetime)
std_completetime = np.std(finish)
std_dist = np.std(travel)

average_complete2 = sum(complete2)/case_num
average_computetime2 = sum(computetime2)/case_num
avg_completetime2 = sum(np.max(finish2,axis=1))/case_num
#avg_dist2 = sum(np.max(travel2,axis=1))/case_num
avg_dist2 = np.sum(travel)/case_num
std_complete2 = np.std(complete2)
std_computetime2 = np.std(computetime2)
std_completetime2 = np.std(finish2)
std_dist2 = np.std(travel2)


#avg_better_time = (avg_completetime - avg_completetime2)/avg_completetime
#avg_better_dist = (avg_dist - avg_dist2)/avg_dist



# fig = plt.figure(figsize =(10, 7))
# ax = fig.add_axes([0, 1])
# ax.set_xticklabels(['AuctionBox', 'NoAuctionBox'])
# plt.title("Auction Algorithm box plot")


#plt.boxplot([complete,complete3])

# O = list(range(1,int(n)+1))

# # Ending point location index
# D = list(range(int(n)+1,2*n+1))

# def plot_tours(solution_x):
    
    
#     tours = [[i, j] for i in range(solution_x.shape[0]) for j in range(solution_x.shape[1]) if solution_x[i, j] ==1]
#     for tour in tours:
#         plt.plot([X[tour[0]],X[tour[1]]], [Y[tour[0]],Y[tour[1]]], color = "black", linewidth=0.5)
    
#     array_tours = np.array(tours)
#     route = [0]
#     next_index = 0
#     for i in range(len(tours)):
#         next_point = int(array_tours[next_index,1])
#         next_index = np.argwhere(array_tours[:,0] == next_point)
#         route.append(next_point)
        
        
#     tour_x = []
#     tour_y = []
#     for i in route:
#         tour_x.append(X[i])
#         tour_y.append(Y[i])
    
#     u = np.diff(tour_x)
#     v = np.diff(tour_y)
#     pos_x = tour_x[:-1]+u/2
#     pos_y = tour_y[:-1]+v/2
#     norm = np.sqrt(u**2+v**2)
    
#     colors = ['r','orange','g','b','c','m','y','k','blueviolet','lawngreen']#,'m', 'y', 'k']
#     for ii in range(len(O)):
#         if e[ii] > 1:
#             plt.scatter(X[O[ii]],Y[O[ii]],marker='^',color=colors[ii],s=70,label = 'Staring Points')
#             plt.scatter(X[D[ii]],Y[D[ii]],marker=',',color=colors[ii],s=70,label = 'Ending Points')
#         else:
#             plt.scatter(X[O[ii]],Y[O[ii]],marker='^',color=colors[ii],s=10,label = 'Staring Points')
#             plt.scatter(X[D[ii]],Y[D[ii]],marker=',',color=colors[ii],s=10,label = 'Ending Points')
#     plt.scatter(depot[0],depot[1],s=80,marker='*',label='Depot')
#     plt.xlabel("X"), plt.ylabel("Y"), plt.title("Tours"),plt.legend(bbox_to_anchor=(1.05, 1))
#     plt.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy", zorder=5, pivot="mid")
#     plt.show()


# def plot_tours3(X_sol11,X_sol21,X_sol31):
    
#     #,X_sol12,X_sol22,X_sol32
#     tours1 = [[i, j] for i in range(X_sol11.shape[0]) for j in range(X_sol11.shape[1]) if X_sol11[i, j] ==1]
#     tours2 = [[i, j] for i in range(X_sol21.shape[0]) for j in range(X_sol21.shape[1]) if X_sol21[i, j] ==1]
#     tours3 = [[i, j] for i in range(X_sol31.shape[0]) for j in range(X_sol31.shape[1]) if X_sol31[i, j] ==1]
    
#     for tour in tours1:
#         if tour == tours1[-1]:
#             plt.plot([X[tour[0]],X[tour[1]]], [Y[tour[0]],Y[tour[1]]], color = "red", alpha=0.5,linewidth=1,label='Robot1')
#         else:
#             plt.plot([X[tour[0]],X[tour[1]]], [Y[tour[0]],Y[tour[1]]], color = "red", alpha=0.5,linewidth=1)
    
#     for tour in tours2:
#         if tour == tours2[-1]:
#             plt.plot([X[tour[0]],X[tour[1]]], [Y[tour[0]],Y[tour[1]]], color = "lime", alpha=0.6,linewidth=3,label='Robot2')
#         else:
#             plt.plot([X[tour[0]],X[tour[1]]], [Y[tour[0]],Y[tour[1]]], color = "lime", alpha=0.6,linewidth=3)
        
#     for tour in tours3:
#         if tour == tours3[-1]:
#             plt.plot([X[tour[0]],X[tour[1]]], [Y[tour[0]],Y[tour[1]]], color = "navy", alpha=0.3,linewidth=5,label='Robot3')
#         else:
#             plt.plot([X[tour[0]],X[tour[1]]], [Y[tour[0]],Y[tour[1]]], color = "navy", alpha=0.3,linewidth=5)
    
#     plt.legend(bbox_to_anchor=(1.05, 1))
#     # for tour in tours4:
#     #     plt.plot([X[tour[0]],X[tour[1]]], [Y[tour[0]],Y[tour[1]]], color = "red", alpha=0.15,linewidth=0.5)
        
#     # for tour in tours5:
#     #     plt.plot([X[tour[0]],X[tour[1]]], [Y[tour[0]],Y[tour[1]]], color = "green", alpha=0.15,linewidth=0.5)
        
#     # for tour in tours6:
#     #     plt.plot([X[tour[0]],X[tour[1]]], [Y[tour[0]],Y[tour[1]]], color = "blue", alpha=0.15,linewidth=0.5)
    
#     def arrows(tours):
#         array_tours = np.array(tours)
#         route = [0]
#         next_index = 0
#         for i in range(len(tours)):
#             next_point = int(array_tours[next_index,1])
#             next_index = np.argwhere(array_tours[:,0] == next_point)
#             route.append(next_point)
            
            
#         tour_x = []
#         tour_y = []
#         for i in route:
#             tour_x.append(X[i])
#             tour_y.append(Y[i])
        
#         u = np.diff(tour_x)
#         v = np.diff(tour_y)
#         pos_x = tour_x[:-1]+u/2
#         pos_y = tour_y[:-1]+v/2
#         norm = np.sqrt(u**2+v**2)
        
#         plt.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy", zorder=5, pivot="mid")
    
#     colors = ['r','orange','g','b','c','m','y','k','blueviolet','lawngreen']#,'m', 'y', 'k']    
#     for ii in range(len(O)):
#         if ee[ii] > 1:
#             plt.scatter(X[O[ii]],Y[O[ii]],marker='^',color=colors[ii],s=70,label = 'Pickup Task')
#             plt.scatter(X[D[ii]],Y[D[ii]],marker=',',color=colors[ii],s=70,label = 'Delivery Task')
#         else:
#             plt.scatter(X[O[ii]],Y[O[ii]],marker='^',color=colors[ii],s=25,label = 'Pickup Task')
#             plt.scatter(X[D[ii]],Y[D[ii]],marker=',',color=colors[ii],s=25,label = 'Delivery Task')
#     plt.scatter(depot[0],depot[1],s=100,marker='*',label='Depot')
#     plt.xlabel("X"), plt.ylabel("Y"), plt.title("Auction (Total Traveling Distance = 8911.56m)"),plt.legend(bbox_to_anchor=(1.05, 1))    
#     arrows(tours1)
#     arrows(tours3)
#     arrows(tours2)
#     #arrows(tours3)
#     plt.show()


# x1 = load[0]
# x2 = load[1]
# x3 = load[2]

# n = 2*n
# X_sol11 = np.empty([n+2, n+2])
# #X_sol12 = np.empty([n+1, n+1])
# X_sol21 = np.empty([n+2, n+2])
# #X_sol22 = np.empty([n+1, n+1])
# X_sol31 = np.empty([n+2, n+2])
# #X_sol32 = np.empty([n+1, n+1])

# # X = np.insert(X,0,depot[0])
# # Y = np.insert(Y,0,depot[1])


# for i in range(len(x1)-1):
#     if i == 0:
#         x = 0
#         y = x1[i+1]+1
#     else:
#         x = x1[i]+1
#         y = x1[i+1]+1
    
#     X_sol11[x, y] = 1
#     #X_sol12[i, j] = round(solx12[index])
#     #X_sol21[i, j] = round(solx21[index])
#     #X_sol22[i, j] = round(solx22[index])
#     #X_sol31[i, j] = round(solx31[index])
#     #X_sol32[i, j] = round(solx32[index])
        
# for i in range(len(x2)-1):
#     if i == 0:
#         x = 0
#         y = x2[i+1]+1
#     else:
#         x = x2[i]+1
#         y = x2[i+1]+1
    

#     X_sol21[x, y] = 1
        
# for i in range(len(x3)-1):
#     if i == 0:
#         x = 0
#         y = x3[i+1]+1
#     else:
#         x = x3[i]+1
#         y = x3[i+1]+1

#     X_sol31[x, y] = 1

# plot_tours3(X_sol11,X_sol21,X_sol31)

