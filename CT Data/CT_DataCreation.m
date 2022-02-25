n = 30;
nCase = 10;
for ii = 1:nCase
    x = randi([1,1000],[n 1]);
    y = randi([1,1000],[n 1]);
    
    depot = randi([1,1000],[2 1]);
    
    deadline_pickup = randi([1000 2000],[n/2 1]);
    deadline_delivery = randi([3000 3600],[n/2 1]);
    deadline = [deadline_pickup;deadline_delivery];
    
    
%     q = randi([1,2],[n/2 1]);
%     q(n/4+1:n/2) = 3;
%     rand_index = randperm(n/2);
%     q = q(rand_index);
    q = 3*ones(n/2,1);
    q = [q;-q];
%     e = randi([1,2],[n/2 1]);
%     e = [e;e];
%     
%     q = [3]
% 
%     rand_index = randperm(n/2);
%      e = ones(n,1);
    e = zeros(n/2,1);
    for i = 1:length(q)/2
        
        if q(i) <= 2
            e(i) = 1;
        else
            e(i) = 3;
        end
        
    end
    e = [e;e];
    
    taskdata = [x,y,q,e,deadline];
    
    save(['30T_3RT_run_' num2str(ii) '.mat'],'taskdata','depot')
    
end