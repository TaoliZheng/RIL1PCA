function [ev,pc] = L2PCA_approx(ev_prev,pc_prev,K,X_diff)
ev=ev_prev; pc=pc_prev;
for k=1:K
    ev(k) = ev_prev(k)+ pc_prev(:,k)'*X_diff*pc_prev(:,k);
    for k2 = 1:K
        if k~=k2
            pc(:,k) = pc(:,k)+ ((pc_prev(:,k2)'*X_diff*pc_prev(:,k2))/(ev_prev(k)-ev_prev(k2))*pc_prev(:,k2));
        end
    end
end

            