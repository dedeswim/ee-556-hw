function err = compute_error(A_test,b_test,xopt)
n = size(A_test,1); err = 0;
for i = 1:n
   if b_test(i)*(A_test(i,:)*xopt) <= 0
       err = err + 1;
   end
end
err = err/n;
end
