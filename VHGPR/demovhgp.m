load motorcycle

[NMSE, NLPD, Ey, Vy, mutst, diagSigmatst, atst, diagCtst, LambdaTheta, convergence] = ... 
    vhgpr_ui(X, y, X, y,100);

figure
plotvarianza(X, mutst, diagSigmatst)
hold on
plot(X, mutst,'r')

figure
plotvarianza(X, Ey, Vy)
hold on
plot(X, y,'xb', X, Ey,'r')
