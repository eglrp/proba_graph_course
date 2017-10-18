M = dlmread('classificationC.train');
X=M(:,1:2);
Y=M(:,3);
M = dlmread('classificationC.test');
Xt=M(:,1:2);
Yt=M(:,3);

lw=2; %Linewidth
fs=18; %Fontsize
fw='Bold'; %FontWeight
fsa=16; %Fontsize

n0 = length(find(Y==0));
n1 = length(find(Y==1));
n = size(Y,1);
p = n1/n;
X0s = X(Y==0,:);
X1s = X(Y==1,:);



%%
figure(1);
scatter(X0s(:,1),X0s(:,2),'r');
hold on;
scatter(X1s(:,1),X1s(:,2),'b');
hold on;
axis([-8 8 -8 8]);


[w,b] = LDA(X,Y);
Yp = predict(X,w,b);
errorLDA = length(find(Yp~=Y));
x1 = linspace(-5,5,10);
x2 = (-b-w(1)*x1)/w(2);
plot(x1,x2,':r','Linewidth',lw);

[w,b] = LogisticR(X,Y,w,b);
Yp = predict(X,w,b);
errorLogisticR = length(find(Yp~=Y));
x1 = linspace(-5,5,10);
x2 = (-b-w(1)*x1)/w(2);
plot(x1,x2,'m','Linewidth',lw);

[w,b] = LinearR(X,Y);
Yp = predict(X,w,b);
errorLinearR = length(find(Yp~=Y));
x1 = linspace(-5,5,10);
x2 = (-b-w(1)*x1)/w(2);
plot(x1,x2,'--k','Linewidth',lw);

title('Training on class C','FontSize',fs,'FontWeight',fw);
legend('sample (x,y=0)','sample (x,y=1)','LDA','LogisticR','LinearR');
print('-depsc','plot_c');

fprintf('train error: LDA %d, Logistic %d, Linear %d, total %d\n',errorLDA,errorLogisticR,errorLinearR,length(Y));
%%
[w,b] = LDA(X,Y);
Yp = predict(Xt,w,b);
errorLDA = length(find(Yp~=Yt));
[w,b] = LogisticR(X,Y,w,b);
Yp = predict(Xt,w,b);
errorLogisticR = length(find(Yp~=Yt));
[w,b] = LinearR(X,Y);
Yp = predict(Xt,w,b);
errorLinearR = length(find(Yp~=Yt));
fprintf('test error: LDA %d, Logistic %d, Linear %d, total %d\n',errorLDA,errorLogisticR,errorLinearR,length(Yt));

%%
% QDA
[mu0,mu1,sigma0,sigma1,p] = QDA(X,Y);
Yp = QDA_predict(X,mu0,mu1,sigma0,sigma1,p);
errorQDA = length(find(Yp~=Y));
fprintf('train error: QDA %d, total %d\n',errorQDA,length(Y));

[g1,g2] = meshgrid(-8:0.05:8);
points = [g1(:),g2(:)];
pointsY = QDA_predict(points,mu0,mu1,sigma0,sigma1,p);
pointsY = reshape(pointsY,size(g1,1),size(g1,2));
left_1 = g1(pointsY<0.5);
left_2 = g2(pointsY<0.5);
right_1 = g1(pointsY>0.5);
right_2 = g2(pointsY>0.5);
figure(1);
scatter(left_1,left_2,'y','filled');
hold on;
scatter(right_1,right_2,'c','filled');
scatter(X0s(:,1),X0s(:,2),'r');
scatter(X1s(:,1),X1s(:,2),'b');
axis([-8 8 -8 8]);
title('QDA trained on class C','FontSize',fs,'FontWeight',fw);
print('-depsc','qda_c');

[mu0,mu1,sigma0,sigma1,p] = QDA(X,Y);
Yp = QDA_predict(Xt,mu0,mu1,sigma0,sigma1,p);
errorQDA = length(find(Yp~=Yt));
fprintf('test error: QDA %d, total %d\n',errorQDA,length(Yt));

